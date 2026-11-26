// 合成结果与真值的评测：PSNR 增益、深度误差（步距单位）、刚毛光晕宽度、验收
import type { FusionMetrics, SyntheticSpec } from '../types';
import { clamp, median, psnr, robustLineFit } from './math';
import type { SyntheticStack } from '../synthetic/generator';
import type { FusionResult } from './cpuFusion';

/** 单层"最清晰"参考：逐像素取与真值颜色最接近的层（理想选择器上界） */
export function bestSingleLayerOracle(stack: SyntheticStack, mask?: Uint8Array): { image: Float32Array; psnr: number } {
  const { width: w, height: h, images, colorTruth } = stack;
  const useMask = mask ?? stack.bodyMask;
  const out = new Float32Array(w * h * 3);
  // 逐层整体 PSNR 里最高的那层（"只取最清晰单层"的公平对照：整栈选一层）
  let best = 0, bestPsnr = -Infinity;
  const flat = new Float32Array(w * h * 3);
  for (let z = 0; z < images.length; z++) {
    const ch = images[z].length / (w * h);
    for (let p = 0; p < w * h; p++) {
      flat[p * 3] = images[z][p * ch];
      flat[p * 3 + 1] = images[z][p * ch + 1];
      flat[p * 3 + 2] = images[z][p * ch + 2];
    }
    const v = psnrFlat(flat, colorTruth, useMask);
    if (v > bestPsnr) { bestPsnr = v; best = z; }
  }
  const ch = images[best].length / (w * h);
  for (let p = 0; p < w * h; p++) {
    out[p * 3] = images[best][p * ch];
    out[p * 3 + 1] = images[best][p * ch + 1];
    out[p * 3 + 2] = images[best][p * ch + 2];
  }
  return { image: out, psnr: bestPsnr };
}

export function psnrFlat(a: Float32Array, b: Float32Array, mask: Uint8Array): number {
  let se = 0, n = 0;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p]) continue;
    for (let c = 0; c < 3; c++) {
      const d = a[p * 3 + c] - b[p * 3 + c];
      se += d * d; n++;
    }
  }
  if (!n) return NaN;
  const mse = se / n;
  return mse < 1e-12 ? 99 : 10 * Math.log10(65025 / mse);
}

/**
 * 深度误差（步距单位）。
 * 深度索引与真值高度可能存在单调仿射偏差（对焦方向约定、层间偏移），
 * 先做鲁棒线性拟合 d_est = a*d_true + b，再统计残差中位数。
 */
export function depthErrorSteps(
  result: FusionResult, truth: Float32Array, bodyMask: Uint8Array,
  excludeLowTexture = true, excludeSpecular = true,
  lowTex?: Uint8Array, specular?: Uint8Array,
): { medianAbs: number; a: number; b: number; usedRatio: number } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let p = 0; p < bodyMask.length; p++) {
    if (!bodyMask[p]) continue;
    if (excludeLowTexture && lowTex?.[p]) continue;
    if (excludeSpecular && specular?.[p]) continue;
    xs.push(truth[p]);
    ys.push(result.depth[p]);
  }
  const { a, b } = robustLineFit(Float64Array.from(xs), Float64Array.from(ys));
  const resid = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) resid[i] = Math.abs(ys[i] - (a * xs[i] + b));
  return { medianAbs: median(resid), a, b, usedRatio: xs.length / bodyMask.filter(Boolean).length };
}

/**
 * 刚毛边缘光晕宽度：对每个刚毛边缘像素，沿梯度法向扫描，
 * 统计合成图与全清晰真值的亮度差超过阈值（12/255）的连续像素数，取中位数。
 */
export function haloWidth(
  fused: Float32Array, truth: Float32Array, setaeEdge: Uint8Array, w: number, h: number,
): number {
  const lum = (rgb: Float32Array, p: number) =>
    0.299 * rgb[p * 3] + 0.587 * rgb[p * 3 + 1] + 0.114 * rgb[p * 3 + 2];
  const widths: number[] = [];
  const THRESH = 12;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      if (!setaeEdge[p]) continue;
      // 简单法向：用真值亮度梯度方向
      const gx = lum(truth, p + 1) - lum(truth, p - 1);
      const gy = lum(truth, p + w) - lum(truth, p - w);
      const L = Math.hypot(gx, gy) || 1;
      const nx = gx / L, ny = gy / L;
      let span = 0;
      for (const dir of [-1, 1]) {
        for (let step = 1; step <= 6; step++) {
          const xx = Math.round(x + nx * step * dir);
          const yy = Math.round(y + ny * step * dir);
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) break;
          const q = yy * w + xx;
          if (Math.abs(lum(fused, q) - lum(truth, q)) > THRESH) span++;
          else break;
        }
      }
      widths.push(span);
    }
  }
  return widths.length ? median(Float64Array.from(widths)) : 0;
}

export interface AcceptanceTargets {
  psnrGainDb: number;
  depthErrorStep: number;
  haloPx: number;
}

export function evaluate(
  stack: SyntheticStack,
  result: FusionResult,
  targets: AcceptanceTargets,
): FusionMetrics {
  const reasons: string[] = [];
  // 评测掩膜：虫体且高置信（排除低纹理/高光/半透明翅——
  // 翅是多层信号区，本就不参与"硬锐利度"评测；低纹理/高光无可靠焦点信号）
  const evalMask = new Uint8Array(stack.bodyMask.length);
  let evalCount = 0;
  for (let p = 0; p < stack.bodyMask.length; p++) {
    if (stack.bodyMask[p] && !result.lowTexture[p] && !result.specular[p] && !stack.wingMask[p]) {
      evalMask[p] = 1; evalCount++;
    }
  }
  if (evalCount < stack.bodyMask.length * 0.2) evalMask.set(stack.bodyMask); // 兜底：置信区过小则用整体

  const fusedPsnr = psnrFlat(result.color, stack.colorTruth, evalMask);
  const oracle = bestSingleLayerOracle(stack, evalMask);
  const gain = fusedPsnr - oracle.psnr;

  const de = depthErrorSteps(
    result, stack.depthTruth, stack.bodyMask, true, true,
    result.lowTexture, result.specular,
  );
  const halo = haloWidth(result.color, stack.colorTruth, stack.setaeEdge, stack.width, stack.height);

  let lowRatio = 0, specRatio = 0, holeRatio = 0;
  let bodyCount = 0;
  for (let p = 0; p < stack.bodyMask.length; p++) {
    if (!stack.bodyMask[p]) continue;
    bodyCount++;
    if (result.lowTexture[p]) lowRatio++;
    if (result.specular[p]) specRatio++;
    if (result.holeMask[p]) holeRatio++;
  }
  lowRatio /= bodyCount; specRatio /= bodyCount; holeRatio /= bodyCount;

  if (gain < targets.psnrGainDb) reasons.push(`PSNR 增益 ${gain.toFixed(2)}dB < 目标 ${targets.psnrGainDb}dB`);
  if (de.medianAbs > targets.depthErrorStep) reasons.push(`深度中位误差 ${de.medianAbs.toFixed(2)} 步距 > 目标 ${targets.depthErrorStep}`);
  if (halo > targets.haloPx) reasons.push(`光晕宽度中位 ${halo.toFixed(2)}px > 目标 ${targets.haloPx}px`);

  return {
    fusedPsnrDb: fusedPsnr,
    bestSinglePsnrDb: oracle.psnr,
    gainDb: gain,
    depthMedianAbsErrStep: de.medianAbs,
    haloWidthPxMedian: halo,
    lowTextureMaskRatio: lowRatio,
    specularMaskRatio: specRatio,
    holeFilledRatio: holeRatio,
    pass: reasons.length === 0,
    reasons,
  };
}

export function defaultTargets(spec: SyntheticSpec): AcceptanceTargets {
  void spec;
  return { psnrGainDb: 6, depthErrorStep: 1, haloPx: 2 };
}

export { clamp };
