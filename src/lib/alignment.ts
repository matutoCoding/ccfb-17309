// 对齐：逐层估计缩放+平移（变焦呼吸/导轨晃动校正），残差曲线与控制点重配
// 纯 TS 参考实现；GPU 管线在着色器中复用同一套数学（见 gpu/shaders）
import { clamp, fitSimilarity, median, stddev } from './math';
import { boxMean } from './alignUtils';
import type { AlignParams, AlignResidual, ControlPoint } from '../types';

export interface AlignOptions {
  coarseGrid: number; // 粗匹配网格步长
  searchRadius: number; // 平移搜索半径（像素）
  scaleSteps: number; // 缩放搜索档数
  scaleRangePct: number; // ±百分比
  residualThreshold: number; // 残差超阈值倍数（相对中位）
}

export const DEFAULT_ALIGN: AlignOptions = {
  coarseGrid: 4,
  searchRadius: 10,
  scaleSteps: 7,
  scaleRangePct: 3,
  residualThreshold: 2.5,
};

function luminance(rgba: Float32Array, w: number, h: number): Float32Array {
  const ch = rgba.length / (w * h);
  const lum = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    lum[p] = 0.299 * rgba[p * ch] + 0.587 * rgba[p * ch + 1] + 0.114 * rgba[p * ch + 2];
  }
  return lum;
}

/** 归一化 SSD（越低越匹配），带 stride 抽样。
 *  变换约定与 warpLayer 一致：输出坐标 dx 反查规范坐标 sx = cx + (dx - tx)/scale */
function ssdAt(
  ref: Float32Array, cur: Float32Array, w: number, h: number,
  scale: number, tx: number, ty: number, stride: number,
): number {
  const cx = w / 2, cy = h / 2;
  let se = 0, n = 0;
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const dx = x - cx, dy = y - cy;
      const sx = cx + (dx - tx) / scale;
      const sy = cy + (dy - ty) / scale;
      const xi = sx | 0, yi = sy | 0;
      if (xi < 1 || yi < 1 || xi >= w - 1 || yi >= h - 1) continue;
      // 双线性
      const fx = sx - xi, fy = sy - yi;
      const i00 = yi * w + xi, i10 = i00 + 1, i01 = i00 + w, i11 = i01 + 1;
      const v = cur[i00] * (1 - fx) * (1 - fy) + cur[i10] * fx * (1 - fy)
        + cur[i01] * (1 - fx) * fy + cur[i11] * fx * fy;
      const d = ref[y * w + x] - v;
      se += d * d; n++;
    }
  }
  return n ? se / n : Infinity;
}

/** 归一化互相关（NCC）在给定搜索范围内的最佳位移。
 *  返回把 cur 对齐到 ref 的位移 (tx,ty)：ref(x) ≈ cur(x - tx)，即 cur 右移时 tx 为负 */
function nccShift(
  ref: Float32Array, cur: Float32Array, w: number, h: number,
  search: number,
): { tx: number; ty: number; peak: number } {
  const mr = meanOf(ref), mc = meanOf(cur);
  let best = -Infinity, bx = 0, by = 0;
  for (let dy = -search; dy <= search; dy++) {
    for (let dx = -search; dx <= search; dx++) {
      let num = 0, dr = 0, dc = 0;
      for (let y = search + 1; y < h - search - 1; y++) {
        for (let x = search + 1; x < w - search - 1; x++) {
          const sx = x - dx, sy = y - dy;
          const a = ref[y * w + x] - mr;
          const b = cur[sy * w + sx] - mc;
          num += a * b; dr += a * a; dc += b * b;
        }
      }
      const score = num / Math.sqrt(dr * dc || 1);
      if (score > best) { best = score; bx = dx; by = dy; }
    }
  }
  return { tx: bx, ty: by, peak: best };
}

function meanOf(a: Float32Array): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length;
}

function halfSize(src: Float32Array, w: number, h: number): { out: Float32Array; w: number; h: number } {
  const nw = Math.max(8, w >> 1), nh = Math.max(8, h >> 1);
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    const sx = Math.min(w - 1, x * 2), sy = Math.min(h - 1, y * 2);
    let acc = 0, n = 0;
    for (let dy = 0; dy < 2 && sy + dy < h; dy++) for (let dx = 0; dx < 2 && sx + dx < w; dx++) { acc += src[(sy + dy) * w + sx + dx]; n++; }
    out[y * nw + x] = acc / n;
  }
  return { out, w: nw, h: nh };
}

/**
 * 估计单层相对参考层的相似变换（金字塔 NCC，抗离焦/亮度差）。
 *  返回 warpLayer 所需参数：把当前层规范坐标映射到参考层坐标。
 *  关键：跨离焦层内容差异主导 NCC，若匹配峰不够锐（不可靠），回退单位变换，
 *  避免用错误配准把本来对齐的层扭曲（近焦层尤其要保护）。
 */
export function estimateLayerAlign(
  refRgba: Float32Array, curRgba: Float32Array, w: number, h: number,
  opts: AlignOptions = DEFAULT_ALIGN,
): AlignParams {
  const refFull = boxMean(luminance(refRgba, w, h), w, h, 1);
  const curFull = boxMean(luminance(curRgba, w, h), w, h, 1);

  // 粗级（半分辨率）
  const coarseR = halfSize(refFull, w, h);
  const coarseC = halfSize(curFull, w, h);
  const coarseSearch = Math.ceil(opts.searchRadius / 2) + 1;
  const coarse = nccShift(coarseR.out, coarseC.out, coarseR.w, coarseR.h, coarseSearch);
  // 峰锐度门控：NCC 峰值过低说明两层焦平面差异大、配准不可靠
  if (coarse.peak < 0.35) {
    return { scale: 1, angle: 0, tx: 0, ty: 0 };
  }
  let tx = coarse.tx * 2, ty = coarse.ty * 2;
  let bestPeak = -Infinity;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const cand = nccShiftLimited(refFull, curFull, w, h, tx + dx, ty + dy);
    if (cand > bestPeak) { bestPeak = cand; tx = tx + dx; ty = ty + dy; }
  }
  if (bestPeak < 0.3) return { scale: 1, angle: 0, tx: 0, ty: 0 };

  // 向 0 正则收缩：跨离焦层 NCC 易有系统偏差，限制最大校正幅度，保护近焦层不被扭曲
  const maxShift = opts.searchRadius;
  tx = clamp(tx, -maxShift, maxShift) * 0.7;
  ty = clamp(ty, -maxShift, maxShift) * 0.7;

  // 缩放在呼吸确实存在时才小幅修正，默认贴近 1.0
  let scale = 1;
  if (coarse.peak > 0.5) {
    let scalePeak = coarse.peak;
    for (let i = 0; i < opts.scaleSteps; i++) {
      const t = opts.scaleSteps === 1 ? 0 : i / (opts.scaleSteps - 1) * 2 - 1;
      const s = 1 + t * opts.scaleRangePct / 100;
      if (Math.abs(s - 1) < 1e-6) continue;
      const scaled = resampleBilinear(coarseC.out, coarseR.w, coarseR.h, s);
      const pc = nccShift(coarseR.out, scaled, coarseR.w, coarseR.h, coarseSearch);
      if (pc.peak > scalePeak) { scalePeak = pc.peak; scale = s; }
    }
    // 缩放也向 1 收缩
    scale = 1 + (scale - 1) * 0.5;
  }
  return { scale, angle: 0, tx, ty };
}

/** 固定位移下的 NCC 分数（用于细级局部比较） */
function nccShiftLimited(ref: Float32Array, cur: Float32Array, w: number, h: number, dx: number, dy: number): number {
  const mr = meanOf(ref), mc = meanOf(cur);
  const m = 4;
  let num = 0, dr = 0, dc = 0;
  for (let y = m; y < h - m; y++) for (let x = m; x < w - m; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
    const a = ref[y * w + x] - mr, b = cur[sy * w + sx] - mc;
    num += a * b; dr += a * a; dc += b * b;
  }
  return num / Math.sqrt(dr * dc || 1);
}

/** 以图像中心为原点的缩放重采样（亮度图） */
function resampleBilinear(src: Float32Array, w: number, h: number, scale: number): Float32Array {
  const out = new Float32Array(w * h);
  const cx = w / 2, cy = h / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = cx + (x - cx) / scale, sy = cy + (y - cy) / scale;
    const x0 = sx | 0, y0 = sy | 0;
    if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) { out[y * w + x] = src[clamp(y0, 0, h - 1) * w + clamp(x0, 0, w - 1)]; continue; }
    const fx = sx - x0, fy = sy - y0;
    const i00 = y0 * w + x0;
    out[y * w + x] = src[i00] * (1 - fx) * (1 - fy) + src[i00 + 1] * fx * (1 - fy)
      + src[i00 + w] * (1 - fx) * fy + src[i00 + w + 1] * fx * fy;
  }
  return out;
}

/** 全栈对齐：以中间层为参考，返回每层参数与残差曲线数据 */
export function alignStack(
  images: Float32Array[], w: number, h: number,
  opts: AlignOptions = DEFAULT_ALIGN,
  controlPoints: ControlPoint[] = [],
): { params: AlignParams[]; residuals: AlignResidual[] } {
  const N = images.length;
  const refIdx = N >> 1;
  const params: AlignParams[] = [];
  const rawSsd: number[] = [];
  for (let z = 0; z < N; z++) {
    if (z === refIdx) {
      params.push({ scale: 1, angle: 0, tx: 0, ty: 0 });
      rawSsd.push(0);
      continue;
    }
    const p = estimateLayerAlign(images[refIdx], images[z], w, h, opts);
    params.push(p);
    rawSsd.push(ssdAt(
      boxMean(luminance(images[refIdx], w, h), w, h, 1),
      boxMean(luminance(images[z], w, h), w, h, 1),
      w, h, p.scale, p.tx, p.ty, 2,
    ));
  }

  // 控制点覆盖：用户手动配准的层直接用相似拟合替换
  const byLayer = new Map<number, ControlPoint[]>();
  for (const cp of controlPoints) {
    const arr = byLayer.get(cp.layerIndex) ?? [];
    arr.push(cp);
    byLayer.set(cp.layerIndex, arr);
  }
  for (const [layerIndex, cps] of byLayer) {
    if (cps.length < 2) continue;
    const fit = fitSimilarity(cps.map((c) => ({ ref: { x: c.refX, y: c.refY }, cur: { x: c.curX, y: c.curY } })));
    params[layerIndex] = fit;
  }

  const nonzero = rawSsd.filter((v) => v > 0);
  const med = median(nonzero);
  const sd = stddev(nonzero);
  const threshold = med + opts.residualThreshold * sd;
  const residuals: AlignResidual[] = params.map((p, i) => ({
    layerIndex: i,
    ssd: rawSsd[i],
    scale: p.scale,
    tx: p.tx,
    ty: p.ty,
    overThreshold: i !== refIdx && rawSsd[i] > threshold,
  }));
  return { params, residuals };
}

/** 应用变换重采样一层（双线性），输出与输入同尺寸 RGBA。
 *  变换约定（与 ssdAt 一致）：规范坐标 x_s 经 x_out = s*x_s + t 得到输出坐标；
 *  输出像素反查规范坐标 x_s = (x_out - t) / s。 */
export function warpLayer(
  rgba: Float32Array, w: number, h: number, p: AlignParams,
): Float32Array {
  const ch = rgba.length / (w * h);
  const out = new Float32Array(w * h * ch);
  const cx = w / 2, cy = h / 2;
  const cos = Math.cos(p.angle), sin = Math.sin(p.angle);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // 输出坐标 -> 平移逆 -> 旋转逆 -> 缩放逆 -> 规范坐标
      const dx = x - cx - p.tx, dy = y - cy - p.ty;
      const rx = cos * dx + sin * dy;
      const ry = -sin * dx + cos * dy;
      const sx = cx + rx / p.scale;
      const sy = cy + ry / p.scale;
      const xi = sx | 0, yi = sy | 0;
      const o = (y * w + x) * ch;
      if (xi < 0 || yi < 0 || xi >= w - 1 || yi >= h - 1) {
        for (let c = 0; c < ch; c++) out[o + c] = c === 3 ? 255 : 0;
        continue;
      }
      const fx = clamp(sx - xi, 0, 1), fy = clamp(sy - yi, 0, 1);
      const i00 = (yi * w + xi) * ch, i10 = i00 + ch, i01 = i00 + w * ch, i11 = i01 + ch;
      for (let c = 0; c < ch; c++) {
        out[o + c] = rgba[i00 + c] * (1 - fx) * (1 - fy) + rgba[i10 + c] * fx * (1 - fy)
          + rgba[i01 + c] * (1 - fx) * fy + rgba[i11 + c] * fx * fy;
      }
    }
  }
  return out;
}
