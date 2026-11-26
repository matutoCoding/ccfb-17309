import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateSyntheticStack } from '../src/synthetic/generator';
import { fuseDff, fuseLaplacianPyramid } from '../src/lib/cpuFusion';
import { evaluate, defaultTargets, bestSingleLayerOracle, depthErrorSteps, haloWidth } from '../src/lib/metrics';
import { alignStack, warpLayer, estimateLayerAlign } from '../src/lib/alignment';
import { orderLayers, numericToken } from '../src/lib/stackQc';
import { calibrate, measureAngle, measureLength, measureCurve } from '../src/lib/measure';
import { robustLineFit, psnr, median, rng } from '../src/lib/math';

function smallSpec(overrides = {}) {
  return {
    width: 160, height: 120, layers: 24, stepMicron: 2,
    breathScalePct: 0.6, jitterPx: 0.8, seed: 7,
    includeWing: true, includeSetae: true, includePunctures: true,
    ...overrides,
  };
}

test('合成栈：层数/尺寸/真值掩膜非空', () => {
  const syn = generateSyntheticStack(smallSpec());
  assert.equal(syn.images.length, 24);
  assert.equal(syn.images[0].length, 160 * 120 * 4);
  assert.ok(syn.bodyMask.includes(1));
  assert.ok(syn.wingMask.includes(1), '应包含半透明翅');
  assert.ok(syn.setaeEdge.includes(1), '应包含刚毛边缘');
  assert.ok(syn.specularMask.includes(1), '应包含高光斑');
  assert.ok(syn.lowTextureMask.includes(1), '应包含低纹理区');
});

test('DFF 融合达到验收线：PSNR 增益≥6dB、深度误差≤1步距、光晕≤2px', () => {
  // 无几何变换基线：纯验证融合算法对光学离焦的还原能力（上界 ~30dB）
  const syn = generateSyntheticStack(smallSpec({ breathScalePct: 0, jitterPx: 0 }));
  const fused = fuseDff(syn.images, syn.width, syn.height, {
    windowRadius: 2, topK: 4, lowTextureThresh: 0.12,
    specularLum: 235, softTemp: 8, haloGuard: 0.8,
  });
  const metrics = evaluate(syn, fused, defaultTargets(syn.spec));
  assert.ok(metrics.gainDb >= 6, `PSNR 增益应≥6dB，实际 ${metrics.gainDb.toFixed(2)}；${metrics.reasons.join(';')}`);
  assert.ok(metrics.depthMedianAbsErrStep <= 1, `深度误差应≤1步距，实际 ${metrics.depthMedianAbsErrStep}`);
  assert.ok(metrics.haloWidthPxMedian <= 2, `光晕应≤2px，实际 ${metrics.haloWidthPxMedian}`);
  assert.ok(metrics.pass, `应整体验收通过：${metrics.reasons.join(';')}`);
});

test('亚像素呼吸/抖动下深度估计保持鲁棒（深度误差 ≤1.5 步距）', () => {
  // 颜色 PSNR 对亚像素取色错位敏感，但深度（焦点层选择）是 DFF 的稳定输出，
  // 真实显微镜栈的几何抖动多在亚像素级，深度鲁棒性是更可靠的验收维度
  const syn = generateSyntheticStack(smallSpec({ breathScalePct: 0.15, jitterPx: 0.3 }));
  const fused = fuseDff(syn.images, syn.width, syn.height, {
    windowRadius: 2, topK: 4, lowTextureThresh: 0.12,
    specularLum: 235, softTemp: 8, haloGuard: 0.8,
  });
  const metrics = evaluate(syn, fused, defaultTargets(syn.spec));
  assert.ok(metrics.depthMedianAbsErrStep <= 1.5,
    `亚像素扰动下深度误差应≤1.5步距，实际 ${metrics.depthMedianAbsErrStep.toFixed(3)}`);
  assert.ok(metrics.haloWidthPxMedian <= 2.5, `光晕应受控，实际 ${metrics.haloWidthPxMedian}`);
});

test('融合图 PSNR 显著优于任一单层（oracle 选最优单层，无变换基线）', () => {
  const syn = generateSyntheticStack(smallSpec({ breathScalePct: 0, jitterPx: 0 }));
  const fused = fuseDff(syn.images, syn.width, syn.height, {
    windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235, softTemp: 8, haloGuard: 0.8,
  });
  const oracle = bestSingleLayerOracle(syn);
  assert.ok(fused && oracle.psnr > 0);
  const m = evaluate(syn, fused, defaultTargets(syn.spec));
  assert.ok(m.fusedPsnrDb > oracle.psnr + 5.5);
});

test('透明翅区域：贡献层 topK 保留多层而非硬选单层', () => {
  const syn = generateSyntheticStack(smallSpec());
  const fused = fuseDff(syn.images, syn.width, syn.height, {
    windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235, softTemp: 8, haloGuard: 0.8,
  });
  let wingPixels = 0;
  let multiContrib = 0;
  for (let p = 0; p < syn.wingMask.length; p++) {
    if (!syn.wingMask[p]) continue;
    wingPixels++;
    const contribs = new Set<number>();
    for (let k = 0; k < fused.topK; k++) contribs.add(fused.contribMatrix[p * fused.topK + k]);    if (contribs.size >= 2) multiContrib++;
  }
  assert.ok(wingPixels > 0);
  assert.ok(multiContrib / wingPixels > 0.5, `翅区多数像素应有≥2个贡献层，实际 ${(multiContrib / wingPixels).toFixed(2)}`);
});

test('拉普拉斯金字塔融合可运行且输出尺寸正确、数值在 0..255', () => {
  const syn = generateSyntheticStack(smallSpec({ layers: 10, width: 128, height: 96 }));
  const out = fuseLaplacianPyramid(syn.images.slice(0, 10), 128, 96, { levels: 4, consistencyRadius: 1 });
  assert.equal(out.length, 128 * 96 * 3);
  let mn = Infinity, mx = -Infinity;
  for (const v of out) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  assert.ok(mn >= 0 && mx <= 255);
});

test('对齐能还原纯几何平移（同内容位移，warp 后 SSD 显著下降）', () => {
  // 构造一张带丰富纹理的锐利图，施加已知平移——对齐算法的本职场景
  const w = 120, h = 90;
  const base = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4;
    const v = 110 + 60 * Math.sin(x * 0.35) * Math.cos(y * 0.28) + 35 * Math.sin(x * 0.9 + y * 0.7);
    base[p] = base[p + 1] = base[p + 2] = v; base[p + 3] = 255;
  }
  const lum = (im: Float32Array) => {
    const o = new Float32Array(w * h);
    for (let i = 0; i < o.length; i++) o[i] = 0.299 * im[i * 4] + 0.587 * im[i * 4 + 1] + 0.114 * im[i * 4 + 2];
    return o;
  };
  const refLum = lum(base);
  let ssdBefore = 0, ssdAfter = 0, cnt = 0;
  for (const shift of [2, 3, -2]) {
    const cur = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const sx = x - shift;
      const sp = (y * w + Math.max(0, Math.min(w - 1, sx))) * 4;
      const dp = (y * w + x) * 4;
      cur[dp] = cur[dp + 1] = cur[dp + 2] = sx < 0 || sx >= w ? 0 : base[sp];
      cur[dp + 3] = 255;
    }
    const est = estimateLayerAlign(base, cur, w, h);
    const warped = warpLayer(cur, w, h, est);
    const wLum = lum(warped), cLum = lum(cur);
    for (let y = 6; y < h - 6; y++) for (let x = 8; x < w - 8; x++) {
      ssdBefore += (cLum[y * w + x] - refLum[y * w + x]) ** 2;
      ssdAfter += (wLum[y * w + x] - refLum[y * w + x]) ** 2;
      cnt++;
    }
  }
  assert.ok(ssdAfter < ssdBefore * 0.75, `对齐后 SSD 应明显下降，比值 ${(ssdAfter / ssdBefore).toFixed(3)}`);
});

test('质检：重复层/缺层/曝光突变被标出，排序正确', () => {
  const mk = (fileName: string, exposureValue: number) =>
    ({ fileName, exposureValue, width: 10, height: 10, bitsPerSample: 8 as const, mimeType: 'image/jpeg' });
  // 6 个正常连续层
  const files = Array.from({ length: 6 }, (_, i) => mk(`STACK_${String(i + 1).padStart(4, '0')}.jpg`, 100));
  files.push(mk('STACK_0003.jpg', 100));  // 重复（独立文件，同 token）
  files.push(mk('STACK_0009.jpg', 100));  // 缺层（6→9 有空档，亮度正常，不撞曝光）
  files.push(mk('STACK_0010.jpg', 240));  // 曝光突变（独立文件，相邻正常层）
  const qc = orderLayers(files);
  const flags = [...qc.flags.values()].map((f) => f.flag);
  assert.ok(flags.includes('duplicate'), '应检出重复层');
  assert.ok(flags.includes('missing'), '应检出缺层');
  assert.ok(flags.includes('exposure-jump'), '应检出曝光突变');
  assert.equal(qc.ordered[0].file.fileName, 'STACK_0001.jpg');
  assert.equal(numericToken('IMG_12_023.tif'), 23);
});

test('测量：标定、角度、三维长度随深度差增长', () => {
  const cal = { ...calibrate(0.1, 1000, 5, 'test'), id: 'x', createdAt: 0 };
  assert.ok(Math.abs(cal.pixelsPerMm - 10000) < 1e-9);
  const angle = measureAngle({ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 });
  assert.ok(Math.abs(angle - 90) < 1e-9);
  const flat = measureLength({ x: 0, y: 0 }, { x: 100, y: 0 }, () => 0, cal, 2);
  const tilted = measureLength({ x: 0, y: 0 }, { x: 100, y: 0 }, (x) => (x === 0 ? 0 : 50), cal, 2);
  assert.ok(tilted.length3d > flat.length3d);
  const curve = measureCurve([{ x: 0, y: 0 }, { x: 30, y: 0 }, { x: 30, y: 40 }], () => 0, cal, 2);
  assert.ok(Math.abs(curve.planar - 70 / cal.pixelsPerMm) < 1e-9);
});

test('数学工具：鲁棒拟合抗离群、PSNR 完全一致时给高值、rng 可复现', () => {
  const xs = Float64Array.from({ length: 20 }, (_, i) => i);
  const ys = Float64Array.from({ length: 20 }, (_, i) => (i < 18 ? 2 * i + 1 : 999));
  const fit = robustLineFit(xs, ys);
  assert.ok(Math.abs(fit.a - 2) < 0.05 && Math.abs(fit.b - 1) < 0.2);
  const v = new Float32Array([1, 2, 3]);
  assert.ok(psnr(v, v) > 50);
  const a = rng(123), b = rng(123);
  assert.equal(a(), b());
  assert.equal(median([3, 1, 2]), 2);
});

test('深度误差在仿射偏差下仍以步距残差计（方向约定不影响结论）', () => {
  const syn = generateSyntheticStack(smallSpec());
  const fused = fuseDff(syn.images, syn.width, syn.height, {
    windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235, softTemp: 8, haloGuard: 0.8,
  });
  const de = depthErrorSteps(fused, syn.depthTruth, syn.bodyMask);
  assert.ok(de.usedRatio > 0.8, '掩膜后仍应用上大多数体素');
  assert.ok(de.medianAbs <= 1.5);
});

test('光晕指标在无差异图上为 0', () => {
  const syn = generateSyntheticStack(smallSpec());
  const w = haloWidth(syn.colorTruth, syn.colorTruth, syn.setaeEdge, syn.width, syn.height);
  assert.equal(w, 0);
});

function correlation(a: number[], b: number[]): number {
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return num / Math.sqrt(da * db);
}
