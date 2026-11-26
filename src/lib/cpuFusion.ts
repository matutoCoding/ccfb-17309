// CPU 参考实现：与 GPU 着色器同算法，用于单测真值验证与降级回退
import { clamp, gaussianKernel1d, median } from './math';

export interface FusionResult {
  color: Float32Array; // RGB 0..255
  depth: Float32Array; // 层索引单位
  confidence: Float32Array; // 0..1
  lowTexture: Uint8Array;
  specular: Uint8Array;
  holeMask: Uint8Array; // 经引导滤波补掉的空洞
  /** 每像素主贡献层（argmax 权重） */
  topLayer: Uint16Array;
  /** 每像素 topK 贡献层索引（按权重降序），透明膜翅多层贡献查询用 */
  contributors: Uint16Array[]; // 每层一张图？改为 [pixel*topK]
  contribMatrix: Uint16Array; // 行主序 pixel × topK
  topK: number;
}

/** 归一化拉普拉斯方差焦点度量：高通去基线 + 亮度归一，离焦层趋近 0、对焦层尖锐 */
export function focusMeasure(layer: Float32Array, w: number, h: number, channels: number, radius: number): Float32Array {
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    lum[i] = layer[i * channels] * 0.299 + layer[i * channels + 1] * 0.587 + layer[i * channels + 2] * 0.114;
  }
  // 拉普拉斯响应（高通，离焦的低频结构被抑制）
  const lap = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      lap[i] = 4 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w];
    }
  }
  // 窗口内拉普拉斯方差（方差而非能量：减去局部均值，去基线偏置）
  const mean = boxMean(lap, w, h, radius);
  const sq = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) sq[i] = lap[i] * lap[i];
  const meanSq = boxMean(sq, w, h, radius);
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    // 亮度归一（+20 防暗区除零），让不同亮度区域的焦点响应可比
    out[i] = Math.max(0, meanSq[i] - mean[i] * mean[i]) / (lum[i] + 20);
  }
  return out;
}

function boxMean(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const k = radius * 2 + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let x = -radius; x <= radius; x++) acc += src[y * w + clamp(x, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / k;
      const xOut = clamp(x - radius, 0, w - 1);
      const xIn = clamp(x + radius + 1, 0, w - 1);
      acc += src[y * w + xIn] - src[y * w + xOut];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -radius; y <= radius; y++) acc += tmp[clamp(y, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / k;
      const yOut = clamp(y - radius, 0, h - 1);
      const yIn = clamp(y + radius + 1, 0, h - 1);
      acc += tmp[yIn * w + x] - tmp[yOut * w + x];
    }
  }
  return out;
}

export interface DffOptions {
  windowRadius: number;
  topK: number;
  lowTextureThresh: number; // 焦点能量低于此值 → 低纹理
  specularLum: number; // 亮度高于此且层间方差小 → 高光
  softTemp: number; // softmax 温度（层索引单位）
  haloGuard: number; // 0..1，抑制单层独大造成的边缘光晕
}

/**
 * 按焦点度量取深度的 DFF：
 * - 软选择 topK 层加权混合（透明膜翅多层有信号时不硬选）
 * - 光晕抑制：对权重做边缘感知的轻度平滑，避免刚毛边缘单像素跳层
 * - 置信度 + 低纹理/高光掩膜 + 引导滤波补洞
 */
export function fuseDff(
  images: Float32Array[], w: number, h: number, opts: DffOptions,
): FusionResult {
  const N = images.length;
  const measures = images.map((im) => focusMeasure(im, w, h, im.length / (w * h) === 4 ? 4 : 3, opts.windowRadius));
  const depth = new Float32Array(w * h);
  const confidence = new Float32Array(w * h);
  const topLayer = new Uint16Array(w * h);
  const contrib = new Uint16Array(w * h * opts.topK);
  const lowTex = new Uint8Array(w * h);
  const spec = new Uint8Array(w * h);
  const hole = new Uint8Array(w * h);
  const color = new Float32Array(w * h * 3);

  // 全局自适应低纹理阈值：焦点能量中位数的比例
  const allMax = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    let mx = 0;
    for (let z = 0; z < N; z++) mx = Math.max(mx, measures[z][p]);
    allMax[p] = mx;
  }
  const medMax = median(allMax);
  const lowThresh = medMax * opts.lowTextureThresh;

  for (let p = 0; p < w * h; p++) {
    // 找 topK
    const order = Array.from({ length: N }, (_, z) => z);
    order.sort((a, b) => measures[b][p] - measures[a][p]);
    const k = Math.min(opts.topK, N);
    const top = order.slice(0, k);
    for (let j = 0; j < opts.topK; j++) contrib[p * opts.topK + j] = top[Math.min(j, k - 1)];
    topLayer[p] = top[0];

    // softmax 权重（温度按焦点能量尺度归一）
    const mVals = top.map((z) => measures[z][p]);
    const mMax = Math.max(...mVals);
    const scale = opts.softTemp * Math.max(1, medMax * 0.02);
    const wts = mVals.map((m) => Math.exp((m - mMax) / scale));
    const wSum = wts.reduce((a, b) => a + b, 0);

    let d = 0;
    for (let j = 0; j < k; j++) {
      const wt = wts[j] / wSum;
      d += wt * top[j];
    }
    depth[p] = d;
    // 颜色不在此处混合（softmax 跨层混色会发糊）；统一在下方按深度相邻两层重取

    // 置信度：焦点峰锐度 + 主层权重占比
    const peakShare = wts[0] / wSum;
    const focusStrength = clamp(allMax[p] / Math.max(1e-6, medMax * 1.5), 0, 1);
    confidence[p] = clamp(0.55 * peakShare + 0.45 * focusStrength, 0, 1);

    if (allMax[p] < lowThresh) { lowTex[p] = 1; hole[p] = 1; }

    // 高光检测：主层亮度高且各层亮度方差小
    let lumMean = 0;
    const lums = new Float32Array(N);
    for (let z = 0; z < N; z++) {
      const im = images[z]; const ch = im.length / (w * h);
      lums[z] = 0.299 * im[p * ch] + 0.587 * im[p * ch + 1] + 0.114 * im[p * ch + 2];
      lumMean += lums[z];
    }
    lumMean /= N;
    let lv = 0;
    for (let z = 0; z < N; z++) lv += (lums[z] - lumMean) ** 2;
    lv /= N;
    if (lumMean > opts.specularLum && lv < 25) { spec[p] = 1; hole[p] = 1; confidence[p] = Math.min(confidence[p], 0.25); }
  }

  // 光晕抑制：对深度做边缘保留的轻度正则（仅在疑似跳层光晕处混合）
  if (opts.haloGuard > 0) {
    const smoothed = boxMean(depth, w, h, 1);
    for (let p = 0; p < w * h; p++) {
      if (hole[p]) continue;
      const diff = Math.abs(depth[p] - smoothed[p]);
      if (diff > 1.5) {
        depth[p] = (1 - 0.5 * opts.haloGuard) * depth[p] + 0.5 * opts.haloGuard * smoothed[p];
      }
    }
  }

  // 统一按校正后深度重取颜色：最近两层线性插值（非空洞像素），保持锐利不跨层混色
  for (let p = 0; p < w * h; p++) {
    if (hole[p]) continue;
    const d = clamp(depth[p], 0, N - 1.001);
    const z0 = Math.floor(d), t = d - z0;
    const a = images[z0], b = images[z0 + 1];
    const ca = a.length / (w * h), cbch = b.length / (w * h);
    color[p * 3] = a[p * ca] * (1 - t) + b[p * cbch] * t;
    color[p * 3 + 1] = a[p * ca + 1] * (1 - t) + b[p * cbch + 1] * t;
    color[p * 3 + 2] = a[p * ca + 2] * (1 - t) + b[p * cbch + 2] * t;
  }

  // 引导滤波补深度空洞（以合成亮度为引导）
  const filled = guidedFill(depth, color, w, h, hole, 4, 1e-3 * 255 * 255);
  for (let p = 0; p < w * h; p++) if (hole[p]) depth[p] = filled[p];
  // 空洞像素颜色回填：用补全深度对应层
  for (let p = 0; p < w * h; p++) {
    if (!hole[p]) continue;
    const d = clamp(depth[p], 0, N - 1.001);
    const z0 = Math.floor(d), t = d - z0;
    const a = images[z0], b = images[z0 + 1];
    const ca = a.length / (w * h), cbch = b.length / (w * h);
    color[p * 3] = a[p * ca] * (1 - t) + b[p * cbch] * t;
    color[p * 3 + 1] = a[p * ca + 1] * (1 - t) + b[p * cbch + 1] * t;
    color[p * 3 + 2] = a[p * ca + 2] * (1 - t) + b[p * cbch + 2] * t;
  }

  return {
    color, depth, confidence, lowTexture: lowTex, specular: spec, holeMask: hole,
    topLayer, contributors: [], contribMatrix: contrib, topK: opts.topK,
  };
}

/** 引导滤波（He et al. 2010），仅在 hole 位置回填 */
export function guidedFill(
  input: Float32Array, guideRgb: Float32Array, w: number, h: number,
  hole: Uint8Array, radius: number, eps: number,
): Float32Array {
  const guide = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    guide[i] = 0.299 * guideRgb[i * 3] + 0.587 * guideRgb[i * 3 + 1] + 0.114 * guideRgb[i * 3 + 2];
  }
  const meanI = boxMean(guide, w, h, radius);
  const meanP = boxMean(input, w, h, radius);
  const corrI = boxMean(mul(guide, guide), w, h, radius);
  const corrIP = boxMean(mul(guide, input), w, h, radius);
  const varI = sub(corrI, mul(meanI, meanI));
  const covIP = sub(corrIP, mul(meanI, meanP));
  const a = new Float32Array(w * h);
  const b = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    a[i] = covIP[i] / (varI[i] + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  const meanA = boxMean(a, w, h, radius);
  const meanB = boxMean(b, w, h, radius);
  const out = Float32Array.from(input);
  for (let i = 0; i < w * h; i++) {
    if (hole[i]) out[i] = meanA[i] * guide[i] + meanB[i];
  }
  return out;
}

function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] * b[i];
  return o;
}
function sub(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] - b[i];
  return o;
}

// ---------------- 拉普拉斯金字塔融合（Burt–Coleczynski 思路） ----------------

export interface LaplaceOptions { levels: number; consistencyRadius: number }

export function fuseLaplacianPyramid(images: Float32Array[], w: number, h: number, opts: LaplaceOptions): Float32Array {
  const N = images.length;
  const rgb = images.map((im) => {
    const ch = im.length / (w * h);
    if (ch === 3) return Float32Array.from(im);
    const o = new Float32Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      o[i * 3] = im[i * 4]; o[i * 3 + 1] = im[i * 4 + 1]; o[i * 3 + 2] = im[i * 4 + 2];
    }
    return o;
  });
  // 每层决策图：局部拉普拉斯能量最大者
  const decision = new Uint8Array(w * h);
  const energy = rgb.map((im) => localLaplaceEnergy(im, w, h));
  for (let p = 0; p < w * h; p++) {
    let best = 0, bv = -1;
    for (let z = 0; z < N; z++) if (energy[z][p] > bv) { bv = energy[z][p]; best = z; }
    decision[p] = best;
  }
  // 小区域一致性：3x3/5x5 多数投票
  for (let pass = 0; pass < 2; pass++) {
    const next = Uint8Array.from(decision);
    const r = opts.consistencyRadius;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const votes = new Map<number, number>();
        for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
          const xx = clamp(x + dx, 0, w - 1), yy = clamp(y + dy, 0, h - 1);
          const v = decision[yy * w + xx];
          votes.set(v, (votes.get(v) ?? 0) + 1);
        }
        let bv = 0, bk = decision[y * w + x];
        for (const [k, v] of votes) if (v > bv) { bv = v; bk = k; }
        next[y * w + x] = bk;
      }
    }
    decision.set(next);
  }

  // 建金字塔并按各级模糊决策掩码融合
  const outPyrs: Float32Array[] = [];
  const sizes: { w: number; h: number }[] = [];
  for (let lvl = 0; lvl < opts.levels; lvl++) {
    sizes.push({ w: Math.max(1, Math.round(w / 2 ** lvl)), h: Math.max(1, Math.round(h / 2 ** lvl)) });
  }
  const masksOneHot: Float32Array[] = [];
  for (let z = 0; z < N; z++) {
    const m = new Float32Array(w * h);
    for (let p = 0; p < w * h; p++) if (decision[p] === z) m[p] = 1;
    masksOneHot.push(m);
  }
  const pyrImages = rgb.map((im) => buildGaussian(im, w, h, opts.levels));
  const pyrMasks = masksOneHot.map((m) => buildGaussian(m, w, h, opts.levels));
  for (let lvl = 0; lvl < opts.levels; lvl++) {
    const lap = pyrImages.map((pyr) => {
      if (lvl === opts.levels - 1) return pyr[lvl];
      return laplacianLevel(pyr[lvl], pyr[lvl + 1], sizes[lvl + 1].w, sizes[lvl + 1].h);
    });
    const merged = new Float32Array(lap[0].length);
    const maskSum = new Float32Array(lap[0].length / 3);
    for (let z = 0; z < N; z++) {
      const m = pyrMasks[z][lvl];
      for (let p = 0; p < maskSum.length; p++) {
        const mm = m[p];
        maskSum[p] += mm;
        for (let c = 0; c < 3; c++) merged[p * 3 + c] += lap[z][p * 3 + c] * mm;
      }
    }
    for (let p = 0; p < maskSum.length; p++) {
      if (maskSum[p] < 1e-6) for (let c = 0; c < 3; c++) merged[p * 3 + c] = lap[0][p * 3 + c];
    }
    outPyrs.push(merged);
  }
  return reconstruct(outPyrs, sizes, w, h);
}

function localLaplaceEnergy(rgb: Float32Array, w: number, h: number): Float32Array {
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) lum[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  const lap = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    lap[i] = (4 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w]) ** 2;
  }
  return boxMean(lap, w, h, 1);
}

function buildGaussian(src: Float32Array, w: number, h: number, levels: number): Float32Array[] {
  const pyr: Float32Array[] = [Float32Array.from(src)];
  let cw = w, ch = h;
  const channels = src.length / (w * h);
  for (let l = 1; l < levels; l++) {
    const nw = Math.max(1, cw >> 1), nh = Math.max(1, ch >> 1);
    const blurred = separable(pyr[pyr.length - 1], cw, ch, channels, 1);
    const down = new Float32Array(nw * nh * channels);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      const sx = Math.min(cw - 1, x * 2), sy = Math.min(ch - 1, y * 2);
      for (let c = 0; c < channels; c++) down[(y * nw + x) * channels + c] = blurred[(sy * cw + sx) * channels + c];
    }
    pyr.push(down); cw = nw; ch = nh;
  }
  return pyr;
}

function separable(src: Float32Array, w: number, h: number, channels: number, radius: number): Float32Array {
  const k = gaussianKernel1d(radius, radius * 0.7 + 0.3);
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < channels; c++) {
    let acc = 0;
    for (let i = -radius; i <= radius; i++) acc += src[(y * w + clamp(x + i, 0, w - 1)) * channels + c] * k[i + radius];
    tmp[(y * w + x) * channels + c] = acc;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < channels; c++) {
    let acc = 0;
    for (let i = -radius; i <= radius; i++) acc += tmp[(clamp(y + i, 0, h - 1) * w + x) * channels + c] * k[i + radius];
    out[(y * w + x) * channels + c] = acc;
  }
  return out;
}

function laplacianLevel(cur: Float32Array, next: Float32Array, nextW: number, nextH: number): Float32Array {
  const up = upsample(next, nextW, nextH, cur.length / 3, 3);
  const out = new Float32Array(cur.length);
  for (let i = 0; i < cur.length; i++) out[i] = cur[i] - up[i];
  return out;
}

function upsample(src: Float32Array, srcW: number, srcH: number, targetPixels: number, channels: number): Float32Array {
  const scale = Math.sqrt(targetPixels / (srcW * srcH));
  const tw = Math.round(srcW * scale), th = Math.round(srcH * scale);
  const out = new Float32Array(tw * th * channels);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
    const sx = clamp(Math.floor(x / scale), 0, srcW - 1);
    const sy = clamp(Math.floor(y / scale), 0, srcH - 1);
    for (let c = 0; c < channels; c++) out[(y * tw + x) * channels + c] = src[(sy * srcW + sx) * channels + c];
  }
  return out;
}

function reconstruct(pyr: Float32Array[], sizes: { w: number; h: number }[], w: number, h: number): Float32Array {
  let cur = pyr[pyr.length - 1];
  for (let l = pyr.length - 2; l >= 0; l--) {
    const up = upsample(cur, sizes[l + 1].w, sizes[l + 1].h, sizes[l].w * sizes[l].h, 3);
    const merged = new Float32Array(pyr[l].length);
    for (let i = 0; i < merged.length; i++) merged[i] = pyr[l][i] + up[i];
    cur = merged;
  }
  const out = new Float32Array(w * h * 3);
  for (let i = 0; i < out.length; i++) out[i] = clamp(cur[i], 0, 255);
  return out;
}
