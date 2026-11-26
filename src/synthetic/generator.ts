// 合成测试数据：已知高度场（虫体+刚毛+刻点+半透明翅），按光学景深模型出图层栈
// 纯 TS、无 DOM 依赖，Node 单测与浏览器共用
import type { SyntheticSpec } from '../types';
import { clamp, lerp, rng } from '../lib/math';

export interface SyntheticStack {
  spec: SyntheticSpec;
  width: number;
  height: number;
  layers: number;
  /** 每层 RGBA float（0..255），已含呼吸缩放与导轨抖动 */
  images: Float32Array[];
  /** 每层实际施加的相似变换（供对齐算法真值对照） */
  transforms: { scale: number; tx: number; ty: number }[];
  /** 真值高度场，单位=对焦步距（0..layers-1） */
  depthTruth: Float32Array;
  /** 全清晰真值彩色图 RGB */
  colorTruth: Float32Array;
  /** 虫体掩膜 */
  bodyMask: Uint8Array;
  /** 高光掩膜 */
  specularMask: Uint8Array;
  /** 低纹理掩膜 */
  lowTextureMask: Uint8Array;
  /** 半透明翅掩膜（多层信号区） */
  wingMask: Uint8Array;
  /** 刚毛边缘像素（光晕评测用） */
  setaeEdge: Uint8Array;
  /** 每层对焦位置（模拟 EXIF）与文件名 */
  focusPositions: number[];
  fileNames: string[];
}

interface HeightField {
  h: Float32Array;
  color: Float32Array; // RGB 全清晰
  body: Uint8Array;
  specular: Uint8Array;
  lowTex: Uint8Array;
  wing: Uint8Array;
  wingH: Float32Array;
  wingAlpha: Float32Array;
  setaeEdge: Uint8Array;
}

/** 可分离高斯模糊（单通道平面数组，边界 clamp） */
export function gaussianBlur(src: Float32Array, w: number, h: number, channels: number, sigma: number): Float32Array {
  if (sigma < 0.01) return Float32Array.from(src);
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k[i + radius] = v; sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  // 水平
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let i = -radius; i <= radius; i++) {
          const xx = clamp(x + i, 0, w - 1);
          acc += src[(y * w + xx) * channels + c] * k[i + radius];
        }
        tmp[(y * w + x) * channels + c] = acc;
      }
    }
  }
  // 垂直
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let i = -radius; i <= radius; i++) {
          const yy = clamp(y + i, 0, h - 1);
          acc += tmp[(yy * w + x) * channels + c] * k[i + radius];
        }
        out[(y * w + x) * channels + c] = acc;
      }
    }
  }
  return out;
}

function buildHeightField(spec: SyntheticSpec): HeightField {
  const { width: w, height: h, layers: N, seed } = spec;
  const rand = rng(seed);
  const hf: HeightField = {
    h: new Float32Array(w * h),
    color: new Float32Array(w * h * 3),
    body: new Uint8Array(w * h),
    specular: new Uint8Array(w * h),
    lowTex: new Uint8Array(w * h),
    wing: new Uint8Array(w * h),
    wingH: new Float32Array(w * h),
    wingAlpha: new Float32Array(w * h),
    setaeEdge: new Uint8Array(w * h),
  };
  const cx = w * 0.42, cy = h * 0.52;
  const rx = w * 0.30, ry = h * 0.36;
  const zBase = N * 0.30, zAmp = N * 0.34;

  // 1) 虫体：椭球穹顶 + 鞘翅中缝
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const nx = (x - cx) / rx, ny = (y - cy) / ry;
      const r2 = nx * nx + ny * ny;
      if (r2 <= 1) {
        hf.body[idx] = 1;
        const dome = Math.sqrt(Math.max(0, 1 - r2));
        let z = zBase + zAmp * dome;
        // 鞘翅中缝（一条浅沟）
        const seam = Math.exp(-Math.abs(x - cx) / 1.6) * 0.9;
        z -= seam * N * 0.02;
        hf.h[idx] = z;
        // 基色：红棕鞘翅，带细微纹理噪声 + 稳定的高频颗粒（焦点度量需要真实纹理信号）
        const grain = Math.sin(x * 1.7 + y * 0.9) * 4 + Math.sin(x * 0.6 - y * 1.3) * 3;
        const tex = (rand() - 0.5) * 10 + grain;
        hf.color[idx * 3] = clamp(118 + tex, 0, 255);
        hf.color[idx * 3 + 1] = clamp(62 + tex * 0.6, 0, 255);
        hf.color[idx * 3 + 2] = clamp(34 + tex * 0.4, 0, 255);
      } else {
        // 背景：中性灰载物台
        hf.color[idx * 3] = 206 + (rand() - 0.5) * 6;
        hf.color[idx * 3 + 1] = 204 + (rand() - 0.5) * 6;
        hf.color[idx * 3 + 2] = 198 + (rand() - 0.5) * 6;
        hf.h[idx] = 0;
      }
    }
  }

  // 2) 刻点：高斯小坑 + 暗化
  if (spec.includePunctures) {
    for (let p = 0; p < 70; p++) {
      const px = cx + (rand() * 2 - 1) * rx * 0.8;
      const py = cy + (rand() * 2 - 1) * ry * 0.8;
      const pr = 1.6 + rand() * 2.2;
      const depth = N * (0.015 + rand() * 0.02);
      const R = Math.ceil(pr * 2.5);
      for (let y = Math.max(0, py - R | 0); y < Math.min(h, py + R + 1); y++) {
        for (let x = Math.max(0, px - R | 0); x < Math.min(w, px + R + 1); x++) {
          const idx = y * w + x;
          if (!hf.body[idx]) continue;
          const d2 = (x - px) ** 2 + (y - py) ** 2;
          const g = Math.exp(-d2 / (2 * pr * pr));
          hf.h[idx] -= depth * g;
          hf.color[idx * 3] *= 1 - 0.35 * g;
          hf.color[idx * 3 + 1] *= 1 - 0.35 * g;
          hf.color[idx * 3 + 2] *= 1 - 0.35 * g;
        }
      }
    }
  }

  // 3) 刚毛：细亮脊线，沿线段抬升高度并记录边缘
  if (spec.includeSetae) {
    const setae: { x0: number; y0: number; x1: number; y1: number }[] = [];
    for (let s = 0; s < 46; s++) {
      const ang = rand() * Math.PI * 2;
      const bx = cx + Math.cos(ang) * rx * 0.55 * (0.4 + rand() * 0.5);
      const by = cy + Math.sin(ang) * ry * 0.55 * (0.4 + rand() * 0.5);
      const len = 6 + rand() * 12;
      const dir = ang + (rand() - 0.5) * 0.9;
      setae.push({ x0: bx, y0: by, x1: bx + Math.cos(dir) * len, y1: by + Math.sin(dir) * len });
    }
    const plot = (x: number, y: number, bright: number, ridge: number) => {
      const xi = x | 0, yi = y | 0;
      if (xi < 1 || yi < 1 || xi >= w - 1 || yi >= h - 1) return;
      const idx = yi * w + xi;
      if (!hf.body[idx]) return;
      hf.h[idx] += ridge;
      hf.color[idx * 3] = clamp(hf.color[idx * 3] + bright, 0, 255);
      hf.color[idx * 3 + 1] = clamp(hf.color[idx * 3 + 1] + bright * 0.9, 0, 255);
      hf.color[idx * 3 + 2] = clamp(hf.color[idx * 3 + 2] + bright * 0.7, 0, 255);
    };
    for (const seg of setae) {
      const steps = Math.ceil(Math.hypot(seg.x1 - seg.x0, seg.y1 - seg.y0)) * 2;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = lerp(seg.x0, seg.x1, t), y = lerp(seg.y0, seg.y1, t);
        plot(x, y, 70, N * 0.012);
      }
      // 边缘标记（线两侧像素）
      const dx = seg.x1 - seg.x0, dy = seg.y1 - seg.y0;
      const L = Math.hypot(dx, dy) || 1;
      const nx = -dy / L, ny = dx / L;
      for (let i = 0; i <= steps; i += 2) {
        const t = i / steps;
        const x = lerp(seg.x0, seg.x1, t), y = lerp(seg.y0, seg.y1, t);
        for (const off of [-1.6, 1.6]) {
          const xi = (x + nx * off) | 0, yi = (y + ny * off) | 0;
          if (xi >= 0 && yi >= 0 && xi < w && yi < h && hf.body[yi * w + xi]) hf.setaeEdge[yi * w + xi] = 1;
        }
      }
    }
  }

  // 4) 半透明膜翅：体侧椭圆，独立深度面 + alpha 混合
  if (spec.includeWing) {
    const wcx = w * 0.80, wcy = h * 0.5, wrx = w * 0.14, wry = h * 0.30;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const q = ((x - wcx) / wrx) ** 2 + ((y - wcy) / wry) ** 2;
        if (q <= 1) {
          hf.wing[idx] = 1;
          // 翅脉纹理
          const vein = Math.abs(Math.sin((y - wcy) * 0.55 + Math.sin(x * 0.2) * 2)) > 0.93 ? 1 : 0;
          hf.wingAlpha[idx] = 0.42 + vein * 0.25;
          hf.wingH[idx] = N * (0.18 + 0.10 * (1 - q)); // 与虫体不同深度
        }
      }
    }
  }

  // 5) 高光斑（镜面反射，所有层都亮 → 深度不可信）
  for (let s = 0; s < 5; s++) {
    const sx = cx + (rand() * 2 - 1) * rx * 0.5;
    const sy = cy + (rand() * 2 - 1) * ry * 0.5;
    const sr = 2 + rand() * 2.5;
    const R = Math.ceil(sr * 2);
    for (let y = Math.max(0, sy - R | 0); y < Math.min(h, sy + R + 1); y++) {
      for (let x = Math.max(0, sx - R | 0); x < Math.min(w, sx + R + 1); x++) {
        const idx = y * w + x;
        if (!hf.body[idx]) continue;
        const g = Math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sr * sr));
        if (g > 0.4) hf.specular[idx] = 1;
        hf.color[idx * 3] = clamp(hf.color[idx * 3] + g * 150, 0, 255);
        hf.color[idx * 3 + 1] = clamp(hf.color[idx * 3 + 1] + g * 150, 0, 255);
        hf.color[idx * 3 + 2] = clamp(hf.color[idx * 3 + 2] + g * 145, 0, 255);
      }
    }
  }

  // 6) 低纹理区：前胸背板一块平滑区（颜色抹平）
  const lcx = cx, lcy = cy - ry * 0.72, lrx = rx * 0.32, lry = ry * 0.16;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!hf.body[idx]) continue;
      const q = ((x - lcx) / lrx) ** 2 + ((y - lcy) / lry) ** 2;
      if (q <= 1) {
        hf.lowTex[idx] = 1;
        hf.color[idx * 3] = 128; hf.color[idx * 3 + 1] = 70; hf.color[idx * 3 + 2] = 40;
      }
    }
  }
  return hf;
}

/** 离焦半径（像素）与离焦步距的关系：薄景深，sigma = k * |Δz|。
 *  k 取 0.4：对焦层锐利（理想融合上界 >30dB），离焦层仍明显模糊以驱动焦点度量 */
const DEFOCUS_K = 0.4;

export function generateSyntheticStack(spec: SyntheticSpec): SyntheticStack {
  const { width: w, height: h, layers: N } = spec;
  const rand = rng(seedPair(spec.seed));
  const field = buildHeightField(spec);

  // 全清晰真值图：与图层渲染同口径——锐利体表面 + 锐利半透明翅叠加（保证理想融合上界可达高 PSNR）
  const colorTruth = Float32Array.from(field.color);
  for (let idx = 0; idx < w * h; idx++) {
    if (field.wing[idx]) {
      const vein = field.wingAlpha[idx] > 0.5 ? 1 : 0;
      const wc = vein ? 90 : 225;
      const a = field.wingAlpha[idx];
      colorTruth[idx * 3] = colorTruth[idx * 3] * (1 - a) + wc * a;
      colorTruth[idx * 3 + 1] = colorTruth[idx * 3 + 1] * (1 - a) + wc * a;
      colorTruth[idx * 3 + 2] = colorTruth[idx * 3 + 2] * (1 - a) + wc * a;
    }
  }

  // 预计算密集离焦级别（sigma 步长 0.25，覆盖最大可能离焦），空间变化离焦按级线性插值，
  // 避免粗量化导致"理想深度取色"也对不上真值
  const maxSigma = DEFOCUS_K * N + 1;
  const sigmaLevels: number[] = [];
  for (let s = 0; s <= maxSigma + 0.001; s += 0.25) sigmaLevels.push(Number(s.toFixed(3)));
  const bodyBlurred = sigmaLevels.map((sig) => gaussianBlur(field.color, w, h, 3, sig));
  // 翅的本色（浅灰白 + 翅脉）
  const wingColor = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    const vein = field.wingAlpha[i] > 0.5 ? 1 : 0;
    wingColor[i * 3] = wingColor[i * 3 + 1] = wingColor[i * 3 + 2] = vein ? 90 : 225;
  }
  const wingBlurred = sigmaLevels.map((sig) => gaussianBlur(wingColor, w, h, 3, sig));

  // 每层变换：呼吸（平滑缩放曲线）+ 抖动
  const transforms: SyntheticStack['transforms'] = [];
  for (let z = 0; z < N; z++) {
    const t = z / Math.max(1, N - 1);
    const breath = spec.breathScalePct / 100 * Math.sin(t * Math.PI); // 中间层缩放最大
    const scale = 1 + breath;
    const tx = (rand() * 2 - 1) * spec.jitterPx;
    const ty = (rand() * 2 - 1) * spec.jitterPx;
    transforms.push({ scale, tx, ty });
  }

  const images: Float32Array[] = [];
  const focusPositions: number[] = [];
  const fileNames: string[] = [];

  for (let z = 0; z < N; z++) {
    const tr = transforms[z];
    const img = new Float32Array(w * h * 4);
    const cx = w / 2, cy = h / 2;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // 逆变换：输出像素 -> 规范坐标
        const dx0 = x - cx - tr.tx, dy0 = y - cy - tr.ty;
        const sx = cx + dx0 / tr.scale;
        const sy = cy + dy0 / tr.scale;
        const xi = sx | 0, yi = sy | 0;
        const o = (y * w + x) * 4;
        if (xi < 0 || yi < 0 || xi >= w || yi >= h) {
          img[o] = img[o + 1] = img[o + 2] = 206; img[o + 3] = 255;
          continue;
        }
        const idx = yi * w + xi;
        // 体表面离焦
        const deltaBody = Math.abs(z - field.h[idx]);
        const sigBody = DEFOCUS_K * deltaBody;
        sampleBlurred(bodyBlurred, sigmaLevels, sigBody, idx, img, o);
        // 半透明翅叠加（多层信号，不做硬选择）
        if (field.wing[idx]) {
          const deltaWing = Math.abs(z - field.wingH[idx]);
          const tmp = new Float32Array(3);
          sampleBlurred(wingBlurred, sigmaLevels, DEFOCUS_K * deltaWing, idx, tmp, 0);
          const a = field.wingAlpha[idx];
          img[o] = img[o] * (1 - a) + tmp[0] * a;
          img[o + 1] = img[o + 1] * (1 - a) + tmp[1] * a;
          img[o + 2] = img[o + 2] * (1 - a) + tmp[2] * a;
        }
        // 传感器噪声（仅离焦层加；最近焦层保持真值，保证理想融合上界可达高 PSNR）
        if (Math.abs(z - field.h[idx]) > 0.5) {
          const n = (rand() - 0.5) * 4;
          img[o] = clamp(img[o] + n, 0, 255);
          img[o + 1] = clamp(img[o + 1] + n, 0, 255);
          img[o + 2] = clamp(img[o + 2] + n, 0, 255);
        }
        img[o + 3] = 255;
      }
    }
    images.push(img);
    focusPositions.push(z * spec.stepMicron);
    fileNames.push(`STACK_${String(z + 1).padStart(4, '0')}.jpg`);
  }

  return {
    spec, width: w, height: h, layers: N, images, transforms,
    depthTruth: field.h, colorTruth, bodyMask: field.body,
    specularMask: field.specular, lowTextureMask: field.lowTex,
    wingMask: field.wing, setaeEdge: field.setaeEdge,
    focusPositions, fileNames,
  };
}

function seedPair(seed: number): number {
  return (seed * 2654435761) >>> 0;
}

function sampleBlurred(levels: Float32Array[], sigmas: number[], sigma: number, idx: number, out: Float32Array, o: number): void {
  if (sigma <= sigmas[0]) {
    const src = levels[0];
    out[o] = src[idx * 3]; out[o + 1] = src[idx * 3 + 1]; out[o + 2] = src[idx * 3 + 2];
    return;
  }
  let k = 0;
  while (k < sigmas.length - 1 && sigmas[k + 1] < sigma) k++;
  if (k >= sigmas.length - 1) {
    const src = levels[sigmas.length - 1];
    out[o] = src[idx * 3]; out[o + 1] = src[idx * 3 + 1]; out[o + 2] = src[idx * 3 + 2];
    return;
  }
  const t = (sigma - sigmas[k]) / (sigmas[k + 1] - sigmas[k]);
  const a = levels[k], b = levels[k + 1];
  out[o] = lerp(a[idx * 3], b[idx * 3], t);
  out[o + 1] = lerp(a[idx * 3 + 1], b[idx * 3 + 1], t);
  out[o + 2] = lerp(a[idx * 3 + 2], b[idx * 3 + 2], t);
}
