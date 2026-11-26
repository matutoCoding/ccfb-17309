// 轻量数学工具：向量/矩阵/拟合/统计，全部纯函数，便于 Node 单测

export interface Vec2 { x: number; y: number }

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function median(values: ArrayLike<number>): number {
  const a = Float64Array.from(values).sort();
  const n = a.length;
  if (n === 0) return NaN;
  const mid = n >> 1;
  return n % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

export function mean(values: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i];
  return s / Math.max(1, values.length);
}

export function stddev(values: ArrayLike<number>): number {
  const m = mean(values);
  let s = 0;
  for (let i = 0; i < values.length; i++) { const d = values[i] - m; s += d * d; }
  return Math.sqrt(s / Math.max(1, values.length));
}

/** MAD（中位绝对偏差），用于鲁棒尺度估计 */
export function mad(values: ArrayLike<number>): number {
  const m = median(values);
  const dev = Float64Array.from(values, (v) => Math.abs(v - m));
  return median(dev);
}

/** 鲁棒线性拟合 d_est = a * d_true + b（最小二乘初值 + 迭代剔除） */
export function robustLineFit(xs: ArrayLike<number>, ys: ArrayLike<number>, rejectSigma = 2.5): { a: number; b: number } {
  const n = xs.length;
  let mask = new Uint8Array(n).fill(1);
  let a = 1, b = 0;
  for (let iter = 0; iter < 5; iter++) {
    let sx = 0, sy = 0, sxx = 0, sxy = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      if (!mask[i]) continue;
      sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; cnt++;
    }
    if (cnt < 2) break;
    const den = cnt * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) break;
    a = (cnt * sxy - sx * sy) / den;
    b = (sy - a * sx) / cnt;
    const resid = new Float64Array(n);
    for (let i = 0; i < n; i++) resid[i] = ys[i] - (a * xs[i] + b);
    let used: number[] = [];
    for (let i = 0; i < n; i++) if (mask[i]) used.push(resid[i]);
    const sigma = Math.max(1e-6, 1.4826 * mad(used));
    const next = new Uint8Array(n);
    for (let i = 0; i < n; i++) next[i] = Math.abs(resid[i]) <= rejectSigma * sigma ? 1 : 0;
    if (next.every((v, i) => v === mask[i])) { mask = next; break; }
    mask = next;
  }
  return { a, b };
}

/** PSNR（0..255 量纲，MSE 在掩膜内计算） */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>, mask?: Uint8Array): number {
  let se = 0, n = 0;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const d = a[i] - b[i];
    se += d * d; n++;
  }
  if (!n) return NaN;
  const mse = se / n;
  if (mse < 1e-12) return 99;
  return 10 * Math.log10((255 * 255) / mse);
}

/** 确定性伪随机（mulberry32），合成数据需要可复现 */
export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 2D 高斯（可分离卷积核） */
export function gaussianKernel1d(radius: number, sigma: number): Float32Array {
  const k = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + radius] = v; sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

/** 由两组对应点估计相似变换（最小二乘，至少 2 对点），返回 [scale, angle, tx, ty] */
export function fitSimilarity(pairs: { ref: Vec2; cur: Vec2 }[]): { scale: number; angle: number; tx: number; ty: number } {
  // 先质心归一化，再解 [a -b; b a] 旋转缩放
  let cxr = 0, cyr = 0, cxc = 0, cyc = 0;
  for (const p of pairs) { cxr += p.ref.x; cyr += p.ref.y; cxc += p.cur.x; cyc += p.cur.y; }
  const n = pairs.length;
  cxr /= n; cyr /= n; cxc /= n; cyc /= n;
  let sxx = 0, sxy = 0, syx = 0, syy = 0, srr = 0;
  for (const p of pairs) {
    const rx = p.ref.x - cxr, ry = p.ref.y - cyr;
    const cx = p.cur.x - cxc, cy = p.cur.y - cyc;
    sxx += rx * cx; sxy += rx * cy; syx += ry * cx; syy += ry * cy;
    srr += rx * rx + ry * ry;
  }
  // cur = R(ref) 时 a=(sxx+syy)/srr, b=(syx-sxy)/srr；这里求 cur->ref 的逆
  const a = (sxx + syy) / Math.max(1e-9, srr);
  const b = (syx - sxy) / Math.max(1e-9, srr);
  // cur' = a*cur + b*... 实际映射: ref = M*cur, M=[[a,b],[-b,a]]
  const scale = Math.hypot(a, b);
  const angle = Math.atan2(-b, a);
  const tx = cxr - (a * cxc + b * cyc);
  const ty = cyr - (-b * cxc + a * cyc);
  return { scale, angle, tx, ty };
}

export function uid(prefix = 'id'): string {
  const rnd = (globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`);
  return `${prefix}_${rnd}`;
}

/** 指数移动平均 */
export function ema(prev: number, next: number, alpha: number): number {
  return alpha * next + (1 - alpha) * prev;
}
