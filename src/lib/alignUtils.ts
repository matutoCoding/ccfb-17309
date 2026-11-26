// boxMean 独立导出，供对齐与 CPU 融合共用
export function boxMean(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const k = radius * 2 + 1;
  const clampI = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let x = -radius; x <= radius; x++) acc += src[y * w + clampI(x, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / k;
      acc += src[y * w + clampI(x + radius + 1, 0, w - 1)] - src[y * w + clampI(x - radius, 0, w - 1)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -radius; y <= radius; y++) acc += tmp[clampI(y, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / k;
      acc += tmp[clampI(y + radius + 1, 0, h - 1) * w + x] - tmp[clampI(y - radius, 0, h - 1) * w + x];
    }
  }
  return out;
}
