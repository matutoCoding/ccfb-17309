// 图像编码：RGBA f32 → PNG；深度/置信度 → 16 位灰度 PNG（高位深保留）
// 浏览器无原生 16 位 PNG 写出，这里把 16 位值拆到 RG 两个 8 位通道（自定义 "gray16-rg" 约定），
// 读取端用 decodeGray16Png 还原；展示端可直接当伪彩图。

export async function encodePng(rgbaF32: Float32Array, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  for (let i = 0, j = 0; i < rgbaF32.length; i += 4, j += 4) {
    img.data[j] = clamp8(rgbaF32[i] * 255);
    img.data[j + 1] = clamp8(rgbaF32[i + 1] * 255);
    img.data[j + 2] = clamp8(rgbaF32[i + 2] * 255);
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

/** f32 数组（给定 maxValue 归一化）→ RG 拆分的 16 位 PNG */
export async function encodeGray16Png(values: Float32Array, width: number, height: number, maxValue: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  const scale = 65535 / Math.max(1e-6, maxValue);
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(0, Math.min(65535, Math.round(values[i] * scale)));
    img.data[i * 4] = v & 0xff;        // 低 8 位
    img.data[i * 4 + 1] = v >> 8;      // 高 8 位
    img.data[i * 4 + 2] = 0;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  // 把归一化尺度写进 blob 的附加 sidecar 由调用方在记录里保存（此处仅约定）
  return blob;
}

export async function decodeGray16Png(blob: Blob, maxValue: number): Promise<Float32Array> {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const out = new Float32Array(bitmap.width * bitmap.height);
  const scale = maxValue / 65535;
  for (let i = 0; i < out.length; i++) {
    const v = data[i * 4] | (data[i * 4 + 1] << 8);
    out[i] = v * scale;
  }
  bitmap.close();
  return out;
}

export async function decodePng(blob: Blob): Promise<ImageData> {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return img;
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

/** 缩略图（最大边 256）JPEG，用于图层列表快速预览 */
export async function makeThumbnail(blob: Blob, maxEdge = 256): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
}
