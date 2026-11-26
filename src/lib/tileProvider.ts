// 浏览器侧 tile 供给：从 IndexedDB Blob 解码 → 对齐 warp → 按 tile 切 f32
// 解码结果做 LRU 缓存，避免同一层在相邻 tile 反复解码
import type { AlignParams, LayerRecord } from '../types';
import { db } from './storage';
import type { LayerTileProvider, TileRect } from '../gpu/fusionEngine';
import { warpLayer } from './alignment';

const MAX_DECODED = 6;

export class BlobTileProvider implements LayerTileProvider {
  readonly layerCount: number;
  readonly aligns: AlignParams[];
  private layers: LayerRecord[];
  private cache = new Map<number, Float32Array>();
  private width: number;
  private height: number;

  constructor(layers: LayerRecord[], aligns: AlignParams[], width: number, height: number) {
    this.layers = layers;
    this.aligns = aligns;
    this.layerCount = layers.length;
    this.width = width;
    this.height = height;
  }

  private async decodeFull(layerIndex: number): Promise<Float32Array> {
    const hit = this.cache.get(layerIndex);
    if (hit) { this.cache.delete(layerIndex); this.cache.set(layerIndex, hit); return hit; }
    const rec = this.layers[layerIndex];
    const blob = await db.getBlob(rec.blobKey);
    if (!blob) throw new Error(`图层数据缺失：${rec.fileName}`);
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    bitmap.close();
    // 归一化 0..1 RGBA
    const rgba = new Float32Array(img.data.length);
    for (let i = 0; i < img.data.length; i++) rgba[i] = img.data[i] / 255;
    // 对齐 warp（CPU 双线性；若为单位变换直接跳过）
    const a = this.aligns[layerIndex];
    const warped = (Math.abs(a.scale - 1) < 1e-4 && Math.abs(a.tx) < 1e-3 && Math.abs(a.ty) < 1e-3)
      ? rgba
      : warpLayer(rgba, bitmap.width, bitmap.height, a);
    if (this.cache.size >= MAX_DECODED) {
      const oldest = this.cache.keys().next().value!;
      this.cache.delete(oldest);
    }
    this.cache.set(layerIndex, warped);
    return warped;
  }

  async loadTile(layerIndex: number, rect: TileRect): Promise<Float32Array> {
    const full = await this.decodeFull(layerIndex);
    const out = new Float32Array(rect.w * rect.h * 4);
    for (let y = 0; y < rect.h; y++) {
      const sy = Math.min(this.height - 1, rect.y + y);
      for (let x = 0; x < rect.w; x++) {
        const sx = Math.min(this.width - 1, rect.x + x);
        const si = (sy * this.width + sx) * 4;
        const di = (y * rect.w + x) * 4;
        out[di] = full[si]; out[di + 1] = full[si + 1];
        out[di + 2] = full[si + 2]; out[di + 3] = full[si + 3];
      }
    }
    return out;
  }
}

/** 内存型 provider（合成栈直接使用，无需 IndexedDB） */
export class MemoryTileProvider implements LayerTileProvider {
  readonly layerCount: number;
  readonly aligns: AlignParams[];
  private images: Float32Array[];
  private width: number;
  private height: number;

  constructor(images: Float32Array[], aligns: AlignParams[], width: number, height: number) {
    this.images = images;
    this.aligns = aligns;
    this.layerCount = images.length;
    this.width = width;
    this.height = height;
  }

  async loadTile(layerIndex: number, rect: TileRect): Promise<Float32Array> {
    const full = this.images[layerIndex];
    const ch = full.length / (this.width * this.height);
    const out = new Float32Array(rect.w * rect.h * 4);
    for (let y = 0; y < rect.h; y++) {
      const sy = Math.min(this.height - 1, rect.y + y);
      for (let x = 0; x < rect.w; x++) {
        const sx = Math.min(this.width - 1, rect.x + x);
        const si = (sy * this.width + sx) * ch;
        const di = (y * rect.w + x) * 4;
        out[di] = full[si]; out[di + 1] = full[si + 1]; out[di + 2] = full[si + 2];
        out[di + 3] = ch === 4 ? full[si + 3] : 1;
      }
    }
    return out;
  }
}
