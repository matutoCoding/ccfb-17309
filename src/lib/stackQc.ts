// 图层栈导入：文件名/EXIF 对焦排序、缺层/重复层/曝光突变检测
import type { LayerFlag } from '../types';
import { mean, median, stddev } from './math';

export interface ImportedFile {
  fileName: string;
  focusPosition?: number; // 来自 EXIF
  exposureValue: number; // 平均亮度
  width: number;
  height: number;
  bitsPerSample: 8 | 16;
  mimeType: string;
}

export interface QcResult {
  ordered: Array<{ file: ImportedFile; index: number; focusPosition: number }>;
  flags: Map<string, { flag: LayerFlag; reason: string }>;
}

/** 从文件名抽取序号：STACK_0001.jpg / IMG_12_023.tif / 2024-001-a17.png */
export function numericToken(name: string): number {
  const all = name.match(/\d+/g);
  if (!all || all.length === 0) return NaN;
  return Number(all[all.length - 1]);
}

/** 按 EXIF 对焦位置优先，其次文件名序号排序 */
export function orderLayers(files: ImportedFile[]): QcResult {
  const flags = new Map<string, { flag: LayerFlag; reason: string }>();
  const withToken = files.map((f) => ({
    f,
    token: Number.isFinite(f.focusPosition) ? f.focusPosition! : numericToken(f.fileName),
  }));
  withToken.sort((a, b) => a.token - b.token);

  // 重复 token / 重复曝光指纹（均值高度近似的相邻层）
  const seen = new Map<number, string>();
  for (const { f, token } of withToken) {
    if (seen.has(token)) {
      flags.set(f.fileName, { flag: 'duplicate', reason: `对焦位置 ${token} 与 ${seen.get(token)} 重复` });
    } else {
      seen.set(token, f.fileName);
    }
  }

  // 缺层：token 等间距序列中的空档
  const tokens = withToken.map((x) => x.token).filter(Number.isFinite);
  if (tokens.length >= 3) {
    const diffs = new Map<number, number>();
    for (let i = 1; i < tokens.length; i++) {
      const d = tokens[i] - tokens[i - 1];
      diffs.set(d, (diffs.get(d) ?? 0) + 1);
    }
    let step = 1, best = 0;
    for (const [d, c] of diffs) if (c > best) { best = c; step = d; }
    const present = new Set(tokens);
    for (let i = 1; i < withToken.length; i++) {
      const gap = withToken[i].token - withToken[i - 1].token;
      if (gap > step * 1.8) {
        const missing = Math.round(gap / step) - 1;
        flags.set(withToken[i].f.fileName, {
          flag: 'missing',
          reason: `前序缺 ${missing} 层（位置 ${withToken[i - 1].token} → ${withToken[i].token}）`,
        });
      }
    }
  }

  // 曝光突变：相邻层亮度差分中的离群（差分超过 MAD 鲁棒尺度 3 倍；
  // 若鲁棒尺度退化到 0——绝大多数层亮度一致——则回退到绝对阈值）
  const lums = withToken.map((x) => x.f.exposureValue);
  if (withToken.length >= 4) {
    const diffs: number[] = [];
    for (let i = 1; i < withToken.length; i++) diffs.push(lums[i] - lums[i - 1]);
    const absDev = diffs.map((d) => Math.abs(d - median(diffs)));
    const robustSigma = 1.4826 * median(absDev);
    const absThresh = 30; // 退化（多数层一致）时的绝对亮度跳变阈值
    for (let i = 0; i < diffs.length; i++) {
      const isOutlier = robustSigma > 1e-6
        ? Math.abs(diffs[i]) > 3 * robustSigma
        : Math.abs(diffs[i]) > absThresh;
      if (isOutlier) {
        const idx = Math.abs(lums[i + 1] - mean(lums)) > Math.abs(lums[i] - mean(lums)) ? i + 1 : i;
        const name = withToken[idx].f.fileName;
        // 同一文件可同时带缺层/曝光等多个问题；曝光优先级不覆盖已有标记，而是单独标记
        if (!flags.has(name)) {
          flags.set(name, { flag: 'exposure-jump', reason: `相邻亮度跳变 ${diffs[i].toFixed(1)}（鲁棒尺度 ${robustSigma.toFixed(1)}）` });
        } else if (flags.get(name)!.flag !== 'exposure-jump') {
          flags.set(name, { flag: 'exposure-jump', reason: `${flags.get(name)!.flag} + 曝光跳变 ${diffs[i].toFixed(1)}` });
        }
      }
    }
  }

  const ordered = withToken.map((x, i) => ({
    file: x.f,
    index: i,
    focusPosition: Number.isFinite(x.f.focusPosition) ? x.f.focusPosition! : x.token,
  }));
  return { ordered, flags };
}

/** 由 RGBA 图像算平均亮度（导入缩略图时调用） */
export function meanLuminance(rgba: Float32Array | Uint8ClampedArray, pixels: number): number {
  let s = 0;
  const stride = rgba.length / pixels;
  for (let p = 0; p < pixels; p++) {
    s += 0.299 * rgba[p * stride] + 0.587 * rgba[p * stride + 1] + 0.114 * rgba[p * stride + 2];
  }
  return s / pixels;
}
