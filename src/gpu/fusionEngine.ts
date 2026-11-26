// WebGPU 分块流式融合引擎
// 关键约束：大图层栈不整摞进显存——按层×分块滚动驻留，tile 边界检查暂停信号并写 checkpoint
import type { AlignParams, FusionSettings, TileCheckpoint } from '../types';
import type { GpuCaps } from './device';
import {
  BOX_SHADER, DFF_DECIDE_SHADER, FOCUS_SHADER, GUIDED_APPLY_SHADER,
  GUIDED_COEF_SHADER, HALO_GUARD_SHADER,
} from './shaders';

const WG = 8;

type BindEntry = { binding?: number; buffer: GPUBuffer; offset?: number; size?: number } | GPUBindGroupEntry;

export interface TileRect { x: number; y: number; w: number; h: number }

export interface EngineProgress {
  tilesTotal: number;
  tilesDone: number;
  layerIndex: number;
  phase: string;
  fraction: number;
}

export interface FusionOutputs {
  color: Float32Array;   // 全图 RGBA f32 0..1
  depth: Float32Array;   // 全图 f32 层索引
  confidence: Float32Array;
  mask: Uint32Array;     // bit0 低纹理 bit1 高光 bit2 空洞
  contrib: Uint32Array;  // pixel × topK
  topK: number;
  rect?: TileRect;
}

export interface LayerTileProvider {
  /** 返回某层某分块的 RGBA f32（0..1），尺寸 = rect.w*rect.h*4；浏览器侧负责解码+对齐 warp */
  loadTile(layerIndex: number, rect: TileRect): Promise<Float32Array>;
  readonly layerCount: number;
  readonly aligns: AlignParams[];
}

interface PipelineSet {
  focus: GPUComputePipeline;
  box: GPUComputePipeline;
  decide: GPUComputePipeline;
  halo: GPUComputePipeline;
  coef: GPUComputePipeline;
  apply: GPUComputePipeline;
}

/** Tile uniform（64 字节，与 shaders.ts 的 WGSL struct 布局一致） */
function writeUniform(
  dv: DataView, rect: TileRect, W: number, H: number, N: number,
  s: FusionSettings, softTemp: number, lowThresh: number,
): void {
  dv.setUint32(0, rect.x, true); dv.setUint32(4, rect.y, true);
  dv.setUint32(8, rect.w, true); dv.setUint32(12, rect.h, true);
  dv.setUint32(16, W, true); dv.setUint32(20, H, true);
  dv.setUint32(24, N, true); dv.setUint32(28, s.focusWindow, true);
  dv.setUint32(32, s.topK, true); dv.setUint32(36, s.guideRadius, true);
  // params1 (offset 40): softTemp, haloGuard, lowThresh, specLum
  dv.setFloat32(40, softTemp, true);
  dv.setFloat32(44, s.haloGuard, true);
  dv.setFloat32(48, lowThresh, true);
  dv.setFloat32(52, 235, true);
  // params2 (offset 56): overlap, eps（引导滤波）
  dv.setFloat32(56, s.tileOverlap, true);
  dv.setFloat32(60, 1e-3 * 255 * 255, true);
}

export class GpuFusionEngine {
  private device: GPUDevice;
  private caps: GpuCaps;
  private pipes: PipelineSet | null = null;

  constructor(caps: GpuCaps) {
    if (!caps.supported) throw new Error('WebGPU 不可用');
    this.caps = caps;
    this.device = caps.device;
  }

  private async ensurePipelines(): Promise<PipelineSet> {
    if (this.pipes) return this.pipes;
    const d = this.device;
    const mk = (code: string) => d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.pipes = {
      focus: mk(FOCUS_SHADER),
      box: mk(BOX_SHADER),
      decide: mk(DFF_DECIDE_SHADER),
      halo: mk(HALO_GUARD_SHADER),
      coef: mk(GUIDED_COEF_SHADER),
      apply: mk(GUIDED_APPLY_SHADER),
    };
    return this.pipes;
  }

  planTiles(fullW: number, fullH: number, tileSize: number, overlap: number): TileRect[] {
    const tiles: TileRect[] = [];
    const step = tileSize - overlap;
    for (let ty = 0; ty < fullH; ty += step) {
      for (let tx = 0; tx < fullW; tx += step) {
        tiles.push({ x: tx, y: ty, w: Math.min(tileSize, fullW - tx), h: Math.min(tileSize, fullH - ty) });
        if (tx + tileSize >= fullW) break;
      }
      if (ty + tileSize >= fullH) break;
    }
    return tiles;
  }

  private makeUniform(rect: TileRect, W: number, H: number, N: number, s: FusionSettings, softTemp: number, lowThresh: number): GPUBuffer {
    const buf = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ab = new ArrayBuffer(64);
    writeUniform(new DataView(ab), rect, W, H, N, s, softTemp, lowThresh);
    this.device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }

  private dispatch(pipe: GPUComputePipeline, entries: BindEntry[], w: number, h: number): void {
    const normalized: GPUBindGroupEntry[] = entries.map((e, i) => {
      if ('resource' in e) return e as GPUBindGroupEntry;
      const be = e as { buffer: GPUBuffer; offset?: number; size?: number };
      const entry: GPUBindGroupEntry = { binding: i, resource: { buffer: be.buffer } };
      if (be.offset !== undefined || be.size !== undefined) {
        entry.resource = { buffer: be.buffer, offset: be.offset ?? 0, size: be.size! };
      }
      return entry;
    });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: normalized });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(w / WG), Math.ceil(h / WG));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private async readF32(buf: GPUBuffer, count: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  private chooseWaveSize(tilePixels: number, budget: number): number {
    const perLayer = tilePixels * 4 * 4; // RGBA f32
    const headroom = budget * 0.4;       // 度量/输出/uniform 留 60%
    return Math.max(2, Math.min(64, Math.max(2, Math.floor(headroom / perLayer))));
  }

  /**
   * 分块融合主循环。每块：
   *  1) 逐层流式载入 tile（波次驻留），焦点度量在 compute shader 计算
   *  2) 度量读回主机做 topK/softmax 决策（数据量小，避免 N 层图像整摞驻留）
   *  3) GPU 光晕正则 → 主机按深度流式 refetch 取色 → GPU 引导滤波补洞
   *  4) feather 权重拼回全图
   */
  async run(opts: {
    width: number;
    height: number;
    settings: FusionSettings;
    provider: LayerTileProvider;
    signal: { paused: boolean; aborted: boolean };
    checkpoint: TileCheckpoint | null;
    onProgress: (p: EngineProgress, cp: TileCheckpoint) => void;
    shouldPause: () => Promise<void>;
  }): Promise<FusionOutputs> {
    const { width: W, height: H, settings: s, provider } = opts;
    const pipes = await this.ensurePipelines();
    const d = this.device;
    const tiles = this.planTiles(W, H, s.tileSize, s.tileOverlap);
    const N = provider.layerCount;

    const outColor = new Float32Array(W * H * 4);
    const outDepth = new Float32Array(W * H);
    const outConf = new Float32Array(W * H);
    const outMask = new Uint32Array(W * H);
    const outContrib = new Uint32Array(W * H * s.topK);
    const outWeight = new Float32Array(W * H);

    const startTile = opts.checkpoint
      ? tiles.findIndex((t) => t.y > opts.checkpoint!.tileY || (t.y === opts.checkpoint!.tileY && t.x >= opts.checkpoint!.tileX))
      : 0;

    for (let ti = Math.max(0, startTile); ti < tiles.length; ti++) {
      if (opts.signal.aborted) throw new Error('融合已中止');
      await opts.shouldPause();
      const rect = tiles[ti];
      const tp = rect.w * rect.h;

      const lowThresh = await this.estimateLowThreshold(rect, provider, s);
      const softTemp = Math.max(1, lowThresh * 0.02);
      const uniform = this.makeUniform(rect, rect.w, rect.h, N, s, softTemp, lowThresh);

      // 度量缓冲（N 层累计）+ 图像波次缓冲
      const measureBuf = d.createBuffer({ size: tp * 4 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const waveSize = this.chooseWaveSize(tp, s.maxGpuBytes);
      const waveBuf = d.createBuffer({
        size: tp * 4 * 4 * Math.min(N, waveSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      for (let z = 0; z < N; z++) {
        const data = await provider.loadTile(z, rect);
        const slot = z % waveSize;
        d.queue.writeBuffer(waveBuf, slot * tp * 16, data, 0, data.length);
        const mBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.dispatch(pipes.focus, [
          { buffer: uniform },
          { buffer: waveBuf, offset: slot * tp * 16, size: tp * 16 },
          { buffer: mBuf },
        ], rect.w, rect.h);
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(mBuf, 0, measureBuf, z * tp * 4, tp * 4);
        d.queue.submit([enc.finish()]);
        mBuf.destroy();
        opts.onProgress(
          { tilesTotal: tiles.length, tilesDone: ti, layerIndex: z, phase: 'focus-measure', fraction: (ti + z / N) / tiles.length },
          { tileX: rect.x, tileY: rect.y, layerIndex: z },
        );
      }

      const decision = await this.decideOnMeasure(measureBuf, tp, N, s, softTemp, lowThresh);
      const tile = await this.composeTile(rect, provider, decision, pipes, s, softTemp, lowThresh, uniform);

      measureBuf.destroy();
      waveBuf.destroy();
      uniform.destroy();

      this.stitchTile(tile, rect, W, s, outColor, outDepth, outConf, outMask, outContrib, outWeight);

      opts.onProgress(
        { tilesTotal: tiles.length, tilesDone: ti + 1, layerIndex: N, phase: 'tile-done', fraction: (ti + 1) / tiles.length },
        { tileX: rect.x, tileY: rect.y, layerIndex: N },
      );
    }

    // 统一权重归一化
    for (let p = 0; p < W * H; p++) {
      const wsum = outWeight[p] || 1;
      outColor[p * 4] /= wsum; outColor[p * 4 + 1] /= wsum; outColor[p * 4 + 2] /= wsum;
      outDepth[p] /= wsum;
      outConf[p] /= wsum;
    }
    return { color: outColor, depth: outDepth, confidence: outConf, mask: outMask, contrib: outContrib, topK: s.topK };
  }

  /** 低纹理阈值：抽 3 层 tile 度量，取像素最大度量中位数 × 比例系数（GPU 度量） */
  private async estimateLowThreshold(rect: TileRect, provider: LayerTileProvider, s: FusionSettings): Promise<number> {
    const pipes = await this.ensurePipelines();
    const N = provider.layerCount;
    const tp = rect.w * rect.h;
    const samples = [Math.floor(N * 0.25), N >> 1, Math.floor(N * 0.75)];
    const maxes = new Float32Array(tp);
    const uniform = this.makeUniform(rect, rect.w, rect.h, N, s, 1, 0);
    for (const z of samples) {
      const tile = await provider.loadTile(z, rect);
      const imgBuf = this.device.createBuffer({ size: tp * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const mBuf = this.device.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.device.queue.writeBuffer(imgBuf, 0, tile.buffer.slice(0));
      this.dispatch(pipes.focus, [{ buffer: uniform }, { buffer: imgBuf }, { buffer: mBuf }], rect.w, rect.h);
      const m = await this.readF32(mBuf, tp);
      for (let i = 0; i < tp; i++) maxes[i] = Math.max(maxes[i], m[i]);
      imgBuf.destroy(); mBuf.destroy();
    }
    uniform.destroy();
    const sorted = Float32Array.from(maxes).sort();
    return sorted[sorted.length >> 1] * s.confidenceLowTexture;
  }

  /** 主机侧 topK/softmax 决策（度量缓冲量小，读回一次；WGSL 版 DFF_DECIDE_SHADER 已备，用于无分块快路径） */
  private async decideOnMeasure(
    measureBuf: GPUBuffer, tp: number, N: number, s: FusionSettings,
    softTemp: number, lowThresh: number,
  ): Promise<{ depth: Float32Array; conf: Float32Array; mask: Uint32Array; contrib: Uint32Array }> {
    const measures = await this.readF32(measureBuf, tp * N);
    const depth = new Float32Array(tp);
    const conf = new Float32Array(tp);
    const mask = new Uint32Array(tp);
    const contrib = new Uint32Array(tp * s.topK);
    for (let p = 0; p < tp; p++) {
      const vals: { z: number; m: number }[] = [];
      let mx = 0;
      for (let z = 0; z < N; z++) {
        const m = measures[z * tp + p];
        vals.push({ z, m });
        mx = Math.max(mx, m);
      }
      vals.sort((a, b) => b.m - a.m);
      const K = Math.min(s.topK, N);
      const top = vals.slice(0, K);
      let wsum = 0;
      const wts = top.map((v) => { const e = Math.exp((v.m - mx) / softTemp); wsum += e; return e; });
      let d = 0;
      for (let j = 0; j < K; j++) {
        const wt = wts[j] / wsum;
        d += wt * top[j].z;
        contrib[p * s.topK + j] = top[j].z;
      }
      depth[p] = d;
      conf[p] = Math.min(1, 0.55 * (wts[0] / wsum) + 0.45 * Math.min(1, mx / Math.max(1e-6, lowThresh * 1.5)));
      if (mx < lowThresh) mask[p] |= 1 | 4;
    }
    return { depth, conf, mask, contrib };
  }

  /** 块内合成：GPU 光晕正则 → 流式 refetch 取色 → 高光掩膜 → GPU 引导补洞 */
  private async composeTile(
    rect: TileRect, provider: LayerTileProvider,
    decision: { depth: Float32Array; conf: Float32Array; mask: Uint32Array; contrib: Uint32Array },
    pipes: PipelineSet, s: FusionSettings, softTemp: number, lowThresh: number, uniform: GPUBuffer,
  ): Promise<FusionOutputs> {
    const d = this.device;
    const tp = rect.w * rect.h;
    const N = provider.layerCount;

    // 1) 光晕正则（GPU）
    const depthIn = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const depthOut = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const maskBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(depthIn, 0, decision.depth.buffer.slice(0));
    d.queue.writeBuffer(maskBuf, 0, decision.mask.buffer.slice(0));
    this.dispatch(pipes.halo, [{ buffer: uniform }, { buffer: depthIn }, { buffer: maskBuf }, { buffer: depthOut }], rect.w, rect.h);
    const depth = await this.readF32(depthOut, tp);
    depthIn.destroy(); depthOut.destroy(); maskBuf.destroy();

    // 2) 预取本块涉及的层（深度相邻），缓存最近若干层
    const needed = new Set<number>();
    for (let p = 0; p < tp; p++) {
      const z0 = Math.min(N - 1, Math.max(0, Math.floor(depth[p])));
      needed.add(z0); needed.add(Math.min(N - 1, z0 + 1));
    }
    const cache = new Map<number, Float32Array>();
    for (const z of needed) cache.set(z, await provider.loadTile(z, rect));

    // 3) refetch 取色（深度线性插值两层）
    const color = new Float32Array(tp * 4);
    for (let p = 0; p < tp; p++) {
      const dd = Math.min(N - 1.001, Math.max(0, depth[p]));
      const z0 = Math.floor(dd), t = dd - z0;
      const a = cache.get(z0)!, b = cache.get(Math.min(N - 1, z0 + 1))!;
      for (let c = 0; c < 3; c++) color[p * 4 + c] = a[p * 4 + c] * (1 - t) + b[p * 4 + c] * t;
      color[p * 4 + 3] = 1;
    }

    // 4) 高光掩膜：先按层累计每像素亮度统计，再判定（避免 O(pixels×layers) 次异步）
    const lumSum = new Float32Array(tp);
    const lumSq = new Float32Array(tp);
    for (const z of Array.from(cache.keys()).sort((a, b) => a - b)) {
      const tile = cache.get(z)!;
      for (let p = 0; p < tp; p++) {
        const L = 0.299 * tile[p * 4] + 0.587 * tile[p * 4 + 1] + 0.114 * tile[p * 4 + 2];
        lumSum[p] += L; lumSq[p] += L * L;
      }
    }
    const mask = Uint32Array.from(decision.mask);
    for (let p = 0; p < tp; p++) {
      const mean = lumSum[p] / N;
      const variance = lumSq[p] / N - mean * mean;
      if (mean > 235 && variance < 25) {
        mask[p] |= 2 | 4;
        decision.conf[p] = Math.min(decision.conf[p], 0.25);
      }
    }

    // 5) 引导滤波补深度洞（GPU：coef → box×2 → apply）
    const guide = new Float32Array(tp);
    for (let p = 0; p < tp; p++) guide[p] = 0.299 * color[p * 4] + 0.587 * color[p * 4 + 1] + 0.114 * color[p * 4 + 2];
    const filled = await this.guidedFillGpu(guide, depth, mask, rect, pipes, s, uniform);

    return { rect, color, depth: filled, confidence: decision.conf, mask, contrib: decision.contrib, topK: s.topK };
  }

  private async guidedFillGpu(
    guide: Float32Array, depth: Float32Array, mask: Uint32Array,
    rect: TileRect, pipes: PipelineSet, s: FusionSettings, uniform: GPUBuffer,
  ): Promise<Float32Array> {
    const d = this.device;
    const tp = rect.w * rect.h;
    const guideBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const depthBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const aBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE });
    const bBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE });
    d.queue.writeBuffer(guideBuf, 0, guide.buffer.slice(0));
    d.queue.writeBuffer(depthBuf, 0, depth.buffer.slice(0));
    this.dispatch(pipes.coef, [{ buffer: uniform }, { buffer: guideBuf }, { buffer: depthBuf }, { buffer: aBuf }, { buffer: bBuf }], rect.w, rect.h);
    const meanA = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE });
    const meanB = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE });
    this.dispatch(pipes.box, [{ buffer: uniform }, { buffer: aBuf }, { buffer: meanA }], rect.w, rect.h);
    this.dispatch(pipes.box, [{ buffer: uniform }, { buffer: bBuf }, { buffer: meanB }], rect.w, rect.h);
    const maskBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBuf = d.createBuffer({ size: tp * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    d.queue.writeBuffer(maskBuf, 0, mask.buffer.slice(0));
    d.queue.writeBuffer(outBuf, 0, depth.buffer.slice(0));
    this.dispatch(pipes.apply, [{ buffer: uniform }, { buffer: meanA }, { buffer: meanB }, { buffer: guideBuf }, { buffer: maskBuf }, { buffer: outBuf }], rect.w, rect.h);
    const out = await this.readF32(outBuf, tp);
    for (const b of [guideBuf, depthBuf, aBuf, bBuf, meanA, meanB, maskBuf, outBuf]) b.destroy();
    void s;
    return out;
  }

  /** feather 拼回：存加权和，最终在 run() 末尾统一归一 */
  private stitchTile(
    tile: FusionOutputs, rect: TileRect, W: number, s: FusionSettings,
    outColor: Float32Array, outDepth: Float32Array, outConf: Float32Array,
    outMask: Uint32Array, outContrib: Uint32Array, outWeight: Float32Array,
  ): void {
    const overlap = Math.max(1, s.tileOverlap);
    const weightAt = (lx: number, ly: number) => {
      const wx = Math.min(1, (Math.min(lx + 1, rect.w - lx) + overlap * 0.5) / (overlap + 1));
      const wy = Math.min(1, (Math.min(ly + 1, rect.h - ly) + overlap * 0.5) / (overlap + 1));
      return Math.max(0.05, wx * wy);
    };
    for (let ly = 0; ly < rect.h; ly++) {
      for (let lx = 0; lx < rect.w; lx++) {
        const gp = (rect.y + ly) * W + (rect.x + lx);
        const lp = ly * rect.w + lx;
        const wt = weightAt(lx, ly);
        outColor[gp * 4] += tile.color[lp * 4] * wt;
        outColor[gp * 4 + 1] += tile.color[lp * 4 + 1] * wt;
        outColor[gp * 4 + 2] += tile.color[lp * 4 + 2] * wt;
        outDepth[gp] += tile.depth[lp] * wt;
        outConf[gp] += tile.confidence[lp] * wt;
        for (let k = 0; k < tile.topK; k++) outContrib[gp * tile.topK + k] = tile.contrib[lp * tile.topK + k];
        outMask[gp] = tile.mask[lp];
        outWeight[gp] += wt;
      }
    }
  }
}
