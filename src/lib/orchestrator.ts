// 融合编排：作业状态机 + 分块引擎 + IndexedDB 断点/结果持久化
import type { AlignParams, FusionJob, FusionSettings, LayerStack, TileCheckpoint } from '../types';
import { db, newId } from './storage';
import { GpuFusionEngine, type EngineProgress, type LayerTileProvider } from '../gpu/fusionEngine';
import type { GpuCaps } from '../gpu/device';
import { DEFAULT_FUSION_SETTINGS } from './defaults';
import { encodePng, encodeGray16Png } from './imageCodec';

export class FusionOrchestrator {
  private engine: GpuFusionEngine;
  private active = new Map<string, { aborted: boolean; paused: boolean }>();

  constructor(caps: GpuCaps) {
    this.engine = new GpuFusionEngine(caps);
  }

  static async createJob(stack: LayerStack, settings: FusionSettings): Promise<FusionJob> {
    const tileSize = settings.tileSize;
    const overlap = settings.tileOverlap;
    const cols = Math.ceil(stack.width / (tileSize - overlap));
    const rows = Math.ceil(stack.height / (tileSize - overlap));
    const job: FusionJob = {
      id: newId('job'),
      stackId: stack.id,
      settings,
      status: 'queued',
      progress: 0,
      tilesTotal: cols * rows,
      tilesDone: 0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    await db.saveJob(job);
    return job;
  }

  async run(
    job: FusionJob,
    provider: LayerTileProvider,
    onProgress?: (p: EngineProgress) => void,
  ): Promise<FusionJob> {
    const stack = await db.getStack(job.stackId);
    if (!stack) throw new Error('图层栈不存在');
    const signal = { aborted: false, paused: false };
    this.active.set(job.id, signal);
    job.status = 'running';
    job.updatedAt = Date.now();
    await db.saveJob(job);

    try {
      const outputs = await this.engine.run({
        width: stack.width,
        height: stack.height,
        settings: job.settings,
        provider,
        signal,
        checkpoint: job.checkpoint ?? null,
        onProgress: async (p, cp: TileCheckpoint) => {
          job.progress = p.fraction;
          job.tilesDone = p.tilesDone;
          job.checkpoint = cp;
          job.updatedAt = Date.now();
          onProgress?.(p);
          // 每个 tile 边界落一次断点（轻量元数据写入）
          if (p.phase === 'tile-done') await db.saveJob(job);
        },
        shouldPause: async () => {
          while (signal.paused && !signal.aborted) {
            job.status = 'paused';
            await db.saveJob(job);
            await new Promise((r) => setTimeout(r, 250));
            const live = this.active.get(job.id);
            if (!live || live.aborted) signal.aborted = true;
          }
          if (signal.aborted) throw new Error('融合已中止');
          if (job.status === 'paused') { job.status = 'running'; }
        },
      });

      // 结果编码并持久化
      const colorBlob = await encodePng(outputs.color, stack.width, stack.height);
      const depthBlob = await encodeGray16Png(outputs.depth, stack.width, stack.height, provider.layerCount - 1);
      const confBlob = await encodeGray16Png(outputs.confidence, stack.width, stack.height, 1);
      const contribBlob = new Blob([new Uint16Array(outputs.contrib.buffer).buffer], { type: 'application/octet-stream' });
      const colorKey = `${job.id}_color.png`;
      const depthKey = `${job.id}_depth.png`;
      const confKey = `${job.id}_conf.png`;
      const contribKey = `${job.id}_contrib.bin`;
      await Promise.all([
        db.saveBlob(colorKey, colorBlob),
        db.saveBlob(depthKey, depthBlob),
        db.saveBlob(confKey, confBlob),
        db.saveBlob(contribKey, contribBlob),
      ]);
      job.outputColorKey = colorKey;
      job.outputDepthKey = depthKey;
      job.outputConfidenceKey = confKey;
      job.outputContribKey = contribKey;
      job.status = 'done';
      job.progress = 1;
      job.checkpoint = undefined;
      job.updatedAt = Date.now();
      await db.saveJob(job);
      return job;
    } catch (e) {
      if (signal.aborted) {
        job.status = 'paused'; // 断点已保存，可续算
      } else {
        job.status = 'error';
        job.error = (e as Error).message;
      }
      job.updatedAt = Date.now();
      await db.saveJob(job);
      this.active.delete(job.id);
      throw e;
    } finally {
      if (job.status === 'done') this.active.delete(job.id);
    }
  }

  pause(jobId: string): void {
    const s = this.active.get(jobId);
    if (s) s.paused = true;
  }

  resume(jobId: string): void {
    const s = this.active.get(jobId);
    if (s) s.paused = false;
  }

  abort(jobId: string): void {
    const s = this.active.get(jobId);
    if (s) { s.aborted = true; s.paused = false; }
  }

  isActive(jobId: string): boolean {
    return this.active.has(jobId);
  }
}

export function settingsWithBudget(base: Partial<FusionSettings>, gpuBudget: number): FusionSettings {
  return { ...DEFAULT_FUSION_SETTINGS, ...base, maxGpuBytes: gpuBudget };
}

export type { AlignParams };
