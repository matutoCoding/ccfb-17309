// 全局状态：轻量发布订阅 store，持久化委托 IndexedDB
import type {
  AppSettings, Batch, Calibration, Comparison, FusionJob, LayerRecord, LayerStack,
} from '../types';
import { db } from '../lib/storage';
import { DEFAULT_APP_SETTINGS } from '../lib/defaults';
import type { GpuCaps } from '../gpu/device';

type Listener = () => void;

class AppStore {
  batches: Batch[] = [];
  stacks: LayerStack[] = [];
  layers: Record<string, LayerRecord[]> = {};
  jobs: Record<string, FusionJob[]> = {};
  calibrations: Calibration[] = [];
  comparisons: Comparison[] = [];
  settings: AppSettings = { ...DEFAULT_APP_SETTINGS };
  gpuCaps: GpuCaps | { supported: false; reason: string } | null = null;
  currentBatchId: string | null = null;
  currentStackId: string | null = null;
  private listeners = new Set<Listener>();

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit(): void {
    this.listeners.forEach((fn) => fn());
  }

  async hydrate(): Promise<void> {
    [this.batches, this.calibrations, this.comparisons] = await Promise.all([
      db.listBatches(),
      db.listCalibrations(),
      db.listComparisons(),
    ]);
    this.batches.sort((a, b) => b.interceptDate.localeCompare(a.interceptDate));
    const saved = await db.loadSettings();
    if (saved) this.settings = { ...DEFAULT_APP_SETTINGS, ...saved };
    this.emit();
  }

  async saveSettings(patch: Partial<AppSettings>): Promise<void> {
    this.settings = { ...this.settings, ...patch };
    await db.saveSettings(this.settings);
    this.emit();
  }

  async refreshBatches(): Promise<void> {
    this.batches = await db.listBatches();
    this.batches.sort((a, b) => b.interceptDate.localeCompare(a.interceptDate));
    this.emit();
  }

  async refreshStacks(batchId: string): Promise<void> {
    this.stacks = await db.listStacks(batchId);
    for (const s of this.stacks) {
      this.layers[s.id] = await db.listLayers(s.id);
      this.jobs[s.id] = await db.listJobs(s.id);
    }
    this.emit();
  }

  async refreshJobs(stackId: string): Promise<void> {
    this.jobs[stackId] = await db.listJobs(stackId);
    this.emit();
  }

  selectBatch(id: string | null): void {
    this.currentBatchId = id;
    this.currentStackId = null;
    this.stacks = [];
    this.emit();
  }

  selectStack(id: string | null): void {
    this.currentStackId = id;
    this.emit();
  }

  get currentStack(): LayerStack | undefined {
    return this.stacks.find((s) => s.id === this.currentStackId);
  }

  get currentLayers(): LayerRecord[] {
    return this.currentStackId ? this.layers[this.currentStackId] ?? [] : [];
  }

  get currentJobs(): FusionJob[] {
    return this.currentStackId ? this.jobs[this.currentStackId] ?? [] : [];
  }
}

export const store = new AppStore();
