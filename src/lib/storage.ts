// IndexedDB 持久化：元数据表 + Blob 对象库
import type {
  Batch, LayerRecord, LayerStack, FusionJob, Calibration,
  Measurement, Comparison, AppSettings, GroundTruth,
} from '../types';
import { uid } from './math';

const DB_NAME = 'focus-stacker-db';
const DB_VERSION = 1;

const STORES = {
  batches: 'batches',
  stacks: 'stacks',
  layers: 'layers',
  jobs: 'jobs',
  calibrations: 'calibrations',
  measurements: 'measurements',
  comparisons: 'comparisons',
  blobs: 'blobs',
  settings: 'settings',
  groundTruth: 'groundTruth',
} as const;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORES.batches)) db.createObjectStore(STORES.batches, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(STORES.stacks)) {
        const s = db.createObjectStore(STORES.stacks, { keyPath: 'id' });
        s.createIndex('batchId', 'batchId', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.layers)) {
        const s = db.createObjectStore(STORES.layers, { keyPath: 'id' });
        s.createIndex('stackId', 'stackId', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.jobs)) {
        const s = db.createObjectStore(STORES.jobs, { keyPath: 'id' });
        s.createIndex('stackId', 'stackId', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.calibrations)) db.createObjectStore(STORES.calibrations, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(STORES.measurements)) {
        const s = db.createObjectStore(STORES.measurements, { keyPath: 'id' });
        s.createIndex('stackId', 'stackId', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.comparisons)) db.createObjectStore(STORES.comparisons, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(STORES.blobs)) db.createObjectStore(STORES.blobs); // 键即字符串 key
      if (!db.objectStoreNames.contains(STORES.settings)) db.createObjectStore(STORES.settings);
      if (!db.objectStoreNames.contains(STORES.groundTruth)) db.createObjectStore(STORES.groundTruth, { keyPath: 'depthKey' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then((db) => new Promise<T>((resolve, reject) => {
    const t = db.transaction(store, mode);
    const req = fn(t.objectStore(store));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

export async function putRecord<T>(store: keyof typeof STORES, value: T): Promise<T> {
  return tx(STORES[store], 'readwrite', (s) => s.put(value as unknown as object) as unknown as IDBRequest<T>);
}

export async function getRecord<T>(store: keyof typeof STORES, key: string): Promise<T | undefined> {
  return tx<T | undefined>(STORES[store], 'readonly', (s) => s.get(key) as IDBRequest<T | undefined>);
}

export async function deleteRecord(store: keyof typeof STORES, key: string): Promise<void> {
  await tx(STORES[store], 'readwrite', (s) => s.delete(key));
}

export async function getAll<T>(store: keyof typeof STORES): Promise<T[]> {
  return tx<T[]>(STORES[store], 'readonly', (s) => s.getAll() as IDBRequest<T[]>);
}

export async function getAllByIndex<T>(store: keyof typeof STORES, index: string, value: string): Promise<T[]> {
  const db = await openDb();
  return new Promise<T[]>((resolve, reject) => {
    const t = db.transaction(STORES[store], 'readonly');
    const req = t.objectStore(STORES[store]).index(index).getAll(value);
    req.onsuccess = () => resolve(req.result as T[]);
    req.onerror = () => reject(req.error);
  });
}

// ---- 类型化便捷 API ----
export const db = {
  saveBatch: (b: Batch) => putRecord('batches', b),
  listBatches: () => getAll<Batch>('batches'),
  getBatch: (id: string) => getRecord<Batch>('batches', id),
  deleteBatch: async (id: string) => {
    const stacks = await getAllByIndex<LayerStack>('stacks', 'batchId', id);
    for (const st of stacks) await deleteStack(st.id);
    await deleteRecord('batches', id);
  },

  saveStack: (s: LayerStack) => putRecord('stacks', s),
  listStacks: (batchId: string) => getAllByIndex<LayerStack>('stacks', 'batchId', batchId),
  getStack: (id: string) => getRecord<LayerStack>('stacks', id),

  saveLayer: (l: LayerRecord) => putRecord('layers', l),
  listLayers: (stackId: string) =>
    getAllByIndex<LayerRecord>('layers', 'stackId', stackId).then((ls) => ls.sort((a, b) => a.index - b.index)),
  deleteLayer: (id: string) => deleteRecord('layers', id),

  saveJob: (j: FusionJob) => putRecord('jobs', j),
  listJobs: (stackId: string) => getAllByIndex<FusionJob>('jobs', 'stackId', stackId),
  getJob: (id: string) => getRecord<FusionJob>('jobs', id),

  saveCalibration: (c: Calibration) => putRecord('calibrations', c),
  listCalibrations: () => getAll<Calibration>('calibrations'),
  deleteCalibration: (id: string) => deleteRecord('calibrations', id),

  saveMeasurement: (m: Measurement) => putRecord('measurements', m),
  listMeasurements: (stackId: string) => getAllByIndex<Measurement>('measurements', 'stackId', stackId),
  deleteMeasurement: (id: string) => deleteRecord('measurements', id),

  saveComparison: (c: Comparison) => putRecord('comparisons', c),
  listComparisons: () => getAll<Comparison>('comparisons'),

  saveBlob: async (key: string, blob: Blob): Promise<string> =>
    tx(STORES.blobs, 'readwrite', (s) => s.put(blob, key) as IDBRequest<string>),
  getBlob: (key: string) => tx<Blob | undefined>(STORES.blobs, 'readonly', (s) => s.get(key) as IDBRequest<Blob | undefined>),
  deleteBlob: (key: string) => tx(STORES.blobs, 'readwrite', (s) => s.delete(key) as IDBRequest<undefined>),

  async saveSettings(s: AppSettings): Promise<void> { await tx(STORES.settings, 'readwrite', (st) => st.put(s, 'app')); },
  async loadSettings(): Promise<AppSettings | undefined> {
    return tx<AppSettings | undefined>(STORES.settings, 'readonly', (s) => s.get('app') as IDBRequest<AppSettings | undefined>);
  },

  saveGroundTruth: (g: GroundTruth) => putRecord('groundTruth', g),
  getGroundTruth: (depthKey: string) => getRecord<GroundTruth>('groundTruth', depthKey),

  /** 估算已用存储（字节） */
  async estimateUsage(): Promise<{ usage: number; quota: number }> {
    if (navigator.storage?.estimate) {
      const e = await navigator.storage.estimate();
      return { usage: e.usage ?? 0, quota: e.quota ?? 0 };
    }
    return { usage: 0, quota: 0 };
  },
};

export async function deleteStack(stackId: string): Promise<void> {
  const layers = await db.listLayers(stackId);
  for (const l of layers) {
    await db.deleteBlob(l.blobKey).catch(() => undefined);
    await db.deleteBlob(l.thumbKey).catch(() => undefined);
    await deleteRecord('layers', l.id);
  }
  const jobs = await db.listJobs(stackId);
  for (const j of jobs) {
    for (const k of [j.outputColorKey, j.outputDepthKey, j.outputConfidenceKey, j.outputContribKey]) {
      if (k) await db.deleteBlob(k).catch(() => undefined);
    }
    await deleteRecord('jobs', j.id);
  }
  const measures = await db.listMeasurements(stackId);
  for (const m of measures) await deleteRecord('measurements', m.id);
  await deleteRecord('stacks', stackId);
}

export function newId(prefix: string): string {
  return uid(prefix);
}
