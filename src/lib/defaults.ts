// 默认参数集中管理
import type { AppSettings, FusionSettings } from '../types';
import { DEFAULT_GPU_BUDGET } from '../gpu/device';

export const DEFAULT_FUSION_SETTINGS: FusionSettings = {
  algo: 'dff',
  tileSize: 512,
  tileOverlap: 64,
  focusWindow: 2,
  topK: 4,
  haloGuard: 0.8,
  confidenceLowTexture: 0.12,
  confidenceSpecular: 235,
  guideRadius: 4,
  maxGpuBytes: DEFAULT_GPU_BUDGET,
};

export const DEFAULT_APP_SETTINGS: AppSettings = {
  maxGpuBytes: DEFAULT_GPU_BUDGET,
  storageTier: 'balanced',
  focusDir: 'top-to-bottom',
  defaultStepMicron: 2,
  psnrTargetDb: 6,
  depthErrorTargetStep: 1,
  haloTargetPx: 2,
  stressStackLayers: 200,
  stressStackPixels: 24_000_000,
};
