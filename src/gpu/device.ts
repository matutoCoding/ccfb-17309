// WebGPU 设备管理与能力探测
export interface GpuCaps {
  supported: boolean;
  adapter: GPUAdapter;
  device: GPUDevice;
  maxStorageBuffer: number;
  maxStorageTexture: number;
  maxComputeWG: [number, number, number];
  maxBufferSize: number;
  limits: GPUSupportedLimits;
  info?: GPUAdapterInfo;
}

export async function requestGpu(): Promise<GpuCaps | { supported: false; reason: string }> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return { supported: false, reason: '当前浏览器不支持 WebGPU（需 Chrome/Edge 113+ 或对应版本 Safari）' };
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { supported: false, reason: `GPU 适配器请求失败：${(e as Error).message}` };
  }
  if (!adapter) return { supported: false, reason: '未找到可用 GPU 适配器' };
  const required: GPUDeviceDescriptor = {
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
    },
  };
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice(required);
  } catch {
    device = await adapter.requestDevice(); // 保守重试
  }
  device.lost.then((info) => {
    console.error('[WebGPU] device lost:', info.message);
  });
  return {
    supported: true,
    adapter,
    device,
    maxStorageBuffer: device.limits.maxStorageBufferBindingSize,
    maxStorageTexture: device.limits.maxStorageTexturesPerShaderStage ?? 8,
    maxComputeWG: [
      device.limits.maxComputeWorkgroupSizeX,
      device.limits.maxComputeWorkgroupSizeY,
      device.limits.maxComputeWorkgroupSizeZ,
    ],
    maxBufferSize: device.limits.maxBufferSize,
    limits: device.limits,
    info: adapter.info,
  };
}

/** 按显存预算估算可同时驻留的层数（每层按 RGBA8 计） */
export function layersInBudget(
  width: number, height: number, bytesPerPixel: number, budgetBytes: number, overhead = 0.35,
): number {
  const perLayer = width * height * bytesPerPixel;
  const usable = budgetBytes * (1 - overhead);
  return Math.max(2, Math.floor(usable / perLayer));
}

/** 8GB 卡的保守预算默认值（用户可调） */
export const DEFAULT_GPU_BUDGET = 6 * 1024 ** 3;

export function makeBuffer(device: GPUDevice, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
}
