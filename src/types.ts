// 全局数据模型与类型定义

export type ViewPoint = 'dorsal' | 'lateral' | 'ventral';
export const VIEW_POINTS: { id: ViewPoint; label: string }[] = [
  { id: 'dorsal', label: '背面' },
  { id: 'lateral', label: '侧面' },
  { id: 'ventral', label: '腹面' },
];

export type LayerFlag = 'ok' | 'missing' | 'duplicate' | 'exposure-jump' | 'excluded';

export interface Batch {
  id: string;
  code: string; // 截获批次号
  interceptDate: string; // ISO 日期
  cargo: string; // 来源货物
  hostPlant: string; // 寄主植物
  species?: string; // 初步鉴定/复核种名
  notes?: string;
  createdAt: number;
  updatedAt: number;
}

export interface LayerStack {
  id: string;
  batchId: string;
  specimenId: string; // 虫体编号
  viewpoint: ViewPoint;
  width: number;
  height: number;
  stepMicron: number; // 对焦步距（微米）
  lensMagnification: number; // 放大倍率（用于测微尺标定匹配）
  startFocusPosition?: number; // 起始对焦位置
  layerIds: string[]; // 按对焦顺序
  createdAt: number;
  updatedAt: number;
}

export interface LayerRecord {
  id: string;
  stackId: string;
  index: number; // 对焦顺序（0 = 最近/最远按约定，见 focusDir）
  fileName: string;
  mimeType: string;
  bitsPerSample: 8 | 16;
  width: number;
  height: number;
  focusPosition: number; // EXIF/文件名解析出的对焦位置
  exposureValue: number; // 平均亮度，用于曝光突变检测
  flag: LayerFlag;
  flagReason?: string;
  blobKey: string; // IndexedDB 图片 Blob 键
  thumbKey: string; // 缩略图 Blob 键（jpeg）
  exif?: Record<string, number | string>;
}

export type FusionAlgo = 'dff' | 'laplacian-pyramid';

export interface FusionSettings {
  algo: FusionAlgo;
  tileSize: number; // 分块边长（像素）
  tileOverlap: number; // 分块重叠（像素）
  focusWindow: number; // 焦点度量窗口半径
  topK: number; // 软选择层数（透明膜翅多层信号融合）
  haloGuard: number; // 光晕抑制强度 0..1
  confidenceLowTexture: number; // 低纹理置信度阈值
  confidenceSpecular: number; // 高光掩膜阈值
  guideRadius: number; // 引导滤波半径
  maxGpuBytes: number; // 显存预算
}

export type FusionStatus = 'queued' | 'running' | 'paused' | 'done' | 'error';

export interface TileCheckpoint {
  tileY: number;
  tileX: number;
  layerIndex: number;
}

export interface FusionJob {
  id: string;
  stackId: string;
  settings: FusionSettings;
  status: FusionStatus;
  progress: number; // 0..1
  tilesTotal: number;
  tilesDone: number;
  checkpoint?: TileCheckpoint;
  outputColorKey?: string;
  outputDepthKey?: string;
  outputConfidenceKey?: string;
  outputContribKey?: string; // 每像素贡献层（uint16，低16位主贡献层）
  residualStats?: AlignResidual[];
  metrics?: FusionMetrics;
  error?: string;
  createdAt: number;
  updatedAt: number;
}

export interface AlignParams {
  // 逐层相似变换：x' = s*(cosθ x - sinθ y) + tx; y' = s*(sinθ x + cosθ y) + ty
  scale: number;
  angle: number; // 弧度
  tx: number;
  ty: number;
}

export interface AlignResidual {
  layerIndex: number;
  ssd: number; // 对齐后 SSD（归一化）
  scale: number;
  tx: number;
  ty: number;
  overThreshold: boolean;
}

export interface ControlPoint {
  layerIndex: number;
  // 参考层坐标 -> 当前层坐标
  refX: number;
  refY: number;
  curX: number;
  curY: number;
}

export interface Calibration {
  id: string;
  magnification: number; // 物镜/总放大倍率
  pixelsPerMm: number; // 像素当量
  source: string; // 镜台测微尺记录
  createdAt: number;
}

export type MeasureKind = 'length' | 'width' | 'angle' | 'curve';

export interface Measurement {
  id: string;
  stackId: string;
  kind: MeasureKind;
  label: string;
  // 归一化坐标 0..1（相对合成图），曲线为多个点
  points: { x: number; y: number }[];
  value: number; // 长度 mm / 角度 deg / 曲线长 mm
  value3d?: number; // 按深度折算的三维长度
  createdAt: number;
}

export interface Comparison {
  id: string;
  stackIdA: string;
  stackIdB: string;
  viewpoint: ViewPoint;
  magnification: number;
  opinion: string; // 鉴定意见
  measureIds: string[];
  createdAt: number;
}

export interface AppSettings {
  maxGpuBytes: number; // 显存预算（默认 6GB）
  storageTier: 'best' | 'balanced' | 'low';
  focusDir: 'top-to-bottom' | 'bottom-to-top'; // 文件名递增方向与对焦方向
  defaultStepMicron: number;
  psnrTargetDb: number;
  depthErrorTargetStep: number;
  haloTargetPx: number;
  stressStackLayers: number; // 压力测试栈层数
  stressStackPixels: number; // 压力测试单层像素
}

export interface FusionMetrics {
  fusedPsnrDb: number;
  bestSinglePsnrDb: number;
  gainDb: number;
  depthMedianAbsErrStep: number;
  haloWidthPxMedian: number;
  lowTextureMaskRatio: number;
  specularMaskRatio: number;
  holeFilledRatio: number;
  pass: boolean;
  reasons: string[];
}

export interface SyntheticSpec {
  width: number;
  height: number;
  layers: number;
  stepMicron: number;
  breathScalePct: number; // 变焦呼吸最大缩放 %
  jitterPx: number; // 导轨抖动像素
  seed: number;
  includeWing: boolean;
  includeSetae: boolean;
  includePunctures: boolean;
}

export interface GroundTruth {
  depthKey: string; // 高度场（uint16 归一化到层数）
  colorKey: string; // 全清晰真值图
  maskKey: string; // 虫体掩膜
  spec: SyntheticSpec;
}
