// 测量：镜台测微尺标定像素当量，长度/宽度/角度/曲线长，斜置部位按深度算三维长度
import type { Calibration, Measurement } from '../types';

export interface PixelPoint { x: number; y: number }

/** 由镜台测微尺刻度读数建立/校验标定：已知物理长度(mm)对应的像素跨度 */
export function calibrate(knownLengthMm: number, pixelSpan: number, magnification: number, source: string): Omit<Calibration, 'id' | 'createdAt'> {
  if (pixelSpan <= 0) throw new Error('像素跨度必须为正数');
  return {
    magnification,
    pixelsPerMm: pixelSpan / knownLengthMm,
    source,
  };
}

export function pickCalibration(cals: Calibration[], magnification: number): Calibration | undefined {
  if (cals.length === 0) return undefined;
  let best = cals[0], bestDiff = Infinity;
  for (const c of cals) {
    const diff = Math.abs(c.magnification - magnification);
    if (diff < bestDiff) { bestDiff = diff; best = c; }
  }
  return best;
}

function pxToMm(px: number, cal: Calibration): number {
  return px / cal.pixelsPerMm;
}

function dist2d(a: PixelPoint, b: PixelPoint): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/** 深度差 → 物理 z 差（mm）：depth 单位是层索引，stepMicron 是每层步距 */
function depthDeltaMm(depthA: number, depthB: number, stepMicron: number): number {
  return Math.abs(depthA - depthB) * stepMicron / 1000;
}

/** 直线长度（含三维折算） */
export function measureLength(
  a: PixelPoint, b: PixelPoint,
  depthAt: (x: number, y: number) => number,
  cal: Calibration, stepMicron: number,
): { planar: number; length3d: number } {
  const planarPx = dist2d(a, b);
  const planar = pxToMm(planarPx, cal);
  const dz = depthDeltaMm(depthAt(a.x, a.y), depthAt(b.x, b.y), stepMicron);
  return { planar, length3d: Math.hypot(planar, dz) };
}

/** 宽度：两点直线距离（语义上与长度同算法，单独类型便于报表分组） */
export function measureWidth(
  a: PixelPoint, b: PixelPoint, cal: Calibration,
): number {
  return pxToMm(dist2d(a, b), cal);
}

/** 角度：顶点 b，射线 ba 与 bc 的夹角（度） */
export function measureAngle(a: PixelPoint, b: PixelPoint, c: PixelPoint): number {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const L1 = Math.hypot(v1.x, v1.y), L2 = Math.hypot(v2.x, v2.y);
  if (L1 < 1e-9 || L2 < 1e-9) return 0;
  const cos = Math.min(1, Math.max(-1, dot / (L1 * L2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** 曲线长（含三维）：沿折线逐段累计 */
export function measureCurve(
  points: PixelPoint[],
  depthAt: (x: number, y: number) => number,
  cal: Calibration, stepMicron: number,
): { planar: number; length3d: number } {
  let planar = 0, length3d = 0;
  for (let i = 1; i < points.length; i++) {
    const seg = measureLength(points[i - 1], points[i], depthAt, cal, stepMicron);
    planar += seg.planar;
    length3d += seg.length3d;
  }
  return { planar, length3d };
}

/** 把一次交互测量落成 Measurement 记录（坐标归一化 0..1，跨分辨率可复用） */
export function buildMeasurement(
  kind: Measurement['kind'],
  label: string,
  points: PixelPoint[],
  width: number,
  height: number,
  value: number,
  value3d?: number,
): Omit<Measurement, 'id' | 'createdAt' | 'stackId'> {
  return {
    kind,
    label,
    points: points.map((p) => ({ x: p.x / width, y: p.y / height })),
    value,
    value3d,
  };
}

export function denormalizePoints(m: Measurement, width: number, height: number): PixelPoint[] {
  return m.points.map((p) => ({ x: p.x * width, y: p.y * height }));
}
