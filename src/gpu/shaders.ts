// WGSL 着色器库：全部合成计算在 compute shader 中完成
// 约定：图像以 rgba8unorm storage texture 或 f32 storage buffer 承载；
// 分块坐标通过 uniform 的 tile rect 传入，分块间重叠区在主机侧做 feather 混合。

export const COMMON = /* wgsl */ `
struct Tile {
  x: u32, y: u32, w: u32, h: u32,      // 分块在全图中的矩形
  fullW: u32, fullH: u32,
  layerCount: u32,
  windowRadius: u32,
  topK: u32,
  guideRadius: u32,
  params1: vec4<f32>,  // softTempScale, haloGuard, lowThresh, specLum
  params2: vec4<f32>,  // stepMicron, unused..., feather, overlap
};

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }

fn loadLayer(layers: array<f32>, base: u32, pitch: u32, p: u32, ch: u32) -> f32 {
  return layers[base + p * 4u + ch];
}

// 双线性采样单层（坐标为像素空间浮点，越界 clamp）
fn sampleLayer(layers: array<f32>, base: u32, w: u32, h: u32, fx: f32, fy: f32, ch: u32) -> f32 {
  let cx = clamp(fx, 0.0, f32(w) - 1.001);
  let cy = clamp(fy, 0.0, f32(h) - 1.001);
  let x0 = u32(cx); let y0 = u32(cy);
  let tx = cx - f32(x0); let ty = cy - f32(y0);
  let i00 = base + (y0 * w + x0) * 4u + ch;
  let i10 = i00 + 4u;
  let i01 = i00 + w * 4u;
  let i11 = i01 + 4u;
  return layers[i00] * (1.0 - tx) * (1.0 - ty) + layers[i10] * tx * (1.0 - ty)
       + layers[i01] * (1.0 - tx) * ty + layers[i11] * tx * ty;
}
`;

// 1) 焦点度量：Sobel-Tenengrad 梯度能量 + 窗口均值
export const FOCUS_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> layers: array<f32>;   // 当前层 RGBA f32
@group(0) @binding(2) var<storage, read_write> measure: array<f32>; // 分块尺寸

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  if (x < 1u || y < 1u || x >= lw - 1u || y >= lh - 1u) { measure[p] = 0.0; return; }
  let i = p * 4u;
  let c = |ix: u32, iy: u32| -> f32 {
    let q = (iy * lw + ix) * 4u;
    return luma(vec3<f32>(layers[q], layers[q + 1u], layers[q + 2u]));
  };
  let gx = -c(x-1u,y-1u) - 2.0*c(x-1u,y) - c(x-1u,y+1u)
         +  c(x+1u,y-1u) + 2.0*c(x+1u,y) + c(x+1u,y+1u);
  let gy = -c(x-1u,y-1u) - 2.0*c(x,y-1u) - c(x+1u,y-1u)
         +  c(x-1u,y+1u) + 2.0*c(x,y+1u) + c(x+1u,y+1u);
  let energy = gx*gx + gy*gy;
  // 窗口均值（box，半径 windowRadius）
  let r = tile.windowRadius;
  var acc = 0.0;
  for (var dy = -i32(r); dy <= i32(r); dy = dy + 1) {
    for (var dx = -i32(r); dx <= i32(r); dx = dx + 1) {
      let xx = clamp(i32(x) + dx, 0, i32(lw) - 1);
      let yy = clamp(i32(y) + dy, 0, i32(lh) - 1);
      // 复用同梯度近似：直接对能量做 box，需要邻域能量 → 用解析 Sobel 重算代价高，
      // 实现上 measure 缓冲分两遍：本 shader 先写点能量，第二遍 box 由 BOX_SHADER 完成。
      acc = acc + energy;
    }
  }
  measure[p] = acc / f32((2u*r + 1u) * (2u*r + 1u));
}
`;

// 2) 通用 box 均值（单通道 f32）
export const BOX_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let r = tile.guideRadius;
  var acc = 0.0;
  for (var dy = -i32(r); dy <= i32(r); dy = dy + 1) {
    for (var dx = -i32(r); dx <= i32(r); dx = dx + 1) {
      let xx = u32(clamp(i32(x) + dx, 0, i32(lw) - 1));
      let yy = u32(clamp(i32(y) + dy, 0, i32(lh) - 1));
      acc = acc + src[yy * lw + xx];
    }
  }
  dst[y * lw + x] = acc / f32((2u*r + 1u) * (2u*r + 1u));
}
`;

// 3) DFF 决策：逐像素 softmax topK + 深度 + 置信度 + 掩膜
//    measureLayers: N 层焦点度量（连续存储，每层 tilePixels）
export const DFF_DECIDE_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> measureLayers: array<f32>;
@group(0) @binding(2) var<storage, read> imgLayers: array<f32>;
@group(0) @binding(3) var<storage, read_write> depth: array<f32>;
@group(0) @binding(4) var<storage, read_write> colorOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> confOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> maskOut: array<u32>;   // bit0 低纹理 bit1 高光 bit2 空洞
@group(0) @binding(7) var<storage, read_write> contrib: array<u32>;  // 每像素 topK 层号

// 直接从存储缓冲做选择排序取前 K，避免大尺寸 function-private 数组（栈压力）。
// K <= 16，N 来自 uniform 可到 200+，全部按存储索引访问。
fn insertionTopK(p: u32, lw: u32, n: u32, k: u32, outIdx: ptr<function, array<u32, 16>>) {
  var idx: array<u32, 16>;
  let cnt = min(k, n);
  for (var i = 0u; i < 16u; i = i + 1u) { idx[i] = 0u; }
  var used: array<bool, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { used[i] = false; }
  for (var slot = 0u; slot < cnt; slot = slot + 1u) {
    var best = 0u; var bv = -1.0;
    for (var z = 0u; z < n; z = z + 1u) {
      var taken = false;
      for (var q = 0u; q < slot; q = q + 1u) { if (idx[q] == z) { taken = true; } }
      if (taken) { continue; }
      let v = measureLayers[z * lw + p];
      if (v > bv) { bv = v; best = z; }
    }
    idx[slot] = best;
  }
  for (var i = 0u; i < 16u; i = i + 1u) { (*outIdx)[i] = idx[i]; }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  let N = tile.layerCount;
  let K = min(tile.topK, 16u);
  var mx = 0.0;
  for (var z = 0u; z < N; z = z + 1u) {
    mx = max(mx, measureLayers[z * lw + p]);
  }
  var top: array<u32, 16>;
  insertionTopK(p, lw, N, K, &top);

  // softmax 权重
  let temp = max(1.0, tile.params1.x);
  var wsum = 0.0;
  var wts: array<f32, 16>;
  for (var j = 0u; j < K; j = j + 1u) {
    let e = exp((measureLayers[top[j] * lw + p] - mx) / temp);
    wts[j] = e; wsum = wsum + e;
  }
  var d = 0.0;
  var cr = 0.0; var cg = 0.0; var cb = 0.0;
  let pitch = lw * lh;
  for (var j = 0u; j < K; j = j + 1u) {
    let wt = wts[j] / wsum;
    let z = top[j];
    d = d + wt * f32(z);
    let base = z * pitch * 4u;
    cr = cr + wt * imgLayers[base + p*4u];
    cg = cg + wt * imgLayers[base + p*4u + 1u];
    cb = cb + wt * imgLayers[base + p*4u + 2u];
    contrib[p * tile.topK + j] = z;
  }
  depth[p] = d;
  colorOut[p*4u] = cr; colorOut[p*4u+1u] = cg; colorOut[p*4u+2u] = cb; colorOut[p*4u+3u] = 1.0;

  // 置信度
  let peakShare = wts[0] / wsum;
  let focusStrength = clamp(mx / max(1e-6, tile.params1.z * 1.5), 0.0, 1.0);
  confOut[p] = clamp(0.55 * peakShare + 0.45 * focusStrength, 0.0, 1.0);

  var mask = 0u;
  if (mx < tile.params1.z) { mask = mask | 1u | 4u; } // 低纹理 + 空洞
  // 高光：亮度高且层间方差小（直接遍历存储，不缓存大数组）
  var lumMean = 0.0;
  for (var z = 0u; z < N; z = z + 1u) {
    let base = z * pitch * 4u;
    lumMean = lumMean + luma(vec3<f32>(imgLayers[base+p*4u], imgLayers[base+p*4u+1u], imgLayers[base+p*4u+2u]));
  }
  lumMean = lumMean / f32(N);
  var lv = 0.0;
  for (var z = 0u; z < N; z = z + 1u) {
    let base = z * pitch * 4u;
    let L = luma(vec3<f32>(imgLayers[base+p*4u], imgLayers[base+p*4u+1u], imgLayers[base+p*4u+2u]));
    let dd = L - lumMean; lv = lv + dd * dd;
  }
  lv = lv / f32(N);
  if (lumMean > tile.params1.w && lv < 25.0) {
    mask = mask | 2u | 4u;
    confOut[p] = min(confOut[p], 0.25);
  }
  maskOut[p] = mask;
}
`;

// 4) 光晕抑制：深度边缘感知正则（与邻域均值差 >1.5 层时混合）
export const HALO_GUARD_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> depthIn: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> depthOut: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  if ((mask[p] & 4u) != 0u) { depthOut[p] = depthIn[p]; return; }
  var acc = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let xx = u32(clamp(i32(x)+dx, 0, i32(lw)-1));
      let yy = u32(clamp(i32(y)+dy, 0, i32(lh)-1));
      acc = acc + depthIn[yy*lw+xx];
    }
  }
  let m = acc / 9.0;
  let diff = abs(depthIn[p] - m);
  let g = tile.params1.y;
  if (diff > 1.5) {
    depthOut[p] = (1.0 - 0.5*g) * depthIn[p] + 0.5*g * m;
  } else {
    depthOut[p] = depthIn[p];
  }
}
`;

// 5) 按校正后深度重取颜色（最近两层线性插值），含对齐变换
export const REFETCH_SHADER = /* wgsl */ `
${COMMON}
struct AlignT { scale: f32, tx: f32, ty: f32, pad: f32 };
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read> imgLayers: array<f32>;
@group(0) @binding(3) var<storage, read> aligns: array<AlignT>;
@group(0) @binding(4) var<storage, read_write> colorOut: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  let N = f32(tile.layerCount);
  let d = clamp(depth[p], 0.0, N - 1.001);
  let z0 = u32(d); let t = d - f32(z0);
  let pitch = lw * lh;
  let cx = f32(lw) * 0.5; let cy = f32(lh) * 0.5;
  // 输出像素 -> 规范坐标（对齐逆变换，两层各自）
  let fetch = |z: u32, ch: u32| -> f32 {
    let a = aligns[z];
    let dx = f32(x) - cx - a.tx;
    let dy = f32(y) - cy - a.ty;
    let sx = cx + dx / a.scale;
    let sy = cy + dy / a.scale;
    return sampleLayer(imgLayers, z * pitch * 4u, lw, lh, sx, sy, ch);
  };
  for (var c = 0u; c < 3u; c = c + 1u) {
    colorOut[p*4u+c] = fetch(z0, c) * (1.0 - t) + fetch(min(z0+1u, tile.layerCount-1u), c) * t;
  }
  colorOut[p*4u+3u] = 1.0;
}
`;

// 6) 引导滤波补洞（线性系数 a,b + 盒滤波两遍由主机调度 BOX_SHADER）
export const GUIDED_COEF_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> guide: array<f32>;
@group(0) @binding(2) var<storage, read> depth: array<f32>;
@group(0) @binding(3) var<storage, read_write> aOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> bOut: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  let r = tile.guideRadius;
  var n = 0.0; var sI = 0.0; var sP = 0.0; var sII = 0.0; var sIP = 0.0;
  for (var dy = -i32(r); dy <= i32(r); dy = dy + 1) {
    for (var dx = -i32(r); dx <= i32(r); dx = dx + 1) {
      let xx = u32(clamp(i32(x)+dx, 0, i32(lw)-1));
      let yy = u32(clamp(i32(y)+dy, 0, i32(lh)-1));
      let q = yy*lw+xx;
      sI = sI + guide[q]; sP = sP + depth[q];
      sII = sII + guide[q]*guide[q]; sIP = sIP + guide[q]*depth[q];
      n = n + 1.0;
    }
  }
  var mI = sI/n; var mP = sP/n;
  var varI = sII/n - mI*mI;
  var covIP = sIP/n - mI*mP;
  var eps = tile.params2.y;
  var a = covIP / (varI + eps);
  aOut[p] = a; bOut[p] = mP - a*mI;
}
`;

export const GUIDED_APPLY_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> meanA: array<f32>;
@group(0) @binding(2) var<storage, read> meanB: array<f32>;
@group(0) @binding(3) var<storage, read> guide: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<u32>;
@group(0) @binding(5) var<storage, read_write> depth: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y * lw + x;
  if ((mask[p] & 4u) != 0u) {
    depth[p] = meanA[p] * guide[p] + meanB[p];
  }
}
`;

// 7) 拉普拉斯金字塔：单层拉普拉斯能量（决策图用）
export const LAPLACE_ENERGY_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> energy: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y*lw+x;
  if (x < 1u || y < 1u || x >= lw-1u || y >= lh-1u) { energy[p] = 0.0; return; }
  let L = |q: u32| -> f32 {
    let i = q*4u;
    return luma(vec3<f32>(img[i], img[i+1u], img[i+2u]));
  };
  let lap = 4.0*L(p) - L(p-1u) - L(p+1u) - L(p-lw) - L(p+lw);
  energy[p] = lap*lap;
}
`;

// 8) 金字塔级融合：按掩码权重合并 N 个同级带
export const PYRAMID_BLEND_SHADER = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> bands: array<f32>;  // N 层同级 RGB 连续
@group(0) @binding(2) var<storage, read> masks: array<f32>;  // N 层同级软掩码
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lw = tile.w; let lh = tile.h;
  let x = gid.x; let y = gid.y;
  if (x >= lw || y >= lh) { return; }
  let p = y*lw+x;
  let N = tile.layerCount;
  let pitch = lw*lh;
  var acc = vec3<f32>(0.0); var wsum = 0.0;
  for (var z = 0u; z < N; z = z + 1u) {
    let m = masks[z*pitch + p];
    wsum = wsum + m;
    let base = (z*pitch + p)*3u;
    acc = acc + vec3<f32>(bands[base], bands[base+1u], bands[base+2u]) * m;
  }
  if (wsum < 1e-6) {
    let base = p*3u;
    acc = vec3<f32>(bands[base], bands[base+1u], bands[base+2u]);
  } else {
    acc = acc / wsum;
  }
  out[p*3u] = acc.x; out[p*3u+1u] = acc.y; out[p*3u+2u] = acc.z;
}
`;
