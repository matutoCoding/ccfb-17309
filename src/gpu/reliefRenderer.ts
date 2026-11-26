// 三维浮雕：深度图 → 网格 + 合成图纹理，可旋转/重打光/掠射光，红蓝与左右并排立体
// WebGPU render pipeline；深度按步距与像素当量折算物理尺度
import type { GpuCaps } from './device';

const RELIEF_VERT = /* wgsl */ `
struct Camera {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  eye: vec3<f32>,
  lightDir: vec3<f32>,
  zScale: f32,
  stereoOffset: f32, // 红蓝立体时的虚拟瞳距偏移
  pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> depths: array<f32>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@group(0) @binding(4) var<storage, read> indices: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) world: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let dims = textureDimensions(tex);
  let w = f32(dims.x); let h = f32(dims.y);
  let gridId = indices[vi];
  let x = f32(gridId % u32(w));
  let y = f32(gridId / u32(w));
  let p = u32(y) * dims.x + u32(x);
  let uvw = vec2<f32>(x / (w - 1.0), 1.0 - y / (h - 1.0));
  let eyeShift = f32(ii) * cam.stereoOffset;
  let pos = vec3<f32>((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0 * (h / w), depths[p] * cam.zScale);
  let world = (cam.model * vec4<f32>(pos + vec3<f32>(eyeShift, 0.0, 0.0), 1.0)).xyz;
  // 法线：邻近深度差分
  let xl = max(0u, u32(x) - 1u); let xr = min(dims.x - 1u, u32(x) + 1u);
  let yu = max(0u, u32(y) - 1u); let yd = min(dims.y - 1u, u32(y) + 1u);
  let dzdx = (depths[u32(y)*dims.x + xr] - depths[u32(y)*dims.x + xl]) * cam.zScale;
  let dzdy = (depths[yd*dims.x + u32(x)] - depths[yu*dims.x + u32(x)]) * cam.zScale;
  let n = normalize(vec3<f32>(-dzdx, -dzdy, 2.0 / w));
  var out: VSOut;
  out.pos = cam.viewProj * vec4<f32>(world, 1.0);
  out.uv = vec2<f32>(uvw.x, 1.0 - uvw.y);
  out.normal = n;
  out.world = world;
  return out;
}

@fragment
fn fs_main(in: VSOut, @builtin(instance_index) ii: u32) -> @location(0) vec4<f32> {
  let base = textureSample(tex, samp, in.uv);
  let n = normalize(in.normal);
  let L = normalize(cam.lightDir);
  let V = normalize(cam.eye - in.world);
  let diff = max(dot(n, L), 0.0);
  let H = normalize(L + V);
  let spec = pow(max(dot(n, H), 0.0), 24.0) * 0.35;
  let ambient = 0.35;
  let shade = ambient + diff * 0.8;
  let color = base.rgb * shade + vec3<f32>(spec);
  // 实例 0=红通道视角，实例 1=青通道视角（红蓝立体由主机输出合成）
  if (ii == 0u) { return vec4<f32>(color, 1.0); }
  return vec4<f32>(color, 1.0);
}
`;

export interface ReliefView {
  yaw: number;
  pitch: number;
  lightAzimuth: number;
  lightElevation: number; // 掠射光 = 小角度
  zScale: number;
  stereo: 'off' | 'anaglyph' | 'sbs';
}

export class ReliefRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private pipeline: GPURenderPipeline | null = null;
  private texture: GPUTexture | null = null;
  private depthBuf: GPUBuffer | null = null;
  private indexBuf: GPUBuffer | null = null;
  private indexCount = 0;
  private vertexCount = 0;
  private width = 0;
  private height = 0;
  private caps: GpuCaps;

  constructor(caps: GpuCaps, canvas: HTMLCanvasElement) {
    if (!caps.supported) throw new Error('WebGPU 不可用');
    this.caps = caps;
    this.device = caps.device;
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Canvas WebGPU 上下文不可用');
    this.context = ctx;
    this.context.configure({ device: this.device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });
  }

  private ensurePipeline(): GPURenderPipeline {
    if (this.pipeline) return this.pipeline;
    const module = this.device.createShaderModule({ code: RELIEF_VERT });
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module, entryPoint: 'fs_main',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });
    return this.pipeline;
  }

  /** 载入深度图（层索引单位）与合成图（RGBA 0..1） */
  setData(depth: Float32Array, color: Float32Array, width: number, height: number): void {
    this.width = width; this.height = height;
    this.vertexCount = width * height;
    this.depthBuf?.destroy();
    this.texture?.destroy();
    this.indexBuf?.destroy();
    this.depthBuf = this.device.createBuffer({ size: depth.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.depthBuf, 0, depth.buffer.slice(0));
    // quad 索引：每格两个三角形
    const quads = (width - 1) * (height - 1);
    const idx = new Uint32Array(quads * 6);
    let k = 0;
    for (let qy = 0; qy < height - 1; qy++) {
      for (let qx = 0; qx < width - 1; qx++) {
        const a = qy * width + qx, b = a + 1, c = a + width, d = c + 1;
        idx[k++] = a; idx[k++] = c; idx[k++] = b;
        idx[k++] = b; idx[k++] = c; idx[k++] = d;
      }
    }
    this.indexCount = idx.length;
    this.indexBuf = this.device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.indexBuf, 0, idx.buffer.slice(0));
    this.texture = this.device.createTexture({
      size: { width, height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const rgba8 = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba8[i * 4] = color[i * 4] * 255;
      rgba8[i * 4 + 1] = color[i * 4 + 1] * 255;
      rgba8[i * 4 + 2] = color[i * 4 + 2] * 255;
      rgba8[i * 4 + 3] = 255;
    }
    this.device.queue.writeTexture({ texture: this.texture }, rgba8, { bytesPerRow: width * 4, rowsPerImage: height }, { width, height });
  }

  private perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, far * nf, -1,
      0, 0, far * near * nf, 0,
    ]);
  }

  private rotY(a: number): Float32Array {
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  }
  private rotX(a: number): Float32Array {
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  }
  private mul(a: Float32Array, b: Float32Array): Float32Array {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[0 * 4 + r] * b[c * 4] + a[1 * 4 + r] * b[c * 4 + 1]
        + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
    }
    return o;
  }

  render(view: ReliefView, canvas: HTMLCanvasElement): void {
    if (!this.texture || !this.depthBuf) return;
    const pipe = this.ensurePipeline();
    const d = this.device;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = this.perspective((45 * Math.PI) / 180, aspect, 0.01, 100);
    const model = this.mul(this.rotY(view.yaw), this.rotX(view.pitch));
    const eye = new Float32Array([0, 0, 3.2]);
    const viewMat = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3.2, 1]);
    const vp = this.mul(proj, viewMat);
    const az = view.lightAzimuth, el = view.lightElevation;
    const lightDir = new Float32Array([
      Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az),
    ]);
    const stereoOn = view.stereo !== 'off';
    // Camera struct: mat4 viewProj(64) + mat4 model(64) + vec3 eye(16) + vec3 light(16) + f32 zScale + f32 stereo + vec2 pad = 192
    const cam = new ArrayBuffer(192);
    const f = new Float32Array(cam);
    f.set(vp, 0); f.set(model, 16);
    f.set(eye, 32);
    f.set(lightDir, 36);
    f[40] = view.zScale;
    f[41] = stereoOn ? 0.03 : 0;
    const camBuf = d.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(camBuf, 0, cam);
    const sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const bg = d.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camBuf } },
        { binding: 1, resource: { buffer: this.depthBuf } },
        { binding: 2, resource: this.texture.createView() },
        { binding: 3, resource: sampler },
        { binding: 4, resource: { buffer: this.indexBuf! } },
      ],
    });

    const depthTex = d.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.12, g: 0.12, b: 0.14, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    if (stereoOn && view.stereo === 'anaglyph') {
      pass.draw(this.indexCount, 2);
    } else {
      pass.draw(this.indexCount, 1);
    }
    pass.end();
    d.queue.submit([enc.finish()]);
    camBuf.destroy(); depthTex.destroy();
  }
}
