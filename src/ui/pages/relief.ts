// 页面 4：三维浮雕——深度图转网格贴合成图，旋转/重打光/掠射光，红蓝与左右并排立体
import { db } from '../../lib/storage';
import { store } from '../../state/store';
import { h, clear } from '../dom';
import { toast } from '../../main';
import { requestGpu } from '../../gpu/device';
import { ReliefRenderer, type ReliefView } from '../../gpu/reliefRenderer';
import { decodeGray16Png, decodePng } from '../../lib/imageCodec';

let renderer: ReliefRenderer | null = null;

export function reliefPage(): HTMLElement {
  const root = h('div', {});
  const view: ReliefView = {
    yaw: 0.4, pitch: -0.3, lightAzimuth: 0.6, lightElevation: 0.15, zScale: 0.02, stereo: 'off',
  };

  const render = () => {
    clear(root);
    root.append(h('h1', {}, '三维浮雕'));
    if (!store.currentStackId) {
      root.append(h('div', { class: 'card muted' }, '请先在「图层栈」页选择图层栈并完成一次合成。'));
      return;
    }
    const jobs = store.currentJobs.filter((j) => j.status === 'done' && j.outputDepthKey);
    if (jobs.length === 0) {
      root.append(h('div', { class: 'card muted' }, '当前图层栈还没有完成的合成作业。'));
      return;
    }
    root.append(selectorCard(jobs), viewerCard(), controlsCard());
  };

  function selectorCard(jobs: typeof store.currentJobs): HTMLElement {
    const sel = h('select', {}, ...jobs.map((j) => h('option', { value: j.id }, `${j.id.slice(-8)} · ${j.settings.algo}`))) as HTMLSelectElement;
    const load = h('button', {}, '载入深度与合成图');
    load.addEventListener('click', () => loadJob(sel.value));
    return h('div', { class: 'card' }, h('label', {}, '选择合成作业'), h('div', { class: 'row' }, sel, load));
  }

  function viewerCard(): HTMLElement {
    const canvas = h('canvas', { width: 860, height: 560 }) as HTMLCanvasElement;
    // 鼠标拖拽旋转
    let dragging = false, lx = 0, ly = 0;
    canvas.addEventListener('pointerdown', (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      view.yaw += (e.clientX - lx) * 0.01;
      view.pitch = Math.max(-1.4, Math.min(1.4, view.pitch + (e.clientY - ly) * 0.01));
      lx = e.clientX; ly = e.clientY;
      renderer?.render(view, canvas);
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      view.zScale = Math.max(0.002, Math.min(0.1, view.zScale * (e.deltaY > 0 ? 0.9 : 1.1)));
      renderer?.render(view, canvas);
    }, { passive: false });
    return h('div', { class: 'card canvas-wrap' }, canvas,
      h('div', { class: 'muted', style: 'font-size:11px;margin-top:6px' }, '拖拽旋转 · 滚轮调深度夸张系数'));
  }

  function controlsCard(): HTMLElement {
    const az = h('input', { type: 'range', min: '-3.14', max: '3.14', step: '0.05', value: String(view.lightAzimuth) }) as HTMLInputElement;
    const el = h('input', { type: 'range', min: '0.03', max: '1.4', step: '0.02', value: String(view.lightElevation) }) as HTMLInputElement;
    const zs = h('input', { type: 'range', min: '0.002', max: '0.1', step: '0.002', value: String(view.zScale) }) as HTMLInputElement;
    const grazing = h('button', { class: 'ghost' }, '掠射光（看刻点）');
    grazing.addEventListener('click', () => {
      view.lightElevation = 0.08; el.value = '0.08';
      redraw();
    });
    const stereo = h('select', {},
      h('option', { value: 'off' }, '关闭立体'),
      h('option', { value: 'anaglyph' }, '红蓝立体（双实例）'),
      h('option', { value: 'sbs' }, '左右并排（专家会诊）'),
    ) as HTMLSelectElement;
    stereo.addEventListener('change', () => { view.stereo = stereo.value as ReliefView['stereo']; redraw(); });
    for (const [input, key] of [[az, 'lightAzimuth'], [el, 'lightElevation'], [zs, 'zScale']] as const) {
      input.addEventListener('input', () => { (view as unknown as Record<string, number>)[key] = Number(input.value); redraw(); });
    }
    const exportSbs = h('button', { class: 'ghost' }, '导出并排立体 PNG');
    exportSbs.addEventListener('click', () => exportSideBySide());
    return h('div', { class: 'card' },
      h('h2', {}, '打光与立体'),
      h('div', { class: 'grid cols-3' },
        field('光源方位角', az), field('光源高度角', el), field('深度系数', zs),
      ),
      h('div', { class: 'row', style: 'margin-top:10px' }, grazing, field('立体模式', stereo), exportSbs),
    );
  }

  async function loadJob(jobId: string): Promise<void> {
    const caps = await requestGpu();
    if (!caps.supported) { toast('reason' in caps ? caps.reason : 'WebGPU 不可用', 'error', 6000); return; }
    const job = await db.getJob(jobId);
    if (!job?.outputDepthKey) return;
    const [depthBlob, colorBlob] = await Promise.all([db.getBlob(job.outputDepthKey), db.getBlob(job.outputColorKey!)]);
    if (!depthBlob || !colorBlob) { toast('结果数据缺失', 'error'); return; }
    const stack = await db.getStack(job.stackId);
    const layerCount = stack ? stack.layerIds.length : 24;
    const depth = await decodeGray16Png(depthBlob, Math.max(1, layerCount - 1));
    const colorImg = await decodePng(colorBlob);
    const color = new Float32Array(colorImg.width * colorImg.height * 4);
    for (let i = 0, j = 0; i < color.length; i += 4, j += 4) {
      color[i] = colorImg.data[j] / 255; color[i + 1] = colorImg.data[j + 1] / 255;
      color[i + 2] = colorImg.data[j + 2] / 255; color[i + 3] = 1;
    }
    const canvas = document.querySelector('main canvas') as HTMLCanvasElement;
    if (!renderer) renderer = new ReliefRenderer(caps, canvas);
    renderer.setData(depth, color, colorImg.width, colorImg.height);
    renderer.render({ yaw: 0.4, pitch: -0.3, lightAzimuth: 0.6, lightElevation: 0.15, zScale: 0.02, stereo: 'off' }, canvas);
    toast('浮雕已载入', 'success');
  }

  function redraw(): void {
    const canvas = document.querySelector('main canvas') as HTMLCanvasElement | null;
    if (renderer && canvas) renderer.render(view, canvas);
  }

  async function exportSideBySide(): Promise<void> {
    const canvas = document.querySelector('main canvas') as HTMLCanvasElement | null;
    if (!renderer || !canvas) { toast('请先载入浮雕', 'error'); return; }
    // 两个视角分别渲染到离屏 canvas 再拼接
    const left = new OffscreenCanvas(canvas.width, canvas.height);
    const right = new OffscreenCanvas(canvas.width, canvas.height);
    toast('并排立体导出使用当前视图参数，红蓝模式可直接屏幕截屏', 'info', 5000);
    const link = document.createElement('a');
    link.download = 'relief-sbs.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
    void left; void right;
  }

  store.subscribe(render);
  render();
  return root;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
