// 页面 3：合成——算法选择/参数、作业启停/断点、像素贡献层查询、合成测试与验收
import type { FusionJob, FusionSettings, SyntheticSpec } from '../../types';
import { db } from '../../lib/storage';
import { store } from '../../state/store';
import { h, clear } from '../dom';
import { navigate, toast } from '../../main';
import { DEFAULT_FUSION_SETTINGS } from '../../lib/defaults';
import { BlobTileProvider, MemoryTileProvider } from '../../lib/tileProvider';
import { FusionOrchestrator } from '../../lib/orchestrator';
import { GpuFusionEngine } from '../../gpu/fusionEngine';
import { generateSyntheticStack, type SyntheticStack } from '../../synthetic/generator';
import { fuseDff } from '../../lib/cpuFusion';
import { evaluate, defaultTargets } from '../../lib/metrics';
import { alignStack, warpLayer } from '../../lib/alignment';
import type { AlignParams } from '../../types';
import { requestGpu } from '../../gpu/device';

let orchestrator: FusionOrchestrator | null = null;
let lastSynthetic: SyntheticStack | null = null;

export function fusionPage(): HTMLElement {
  const root = h('div', {});
  const render = () => {
    clear(root);
    if (!store.currentStackId) {
      root.append(h('h1', {}, '合成'), h('div', { class: 'card muted' }, '请先在「图层栈」页选择一个图层栈。'));
      return;
    }
    const stack = store.currentStack!;
    root.append(
      h('h1', {}, '景深合成'),
      h('div', { class: 'card muted' },
        `${stack.specimenId} · ${stack.viewpoint} · ${store.currentLayers.filter((l) => l.flag !== 'excluded').length} 层参与计算 · ${stack.width}×${stack.height}`),
      settingsCard(stack.id),
      jobsCard(stack.id),
      syntheticCard(),
    );
  };
  store.subscribe(render);
  render();
  return root;
}

function settingsCard(stackId: string): HTMLElement {
  const s = { ...DEFAULT_FUSION_SETTINGS, maxGpuBytes: store.settings.maxGpuBytes };
  const algo = h('select', {},
    h('option', { value: 'dff' }, '焦点度量取深度（DFF，带置信度/补洞）'),
    h('option', { value: 'laplacian-pyramid' }, '拉普拉斯金字塔融合'),
  ) as HTMLSelectElement;
  const tile = h('input', { type: 'number', value: String(s.tileSize), min: '128', max: '2048', step: '128' }) as HTMLInputElement;
  const overlap = h('input', { type: 'number', value: String(s.tileOverlap), min: '16', max: '256', step: '16' }) as HTMLInputElement;
  const topK = h('input', { type: 'number', value: String(s.topK), min: '1', max: '8' }) as HTMLInputElement;
  const halo = h('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(s.haloGuard) }) as HTMLInputElement;
  const guide = h('input', { type: 'number', value: String(s.guideRadius), min: '1', max: '16' }) as HTMLInputElement;
  const lowTex = h('input', { type: 'number', value: String(s.confidenceLowTexture), min: '0.02', max: '0.5', step: '0.02' }) as HTMLInputElement;
  const budget = h('input', { type: 'number', value: String(Math.round(s.maxGpuBytes / 1024 ** 3)), min: '1', max: '12' }) as HTMLInputElement;

  const start = h('button', {}, '开始合成');
  start.addEventListener('click', () => startFusion(stackId, collect()));
  const pause = h('button', { class: 'ghost' }, '暂停');
  pause.addEventListener('click', () => {
    const job = store.currentJobs[0];
    if (job) orchestrator?.pause(job.id);
  });
  const resume = h('button', { class: 'ghost' }, '续算');
  resume.addEventListener('click', () => {
    const job = store.currentJobs.find((j) => j.status === 'paused');
    if (job) { orchestrator?.resume(job.id); toast('已恢复，断点续算中', 'info'); }
  });

  function collect(): FusionSettings {
    return {
      ...s,
      algo: algo.value as FusionSettings['algo'],
      tileSize: Number(tile.value), tileOverlap: Number(overlap),
      topK: Number(topK.value), haloGuard: Number(halo.value),
      guideRadius: Number(guide), confidenceLowTexture: Number(lowTex.value),
      maxGpuBytes: Number(budget.value) * 1024 ** 3,
    };
  }

  return h('div', { class: 'card' },
    h('h2', {}, '合成参数'),
    h('div', { class: 'grid cols-3' },
      field('算法', algo), field('分块边长 px', tile), field('重叠 px', overlap),
      field('软选择层数 topK（透明膜翅多层信号）', topK), field('光晕抑制', halo),
      field('引导滤波半径', guide), field('低纹理阈值系数', lowTex),
      field('显存预算 GB（8G 卡建议 6）', budget),
    ),
    h('div', { class: 'row', style: 'margin-top:12px' }, start, pause, resume),
  );
}

function jobsCard(stackId: string): HTMLElement {
  const wrap = h('div', { class: 'card' }, h('h2', {}, '作业与结果'));
  const jobs = store.jobs[stackId] ?? [];
  if (jobs.length === 0) {
    wrap.append(h('div', { class: 'muted' }, '尚无作业。'));
    return wrap;
  }
  for (const job of jobs) {
    const bar = h('div', { class: 'progress' }, h('div', { style: `width:${(job.progress * 100).toFixed(1)}%` }));
    const statusTag = h('span', {
      class: `tag ${job.status === 'done' ? 'ok' : job.status === 'error' ? 'bad' : 'info'}`,
    }, job.status);
    const block = h('div', { style: 'margin-bottom:14px' },
      h('div', { class: 'row' },
        statusTag,
        h('span', { class: 'mono' }, `${job.tilesDone}/${job.tilesTotal} 块 · ${(job.progress * 100).toFixed(1)}%`),
        h('span', { class: 'muted' }, new Date(job.updatedAt).toLocaleString()),
        h('div', { class: 'spacer' }),
        job.outputColorKey ? h('button', {
          class: 'small ghost', onclick: () => openResult(job),
        }, '查看结果/贡献层') : null,
      ),
      h('div', { style: 'margin:8px 0' }, bar),
      job.checkpoint ? h('div', { class: 'muted mono', style: 'font-size:11px' },
        `断点：tile(${job.checkpoint.tileX},${job.checkpoint.tileY}) 层 ${job.checkpoint.layerIndex}，关页面后可从此续算`) : null,
      job.metrics ? metricsTable(job) : null,
      job.error ? h('div', { class: 'tag bad' }, job.error) : null,
    );
    wrap.append(block);
  }
  return wrap;
}

function metricsTable(job: FusionJob): HTMLElement {
  const m = job.metrics!;
  return h('div', { class: 'grid cols-4', style: 'margin-top:8px' },
    metric('全景深 PSNR', `${m.fusedPsnrDb.toFixed(2)} dB`),
    metric('最清晰单层 PSNR', `${m.bestSinglePsnrDb.toFixed(2)} dB`),
    metric('增益', `+${m.gainDb.toFixed(2)} dB`, m.gainDb >= store.settings.psnrTargetDb),
    metric('深度中位误差', `${m.depthMedianAbsErrStep.toFixed(2)} 步距`, m.depthMedianAbsErrStep <= store.settings.depthErrorTargetStep),
    metric('光晕宽度中位', `${m.haloWidthPxMedian.toFixed(2)} px`, m.haloWidthPxMedian <= store.settings.haloTargetPx),
    metric('低纹理掩膜', `${(m.lowTextureMaskRatio * 100).toFixed(1)}%`),
    metric('高光掩膜', `${(m.specularMaskRatio * 100).toFixed(1)}%`),
    metric('补洞比例', `${(m.holeFilledRatio * 100).toFixed(1)}%`),
    h('div', { class: `metric-big ${m.pass ? 'pass' : 'fail'}` }, m.pass ? '验收通过' : '未达标'),
  );
}

function metric(label: string, value: string, ok?: boolean): HTMLElement {
  return h('div', { class: 'card', style: 'padding:10px' },
    h('div', { class: 'muted', style: 'font-size:11px' }, label),
    h('div', { class: `metric-big ${ok === undefined ? '' : ok ? 'pass' : 'fail'}` }, value),
  );
}

async function startFusion(stackId: string, settings: FusionSettings): Promise<void> {
  const caps = await requestGpu();
  if (!caps.supported) {
    toast('reason' in caps ? caps.reason : 'WebGPU 不可用', 'error', 6000);
    return;
  }
  const stack = await db.getStack(stackId);
  if (!stack) return;
  const layers = (await db.listLayers(stackId)).filter((l) => l.flag !== 'excluded');
  if (layers.length < 2) { toast('有效图层不足 2 层', 'error'); return; }

  // 对齐：降采样估计，参数还原到全分辨率尺度
  toast('正在估计对齐参数…', 'info');
  const dsScale = Math.min(1, 1024 / Math.max(stack.width, stack.height));
  const dsImgs: Float32Array[] = [];
  for (const l of layers) {
    const blob = await db.getBlob(l.blobKey);
    const bmp = await createImageBitmap(blob!);
    const cw = Math.max(1, Math.round(bmp.width * dsScale));
    const ch = Math.max(1, Math.round(bmp.height * dsScale));
    const cv = new OffscreenCanvas(cw, ch);
    const ctx = cv.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0, cw, ch);
    const d = ctx.getImageData(0, 0, cw, ch).data;
    const rgba = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) rgba[i] = d[i];
    dsImgs.push(rgba);
    bmp.close();
  }
  const { params } = alignStack(dsImgs, Math.max(1, Math.round(stack.width * dsScale)), Math.max(1, Math.round(stack.height * dsScale)));
  const aligns: AlignParams[] = params.map((p) => ({ ...p, tx: p.tx / dsScale, ty: p.ty / dsScale }));

  orchestrator = new FusionOrchestrator(caps);
  const job = await FusionOrchestrator.createJob(stack, settings);
  await store.refreshJobs(stackId);
  const provider = new BlobTileProvider(layers, aligns, stack.width, stack.height);

  orchestrator.run(job, provider, (p) => {
    store.refreshJobs(stackId);
  }).then(async (done) => {
    await store.refreshJobs(stackId);
    toast(`合成完成：${done.outputColorKey}`, 'success');
  }).catch((e) => {
    store.refreshJobs(stackId);
    if (!String(e.message).includes('中止')) toast(`合成失败：${e.message}`, 'error', 6000);
  });
}

async function openResult(job: FusionJob): Promise<void> {
  const colorBlob = await db.getBlob(job.outputColorKey!);
  const depthBlob = await db.getBlob(job.outputDepthKey!);
  const confBlob = await db.getBlob(job.outputConfidenceKey!);
  const contribBlob = await db.getBlob(job.outputContribKey!);
  if (!colorBlob) return;
  const url = URL.createObjectURL(colorBlob);
  const img = h('img', { src: url, style: 'max-width:100%;border-radius:8px' }) as HTMLImageElement;
  await img.decode();
  const cv = h('canvas', { width: img.naturalWidth, height: img.naturalHeight, style: 'max-width:100%;cursor:crosshair' }) as HTMLCanvasElement;
  const ctx = cv.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const info = h('div', { class: 'mono card' }, '点击合成图像素查询贡献层…');
  const topK = job.settings.topK;
  let contrib: Uint16Array | null = null;
  if (contribBlob) contrib = new Uint16Array(await contribBlob.arrayBuffer());
  cv.addEventListener('click', (e) => {
    const rect = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * cv.width / rect.width);
    const y = Math.floor((e.clientY - rect.top) * cv.height / rect.height);
    const p = y * cv.width + x;
    const layers = contrib ? Array.from(contrib.slice(p * topK, p * topK + topK)) : [];
    info.replaceChildren(h('strong', {}, `像素 (${x}, ${y})`),
      h('div', {}, `贡献层（按权重降序）：${layers.map((z) => `#${z}`).join(', ')}`),
      h('div', { class: 'muted' }, '透明膜翅区域通常多层权重接近，不会硬选单层'));
  });
  const dlg = modal(h('div', {},
    h('h2', {}, '合成结果与贡献层查询'),
    h('div', { class: 'canvas-wrap' }, cv),
    info,
    depthBlob ? h('div', { class: 'muted', style: 'margin-top:8px' }, `深度图/置信度已存：${job.outputDepthKey} / ${job.outputConfidenceKey}`) : null,
    h('div', { class: 'row', style: 'margin-top:10px' },
      h('button', { class: 'ghost', onclick: () => dlg.remove() }, '关闭'),
      h('button', {
        onclick: () => {
          const a = document.createElement('a');
          a.href = url; a.download = `${job.id}_fused.png`; a.click();
        },
      }, '下载 PNG'),
    ),
  ));
  document.body.append(dlg);
}

// ---------------- 合成测试数据 + 验收 ----------------
function syntheticCard(): HTMLElement {
  const w = h('input', { type: 'number', value: '320', min: '64', max: '1024' }) as HTMLInputElement;
  const hh = h('input', { type: 'number', value: '240', min: '64', max: '1024' }) as HTMLInputElement;
  const layers = h('input', { type: 'number', value: '24', min: '6', max: '200' }) as HTMLInputElement;
  const breath = h('input', { type: 'number', value: '1.2', min: '0', max: '5', step: '0.1' }) as HTMLInputElement;
  const jitter = h('input', { type: 'number', value: '1.5', min: '0', max: '8', step: '0.1' }) as HTMLInputElement;
  const seed = h('input', { type: 'number', value: '42' }) as HTMLInputElement;
  const runBtn = h('button', {}, '生成栈并跑验收（CPU 参考算法）');
  const gpuBtn = h('button', { class: 'ghost' }, '同栈走 WebGPU 分块管线');
  const out = h('div', { class: 'card' },
    h('h2', {}, '合成测试（已知高度场真值）'),
    h('div', { class: 'grid cols-3' },
      field('宽 px', w), field('高 px', hh), field('层数', layers),
      field('呼吸缩放 %', breath), field('导轨抖动 px', jitter), field('随机种子', seed),
    ),
    h('div', { class: 'row', style: 'margin-top:10px' }, runBtn, gpuBtn),
    h('div', { class: 'muted', style: 'margin-top:6px' },
      '验收线：PSNR 增益 ≥ 6dB、深度中位误差 ≤ 1 步距、刚毛光晕 ≤ 2px'),
  );
  const resultBox = h('div', {});
  out.append(resultBox);

  function spec(): SyntheticSpec {
    return {
      width: Number(w.value), height: Number(hh.value), layers: Number(layers.value),
      stepMicron: 2, breathScalePct: Number(breath.value), jitterPx: Number(jitter.value),
      seed: Number(seed.value), includeWing: true, includeSetae: true, includePunctures: true,
    };
  }

  runBtn.addEventListener('click', () => {
    const syn = generateSyntheticStack(spec());
    lastSynthetic = syn;
    // 对齐后再融合（与真实流程一致）
    const { params } = alignStack(syn.images, syn.width, syn.height);
    const warped = syn.images.map((im, z) => warpLayer(im, syn.width, syn.height, params[z]));
    const fused = fuseDff(warped, syn.width, syn.height, {
      windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235,
      softTemp: 8, haloGuard: 0.8,
    });
    const metrics = evaluate(syn, fused, defaultTargets(syn.spec));
    resultBox.replaceChildren(renderSyntheticResult(syn, fused.color, metrics));
  });

  gpuBtn.addEventListener('click', async () => {
    const caps = await requestGpu();
    if (!caps.supported) { toast('reason' in caps ? caps.reason : 'WebGPU 不可用', 'error', 6000); return; }
    const syn = lastSynthetic ?? generateSyntheticStack(spec());
    lastSynthetic = syn;
    const engine = new GpuFusionEngine(caps);
    const settings: FusionSettings = {
      ...DEFAULT_FUSION_SETTINGS, tileSize: 192, tileOverlap: 32, maxGpuBytes: store.settings.maxGpuBytes,
    };
    const provider = new MemoryTileProvider(syn.images, syn.images.map(() => ({ scale: 1, angle: 0, tx: 0, ty: 0 })), syn.width, syn.height);
    const outputs = await engine.run({
      width: syn.width, height: syn.height, settings, provider,
      signal: { paused: false, aborted: false }, checkpoint: null,
      onProgress: () => undefined,
      shouldPause: async () => undefined,
    });
    const fused = fuseDff(syn.images, syn.width, syn.height, {
      windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235, softTemp: 8, haloGuard: 0.8,
    });
    // GPU 输出是 RGBA f32 0..1；评测需要 RGB 0..255
    const gpuColor = new Float32Array(syn.width * syn.height * 3);
    for (let p = 0; p < syn.width * syn.height; p++) {
      gpuColor[p * 3] = outputs.color[p * 4] * 255;
      gpuColor[p * 3 + 1] = outputs.color[p * 4 + 1] * 255;
      gpuColor[p * 3 + 2] = outputs.color[p * 4 + 2] * 255;
    }
    fused.color.set(gpuColor);
    fused.depth.set(outputs.depth);
    fused.confidence.set(outputs.confidence);
    fused.lowTexture.set(new Uint8Array(outputs.mask.length).map((_, i) => outputs.mask[i] & 1 ? 1 : 0));
    fused.specular.set(new Uint8Array(outputs.mask.length).map((_, i) => outputs.mask[i] & 2 ? 1 : 0));
    fused.holeMask.set(new Uint8Array(outputs.mask.length).map((_, i) => outputs.mask[i] & 4 ? 1 : 0));
    const metrics = evaluate(syn, fused, defaultTargets(syn.spec));
    resultBox.replaceChildren(renderSyntheticResult(syn, fused.color, metrics));
  });

  return out;
}

function renderSyntheticResult(syn: SyntheticStack, color: Float32Array, metrics: FusionJob['metrics']): HTMLElement {
  const cv = h('canvas', { width: syn.width, height: syn.height }) as HTMLCanvasElement;
  const ctx = cv.getContext('2d')!;
  const img = ctx.createImageData(syn.width, syn.height);
  for (let i = 0; i < syn.width * syn.height; i++) {
    img.data[i * 4] = color[i * 3]; img.data[i * 4 + 1] = color[i * 3 + 1];
    img.data[i * 4 + 2] = color[i * 3 + 2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const m = metrics!;
  return h('div', { style: 'margin-top:12px' },
    h('div', { class: 'row' },
      h('div', { class: 'canvas-wrap' }, h('div', { class: 'muted', style: 'font-size:11px' }, '融合结果'), cv),
      h('div', { style: 'flex:1' },
        h('div', { class: `metric-big ${m.pass ? 'pass' : 'fail'}` }, m.pass ? '验收通过' : '未达标'),
        h('dl', { class: 'kv' },
          h('dt', {}, '全景深 PSNR'), h('dd', { class: 'mono' }, `${m.fusedPsnrDb.toFixed(2)} dB`),
          h('dt', {}, '最清晰单层'), h('dd', { class: 'mono' }, `${m.bestSinglePsnrDb.toFixed(2)} dB`),
          h('dt', {}, '增益（目标+6）'), h('dd', { class: 'mono' }, `+${m.gainDb.toFixed(2)} dB`),
          h('dt', {}, '深度误差（目标≤1步距）'), h('dd', { class: 'mono' }, `${m.depthMedianAbsErrStep.toFixed(3)}`),
          h('dt', {}, '光晕（目标≤2px）'), h('dd', { class: 'mono' }, `${m.haloWidthPxMedian.toFixed(2)} px`),
          h('dt', {}, '判定明细'), h('dd', {}, m.reasons.length ? m.reasons.join('；') : '全部达标'),
        ),
      ),
    ),
  );
}

function modal(content: HTMLElement): HTMLElement {
  const dlg = h('div', { class: 'dialog-backdrop', onclick: (e: Event) => { if (e.target === dlg) dlg.remove(); } },
    h('div', { class: 'dialog' }, content));
  return dlg;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
