// 页面 6：设置——显存预算/存储档位/验收目标/压力测试估算/存储占用
import { store } from '../../state/store';
import { h, clear, fmtBytes } from '../dom';
import { toast } from '../../main';
import { db } from '../../lib/storage';
import { layersInBudget } from '../../gpu/device';
import { generateSyntheticStack } from '../../synthetic/generator';
import { fuseDff } from '../../lib/cpuFusion';
import { evaluate, defaultTargets } from '../../lib/metrics';

export function settingsPage(): HTMLElement {
  const root = h('div', {});
  const render = () => {
    clear(root);
    const s = store.settings;
    root.append(
      h('h1', {}, '设置与验收基线'),
      gpuCard(s),
      targetsCard(s),
      stressCard(s),
      storageCard(),
    );
  };
  store.subscribe(render);
  render();
  return root;
}

function gpuCard(s: typeof store.settings): HTMLElement {
  const budget = h('input', { type: 'range', min: '1', max: '12', step: '0.5', value: String(s.maxGpuBytes / 1024 ** 3) }) as HTMLInputElement;
  const val = h('span', { class: 'mono' }, `${(s.maxGpuBytes / 1024 ** 3).toFixed(1)} GB`);
  budget.addEventListener('input', () => { val.textContent = `${Number(budget.value).toFixed(1)} GB`; });
  const tier = h('select', {},
    h('option', { value: 'best', selected: s.storageTier === 'best' }, '最佳质量（保留原图/16bit）'),
    h('option', { value: 'balanced', selected: s.storageTier === 'balanced' }, '平衡（缩略图+原图）'),
    h('option', { value: 'low', selected: s.storageTier === 'low' }, '省空间（仅缩略图，提示风险）'),
  ) as HTMLSelectElement;
  const focusDir = h('select', {},
    h('option', { value: 'top-to-bottom', selected: s.focusDir === 'top-to-bottom' }, '文件名递增 = 由近及远'),
    h('option', { value: 'bottom-to-top', selected: s.focusDir === 'bottom-to-top' }, '文件名递增 = 由远及近'),
  ) as HTMLSelectElement;
  const step = h('input', { type: 'number', value: String(s.defaultStepMicron), min: '0.1', step: '0.1' }) as HTMLInputElement;
  const save = h('button', {}, '保存设置');
  save.addEventListener('click', async () => {
    await store.saveSettings({
      maxGpuBytes: Number(budget.value) * 1024 ** 3,
      storageTier: tier.value as typeof s.storageTier,
      focusDir: focusDir.value as typeof s.focusDir,
      defaultStepMicron: Number(step.value),
    });
    toast('设置已保存', 'success');
  });
  // 8G/200 层/2400 万像素驻留能力估算
  const perLayer24mp = 24_000_000 * 4;
  const resident = layersInBudget(6000, 4000, 4, s.maxGpuBytes);
  void perLayer24mp;
  return h('div', { class: 'card' },
    h('h2', {}, 'GPU 与存储'),
    h('div', { class: 'grid cols-2' },
      h('div', {}, h('label', {}, `显存预算（当前） ${val.textContent}`), budget),
      field('存储档位', tier),
      field('对焦方向约定', focusDir),
      field('默认步距 μm', step),
    ),
    h('div', { class: 'muted mono', style: 'margin-top:8px' },
      `当前预算下 512px 分块可波次驻留约 ${resident} 层；200 层 2400 万像素栈按分块流式处理，不整摞进显存`),
    h('div', { class: 'row', style: 'margin-top:10px' }, save),
  );
}

function targetsCard(s: typeof store.settings): HTMLElement {
  const psnr = h('input', { type: 'number', value: String(s.psnrTargetDb), step: '0.5' }) as HTMLInputElement;
  const depth = h('input', { type: 'number', value: String(s.depthErrorTargetStep), step: '0.1' }) as HTMLInputElement;
  const halo = h('input', { type: 'number', value: String(s.haloTargetPx), step: '0.5' }) as HTMLInputElement;
  const save = h('button', {}, '保存验收目标');
  save.addEventListener('click', async () => {
    await store.saveSettings({
      psnrTargetDb: Number(psnr.value),
      depthErrorTargetStep: Number(depth.value),
      haloTargetPx: Number(halo.value),
    });
    toast('验收目标已更新', 'success');
  });
  return h('div', { class: 'card' },
    h('h2', {}, '验收指标（合成结果对真值）'),
    h('div', { class: 'grid cols-3' },
      field('全景深 PSNR 增益 dB（≥）', psnr),
      field('深度中位误差 步距（≤）', depth),
      field('刚毛光晕宽度 px（≤）', halo),
    ),
    h('div', { class: 'row', style: 'margin-top:10px' }, save),
  );
}

function stressCard(s: typeof store.settings): HTMLElement {
  const layers = h('input', { type: 'number', value: String(s.stressStackLayers), min: '10', max: '200' }) as HTMLInputElement;
  const pixels = h('input', { type: 'number', value: String(s.stressStackPixels), min: '1000000', step: '1000000' }) as HTMLInputElement;
  const out = h('div', { class: 'muted mono' }, '压力测试在小尺寸合成栈上验证算法指标；200 层/2400 万像素的显存安全由分块流式保证（见 GPU 卡估算）。');
  const run = h('button', {}, '跑 320×240×24 层验收（确定性种子）');
  run.addEventListener('click', () => {
    const syn = generateSyntheticStack({
      width: 320, height: 240, layers: 24, stepMicron: 2,
      breathScalePct: 1.2, jitterPx: 1.5, seed: 42,
      includeWing: true, includeSetae: true, includePunctures: true,
    });
    const fused = fuseDff(syn.images, syn.width, syn.height, {
      windowRadius: 2, topK: 4, lowTextureThresh: 0.12, specularLum: 235, softTemp: 8, haloGuard: 0.8,
    });
    const m = evaluate(syn, fused, defaultTargets(syn.spec));
    out.replaceChildren(
      h('div', { class: m.pass ? 'metric-big pass' : 'metric-big fail' }, m.pass ? '验收通过' : '未达标'),
      h('pre', { class: 'mono' }, JSON.stringify({
        fusedPsnrDb: +m.fusedPsnrDb.toFixed(2),
        bestSinglePsnrDb: +m.bestSinglePsnrDb.toFixed(2),
        gainDb: +m.gainDb.toFixed(2),
        depthMedianAbsErrStep: +m.depthMedianAbsErrStep.toFixed(3),
        haloWidthPxMedian: +m.haloWidthPxMedian.toFixed(2),
        lowTextureMaskRatio: +m.lowTextureMaskRatio.toFixed(3),
        specularMaskRatio: +m.specularMaskRatio.toFixed(3),
        holeFilledRatio: +m.holeFilledRatio.toFixed(3),
        reasons: m.reasons,
      }, null, 2)),
    );
  });
  return h('div', { class: 'card' },
    h('h2', {}, '压力测试'),
    h('div', { class: 'grid cols-2' }, field('目标层数', layers), field('目标单层像素', pixels)),
    h('div', { class: 'row', style: 'margin:10px 0' }, run),
    out,
  );
}

function storageCard(): HTMLElement {
  const out = h('div', { class: 'card' }, h('h2', {}, 'IndexedDB 存储占用'), h('div', { class: 'muted' }, '读取中…'));
  db.estimateUsage().then(({ usage, quota }) => {
    out.replaceChildren(
      h('h2', {}, 'IndexedDB 存储占用'),
      h('div', { class: 'progress', style: 'margin-bottom:8px' },
        h('div', { style: `width:${quota ? Math.min(100, (usage / quota) * 100).toFixed(1) : 0}%` })),
      h('div', { class: 'mono' }, `已用 ${fmtBytes(usage)} / 配额 ${fmtBytes(quota)}（${quota ? ((usage / quota) * 100).toFixed(2) : '0'}%）`),
    );
  }).catch(() => {
    out.replaceChildren(h('h2', {}, 'IndexedDB 存储占用'), h('div', { class: 'muted' }, '当前环境无法读取配额'));
  });
  return out;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
