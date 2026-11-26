import { store } from './state/store';
import { requestGpu } from './gpu/device';
import { h, mount } from './ui/dom';
import { batchesPage } from './ui/pages/batches';
import { stacksPage } from './ui/pages/stacks';
import { fusionPage } from './ui/pages/fusion';
import { reliefPage } from './ui/pages/relief';
import { measurePage } from './ui/pages/measure';
import { settingsPage } from './ui/pages/settings';

const ROUTES: Record<string, { label: string; render: () => Promise<HTMLElement> | HTMLElement }> = {
  '#/batches': { label: '截获批次', render: () => batchesPage() },
  '#/stacks': { label: '图层栈', render: () => stacksPage() },
  '#/fusion': { label: '合成', render: () => fusionPage() },
  '#/relief': { label: '三维浮雕', render: () => reliefPage() },
  '#/measure': { label: '测量比对', render: () => measurePage() },
  '#/settings': { label: '设置', render: () => settingsPage() },
};

async function boot(): Promise<void> {
  const caps = await requestGpu();
  store.gpuCaps = caps;
  await store.hydrate();
  if (!location.hash) location.hash = '#/batches';
  window.addEventListener('hashchange', route);
  route();
}

async function route(): Promise<void> {
  const key = location.hash in ROUTES ? location.hash : '#/batches';
  const caps = store.gpuCaps;
  const gpuStatus = h('div', { class: 'gpu-status' },
    h('div', {}, caps?.supported
      ? h('span', { class: 'gpu-ok' }, '● WebGPU 可用')
      : h('span', { class: 'gpu-bad' }, `● ${caps && 'reason' in caps ? caps.reason : 'WebGPU 未初始化'}`)),
    caps?.supported ? h('div', { class: 'mono' },
      `${caps.info?.vendor ?? 'GPU'} · 预算 ${(store.settings.maxGpuBytes / 1024 ** 3).toFixed(1)}GB`) : null,
  );
  const nav = h('nav', { class: 'side' },
    h('div', { class: 'brand' }, '景深合成与浮雕系统', h('small', {}, '小蠹 / 象甲鉴定工作台')),
    ...Object.entries(ROUTES).map(([hash, r]) =>
      h('a', { class: hash === key ? 'active' : '', onclick: () => { location.hash = hash; } }, r.label)),
    gpuStatus,
  );
  const page = await ROUTES[key].render();
  const shell = h('div', { style: 'display: contents' }, nav, h('main', {}, page));
  mount(document.getElementById('app')!, shell);
}

export function navigate(hash: string): void {
  location.hash = hash;
}

export function toast(message: string, kind: 'info' | 'error' | 'success' = 'info', ms = 3200): void {
  const el = h('div', { class: `toast ${kind}` }, message);
  document.body.append(el);
  setTimeout(() => el.remove(), ms);
}

boot().catch((e) => {
  document.getElementById('app')!.append(
    h('div', { class: 'card' }, h('h1', {}, '启动失败'), h('pre', { class: 'mono' }, String(e?.stack ?? e))),
  );
});
