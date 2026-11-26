// 页面 2：图层栈——导入（几十~两百张 JPG/16bit TIFF）、排序、质检标记、对齐残差曲线、控制点
import type { LayerRecord, LayerStack, ViewPoint } from '../../types';
import { db, newId } from '../../lib/storage';
import { store } from '../../state/store';
import { h, clear } from '../dom';
import { navigate, toast } from '../../main';
import { meanLuminance, orderLayers, type ImportedFile } from '../../lib/stackQc';
import { alignStack, DEFAULT_ALIGN, warpLayer } from '../../lib/alignment';
import type { AlignResidual, ControlPoint } from '../../types';
import { VIEW_POINTS } from '../../types';

export function stacksPage(): HTMLElement {
  const root = h('div', {});
  let residuals: AlignResidual[] | null = null;
  let controlPoints: ControlPoint[] = [];

  async function ensureBatch(): Promise<void> {
    if (!store.currentBatchId && store.batches[0]) store.selectBatch(store.batches[0].id);
    if (store.currentBatchId) await store.refreshStacks(store.currentBatchId);
  }

  const render = () => {
    const batch = store.batches.find((b) => b.id === store.currentBatchId);
    clear(root);
    root.append(
      h('h1', {}, '图层栈'),
      batchSelector(),
      batch
        ? h('div', { class: 'muted', style: 'margin-bottom:12px' },
          `当前批次：${batch.code} · ${batch.cargo} · ${batch.hostPlant}`)
        : h('div', { class: 'card muted' }, '请先在「截获批次」页登记并选择一个批次。'),
      ...(batch ? [importCard(), stackList()] : []),
    );
  };

  function batchSelector(): HTMLElement {
    const sel = h('select', {},
      ...store.batches.map((b) => h('option', { value: b.id, selected: b.id === store.currentBatchId }, `${b.code} · ${b.interceptDate}`)),
    ) as HTMLSelectElement;
    sel.addEventListener('change', () => store.selectBatch(sel.value));
    return h('div', { class: 'card', style: 'max-width:420px' }, h('label', {}, '选择批次'), sel);
  }

  function importCard(): HTMLElement {
    const viewpoint = h('select', {}, ...VIEW_POINTS.map((v) => h('option', { value: v.id }, v.label))) as HTMLSelectElement;
    const specimen = h('input', { placeholder: '虫体编号，如 S01' });
    const step = h('input', { type: 'number', value: String(store.settings.defaultStepMicron), min: '0.1', step: '0.1' }) as HTMLInputElement;
    const mag = h('input', { type: 'number', value: '5', min: '0.1', step: '0.5' }) as HTMLInputElement;
    const fileInput = h('input', { type: 'file', multiple: true, accept: '.jpg,.jpeg,.tif,.tiff,.png' }) as HTMLInputElement;
    const btn = h('button', {}, '导入并质检');
    const status = h('div', { class: 'muted mono', style: 'margin-top:8px' });

    btn.addEventListener('click', async () => {
      const files = Array.from(fileInput.files ?? []);
      if (files.length < 2) { toast('至少选择 2 张图层', 'error'); return; }
      btn.disabled = true;
      try {
        const imported: ImportedFile[] = [];
        for (let i = 0; i < files.length; i++) {
          status.textContent = `解码 ${i + 1}/${files.length}：${files[i].name}`;
          const bmp = await createImageBitmap(files[i]);
          const canvas = new OffscreenCanvas(bmp.width, bmp.height);
          const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
          ctx.drawImage(bmp, 0, 0);
          const data = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
          const lum = meanLuminance(data, bmp.width * bmp.height);
          const isTiff = /\.tiff?$/i.test(files[i].name);
          imported.push({
            fileName: files[i].name,
            exposureValue: lum,
            width: bmp.width, height: bmp.height,
            bitsPerSample: isTiff ? 16 : 8,
            mimeType: files[i].type || (isTiff ? 'image/tiff' : 'image/jpeg'),
          });
          bmp.close();
        }
        const qc = orderLayers(imported);
        status.textContent = `质检完成：${qc.ordered.length} 层，异常 ${qc.flags.size} 个。正在写入 IndexedDB…`;
        const width = imported[0].width, height = imported[0].height;
        const stack: LayerStack = {
          id: newId('stack'),
          batchId: store.currentBatchId!,
          specimenId: specimen.value.trim() || 'S01',
          viewpoint: viewpoint.value as ViewPoint,
          width, height,
          stepMicron: Number(step.value) || store.settings.defaultStepMicron,
          lensMagnification: Number(mag.value) || 5,
          layerIds: [],
          createdAt: Date.now(), updatedAt: Date.now(),
        };
        const nameToFile = new Map(files.map((f) => [f.name, f]));
        for (const item of qc.ordered) {
          const f = nameToFile.get(item.file.fileName)!;
          const buf = await f.arrayBuffer();
          const blob = new Blob([buf], { type: item.file.mimeType });
          const thumbCanvas = new OffscreenCanvas(0, 0);
          const bmp = await createImageBitmap(blob);
          const sc = Math.min(1, 160 / Math.max(bmp.width, bmp.height));
          thumbCanvas.width = Math.max(1, bmp.width * sc);
          thumbCanvas.height = Math.max(1, bmp.height * sc);
          thumbCanvas.getContext('2d')!.drawImage(bmp, 0, 0, thumbCanvas.width, thumbCanvas.height);
          const thumb = await thumbCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.65 });
          bmp.close();
          const flagInfo = qc.flags.get(item.file.fileName);
          const rec: LayerRecord = {
            id: newId('layer'),
            stackId: stack.id,
            index: item.index,
            fileName: item.file.fileName,
            mimeType: item.file.mimeType,
            bitsPerSample: item.file.bitsPerSample,
            width: item.file.width, height: item.file.height,
            focusPosition: item.focusPosition,
            exposureValue: item.file.exposureValue,
            flag: flagInfo?.flag ?? 'ok',
            flagReason: flagInfo?.reason,
            blobKey: `${stack.id}_${item.index}.bin`,
            thumbKey: `${stack.id}_${item.index}_thumb.jpg`,
          };
          await db.saveBlob(rec.blobKey, blob);
          await db.saveBlob(rec.thumbKey, thumb);
          await db.saveLayer(rec);
          stack.layerIds.push(rec.id);
        }
        await db.saveStack(stack);
        await store.refreshStacks(store.currentBatchId!);
        toast(`图层栈已建立：${stack.layerIds.length} 层`, 'success');
        status.textContent = '';
      } catch (e) {
        toast(`导入失败：${(e as Error).message}`, 'error', 6000);
      } finally {
        btn.disabled = false;
      }
    });

    return h('div', { class: 'card' },
      h('h2', {}, '新建图层栈（单虫体单机位）'),
      h('div', { class: 'grid cols-4' },
        field('机位', viewpoint), field('虫体编号', specimen),
        field('对焦步距 (μm)', step), field('放大倍率 (×)', mag),
      ),
      h('div', { style: 'margin-top:10px' }, h('label', {}, '图层文件（JPG / 16bit TIFF，可一次选 200 张）'), fileInput),
      h('div', { class: 'row', style: 'margin-top:10px' }, btn),
      status,
    );
  }

  function stackList(): HTMLElement {
    if (store.stacks.length === 0) return h('div', { class: 'card muted' }, '该批次暂无图层栈。');
    return h('div', {}, ...store.stacks.map((s) => stackCard(s)));
  }

  function stackCard(s: LayerStack): HTMLElement {
    const layers = store.layers[s.id] ?? [];
    const flagged = layers.filter((l) => l.flag !== 'ok');
    const jobs = store.jobs[s.id] ?? [];
    return h('div', { class: 'card' },
      h('div', { class: 'row' },
        h('strong', {}, `${s.specimenId} · ${VIEW_POINTS.find((v) => v.id === s.viewpoint)?.label}`),
        h('span', { class: 'tag info' }, `${s.width}×${s.height}`),
        h('span', { class: 'tag info' }, `${layers.length} 层`),
        h('span', { class: 'tag info mono' }, `步距 ${s.stepMicron}μm · ${s.lensMagnification}×`),
        h('div', { class: 'spacer' }),
        h('button', { class: 'small', onclick: () => { store.selectStack(s.id); navigate('#/fusion'); } }, '去合成'),
        h('button', { class: 'small ghost', onclick: () => { store.selectStack(s.id); navigate('#/relief'); } }, '浮雕'),
        h('button', { class: 'small ghost', onclick: () => { store.selectStack(s.id); navigate('#/measure'); } }, '测量'),
      ),
      flagged.length ? h('div', { class: 'row', style: 'margin-top:8px' },
        ...flagged.map((l) => h('span', { class: `tag ${l.flag === 'duplicate' ? 'bad' : 'warn'}` },
          `${l.fileName}：${l.flagReason ?? l.flag}`))) : null,
      h('div', { class: 'layer-strip', style: 'margin-top:10px' },
        ...layers.map((l) => layerThumb(s, l))),
      h('div', { class: 'row', style: 'margin-top:10px' },
        h('button', { class: 'small ghost', onclick: () => runAlignment(s) }, '估计对齐（呼吸/晃动校正）'),
        residuals && store.currentStackId === s.id ? residualChart(residuals) : null,
      ),
      jobs.length ? h('div', { class: 'muted mono', style: 'margin-top:6px' },
        `历史作业：${jobs.map((j) => `${j.status} ${(j.progress * 100).toFixed(0)}%`).join(' / ')}`) : null,
    );
  }

  function layerThumb(s: LayerStack, l: LayerRecord): HTMLElement {
    const img = h('img', { class: 'thumb' }) as HTMLImageElement;
    db.getBlob(l.thumbKey).then((b) => { if (b) img.src = URL.createObjectURL(b); });
    const toggle = h('button', {
      class: 'small ghost', onclick: async () => {
        l.flag = l.flag === 'excluded' ? 'ok' : 'excluded';
        await db.saveLayer(l);
        await store.refreshStacks(store.currentBatchId!);
      },
    }, l.flag === 'excluded' ? '已剔除' : '剔除');
    return h('div', { class: 'item' },
      img,
      h('div', { class: 'mono', title: l.flagReason ?? '' }, `#${l.index} ${l.flag !== 'ok' ? h('span', { class: 'tag warn' }, l.flag).textContent : ''}`),
      toggle,
    );
  }

  async function runAlignment(s: LayerStack): Promise<void> {
    store.selectStack(s.id);
    const layers = (await db.listLayers(s.id)).filter((l) => l.flag !== 'excluded');
    toast('正在估计逐层缩放与平移…', 'info');
    // 小图直接全分辨率；大图降采样到 1024 长边做估计
    const target = 1024;
    const scale = Math.min(1, target / Math.max(s.width, s.height));
    const images: Float32Array[] = [];
    for (const l of layers) {
      const blob = await db.getBlob(l.blobKey);
      const bmp = await createImageBitmap(blob!);
      const cw = Math.max(1, Math.round(bmp.width * scale)), chh = Math.max(1, Math.round(bmp.height * scale));
      const cv = new OffscreenCanvas(cw, chh);
      const ctx = cv.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(bmp, 0, 0, cw, chh);
      const d = ctx.getImageData(0, 0, cw, chh).data;
      const rgba = new Float32Array(d.length);
      for (let i = 0; i < d.length; i++) rgba[i] = d[i];
      images.push(rgba);
      bmp.close();
    }
    const opts = { ...DEFAULT_ALIGN, searchRadius: Math.ceil(DEFAULT_ALIGN.searchRadius * scale + 2) };
    const w = Math.round(s.width * scale), hh = Math.max(1, Math.round(s.height * scale));
    const aligned = alignStack(images, w, hh, opts, controlPoints);
    residuals = aligned.residuals;
    const over = residuals.filter((x) => x.overThreshold);
    toast(`对齐完成：${over.length} 层残差超阈值，可拖控制点重配`, over.length ? 'info' : 'success', 5000);
    render();
  }

  function residualChart(data: AlignResidual[]): HTMLElement {
    const cv = h('canvas', { class: 'residual-chart', width: 520, height: 180 }) as HTMLCanvasElement;
    requestAnimationFrame(() => {
      const ctx = cv.getContext('2d')!;
      const W = cv.width, H = cv.height, pad = 28;
      ctx.clearRect(0, 0, W, H);
      const maxV = Math.max(...data.map((d) => d.ssd), 1e-6);
      ctx.strokeStyle = '#333947'; ctx.beginPath();
      ctx.moveTo(pad, H - pad); ctx.lineTo(W - 8, H - pad); ctx.stroke();
      ctx.strokeStyle = '#5b9dff'; ctx.lineWidth = 2; ctx.beginPath();
      data.forEach((d, i) => {
        const x = pad + (i / Math.max(1, data.length - 1)) * (W - pad - 12);
        const y = (H - pad) - (d.ssd / maxV) * (H - pad * 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = '#ef6b6b';
      data.forEach((d, i) => {
        if (!d.overThreshold) return;
        const x = pad + (i / Math.max(1, data.length - 1)) * (W - pad - 12);
        const y = (H - pad) - (d.ssd / maxV) * (H - pad * 2);
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
      });
      ctx.fillStyle = '#9aa3b2'; ctx.font = '11px monospace';
      ctx.fillText('层索引 →', W - 70, H - 8);
      ctx.fillText('归一化 SSD', 4, 14);
    });
    return h('div', { class: 'canvas-wrap', style: 'flex:1;min-width:300px' }, cv,
      h('div', { class: 'muted', style: 'font-size:11px' }, '红点=残差超阈值层；控制点重配在合成页进行'));
  }

  const unsub = store.subscribe(render);
  ensureBatch().then(render);
  setTimeout(() => unsub, 1000 * 60 * 30);
  return root;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
