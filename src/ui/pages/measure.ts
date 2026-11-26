// 页面 5：测量比对——测微尺标定（按倍率存）、长度/宽度/角度/曲线、三维长度、并排比对
import type { Calibration, Measurement, MeasureKind } from '../../types';
import { db, newId } from '../../lib/storage';
import { store } from '../../state/store';
import { h, clear } from '../dom';
import { toast } from '../../main';
import {
  buildMeasurement, calibrate, denormalizePoints, measureAngle,
  measureCurve, measureLength, measureWidth, pickCalibration,
} from '../../lib/measure';
import { decodeGray16Png, decodePng } from '../../lib/imageCodec';

export function measurePage(): HTMLElement {
  const root = h('div', {});
  const render = () => {
    clear(root);
    root.append(h('h1', {}, '测量比对'), calibrationCard(), workspaceCard(), comparisonCard());
  };
  store.subscribe(render);
  render();
  return root;
}

function calibrationCard(): HTMLElement {
  const mag = h('input', { type: 'number', value: '5', step: '0.5', min: '0.5' }) as HTMLInputElement;
  const known = h('input', { type: 'number', value: '0.1', step: '0.01', min: '0.001' }) as HTMLInputElement;
  const span = h('input', { type: 'number', placeholder: '图像上该段的像素跨度', min: '1' }) as HTMLInputElement;
  const source = h('input', { placeholder: '镜台测微尺记录编号，如 OBJ-0.01-20261127' });
  const save = h('button', {}, '保存标定');
  const list = h('div', {});

  function refresh() {
    list.replaceChildren(h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, '倍率'), h('th', {}, '像素/mm'), h('th', {}, '来源'), h('th', {}))),
      h('tbody', {}, ...store.calibrations.map((c) => h('tr', {},
        h('td', { class: 'mono' }, `${c.magnification}×`),
        h('td', { class: 'mono' }, c.pixelsPerMm.toFixed(3)),
        h('td', {}, c.source),
        h('td', {}, h('button', {
          class: 'small danger', onclick: async () => { await db.deleteCalibration(c.id); store.calibrations = await db.listCalibrations(); store.emit(); },
        }, '删除')),
      ))),
    ));
  }
  save.addEventListener('click', async () => {
    try {
      const base = calibrate(Number(known.value), Number(span.value), Number(mag.value), source.value || '手动录入');
      const rec: Calibration = { id: newId('cal'), createdAt: Date.now(), ...base };
      await db.saveCalibration(rec);
      store.calibrations = await db.listCalibrations();
      store.emit();
      toast('标定已保存', 'success');
    } catch (e) {
      toast((e as Error).message, 'error');
    }
  });
  refresh();

  return h('div', { class: 'card' },
    h('h2', {}, '镜台测微尺标定（每个放大倍率一套）'),
    h('div', { class: 'grid cols-4' },
      field('倍率 ×', mag), field('已知长度 mm', known), field('像素跨度 px', span), field('记录编号', source),
    ),
    h('div', { class: 'row', style: 'margin-top:10px' }, save),
    h('div', { style: 'margin-top:10px' }, list),
  );
}

let depthData: Float32Array | null = null;
let stepMicron = 2;

function workspaceCard(): HTMLElement {
  const wrap = h('div', { class: 'card' }, h('h2', {}, '量测工作台'));
  const stack = store.currentStack;
  if (!stack) {
    wrap.append(h('div', { class: 'muted' }, '请先在「图层栈」选择一个栈。'));
    return wrap;
  }
  const stackId = stack.id;
  stepMicron = stack.stepMicron;
  const job = store.currentJobs.find((j) => j.status === 'done' && j.outputColorKey);
  if (!job) {
    wrap.append(h('div', { class: 'muted' }, '该栈尚无合成结果，先到「合成」页完成融合。'));
    return wrap;
  }
  const cal = pickCalibration(store.calibrations, stack.lensMagnification);
  if (!cal) wrap.append(h('div', { class: 'tag warn' }, `未找到 ${stack.lensMagnification}× 标定，量测结果仅显示像素`));

  const kind = h('select', {},
    h('option', { value: 'length' }, '长度（含三维）'),
    h('option', { value: 'width' }, '宽度'),
    h('option', { value: 'angle' }, '角度'),
    h('option', { value: 'curve' }, '曲线长（含三维）'),
  ) as HTMLSelectElement;
  const label = h('input', { placeholder: '测量项名称，如 鞘翅长' });
  const cv = h('canvas', { style: 'max-width:100%;cursor:crosshair;background:#0c0e12;border-radius:8px' }) as HTMLCanvasElement;
  const points: { x: number; y: number }[] = [];
  const result = h('div', { class: 'mono' }, '在图上点击选点（角度需 3 点，曲线可多点，双击结束曲线）');
  const measurements: Measurement[] = [];

  (async () => {
    const colorBlob = await db.getBlob(job.outputColorKey!);
    const img = await decodePng(colorBlob!);
    cv.width = img.width; cv.height = img.height;
    cv.getContext('2d')!.putImageData(img, 0, 0);
    if (job.outputDepthKey) {
      const depthBlob = await db.getBlob(job.outputDepthKey);
      depthData = await decodeGray16Png(depthBlob!, stack.layerIds.length - 1);
    }
    drawPoints();
  })();

  const depthAt = (x: number, y: number) => {
    if (!depthData) return 0;
    return depthData[Math.round(y) * cv.width + Math.round(x)] ?? 0;
  };

  function drawPoints(): void {
    // 重绘底图代价高，这里只叠加标记
    const ctx = cv.getContext('2d')!;
    ctx.fillStyle = '#5b9dff';
    ctx.strokeStyle = '#5b9dff';
    ctx.lineWidth = 2;
    for (const p of points) {
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill();
    }
    if (points.length > 1) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
      ctx.stroke();
    }
  }

  cv.addEventListener('click', (e) => {
    const r = cv.getBoundingClientRect();
    points.push({
      x: (e.clientX - r.left) * cv.width / r.width,
      y: (e.clientY - r.top) * cv.height / r.height,
    });
    drawPoints();
    if (kind.value === 'angle' && points.length === 3) commit();
    if ((kind.value === 'length' || kind.value === 'width') && points.length === 2) commit();
  });
  cv.addEventListener('dblclick', () => { if (kind.value === 'curve' && points.length >= 2) commit(); });

  const reset = h('button', { class: 'small ghost' }, '清除点');
  reset.addEventListener('click', () => { points.length = 0; result.textContent = '已清除'; });

  async function commit(): Promise<void> {
    const k = kind.value as MeasureKind;
    let value = 0, value3d: number | undefined;
    if (!cal) {
      value = k === 'angle' ? measureAngle(points[0], points[1], points[2]) : Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
      result.textContent = `无标定：${value.toFixed(2)} ${k === 'angle' ? '°' : 'px'}`;
    } else if (k === 'length') {
      const r2 = measureLength(points[0], points[1], depthAt, cal, stepMicron);
      value = r2.planar; value3d = r2.length3d;
      result.textContent = `平面 ${value.toFixed(4)} mm ｜ 三维（按深度） ${value3d.toFixed(4)} mm`;
    } else if (k === 'width') {
      value = measureWidth(points[0], points[1], cal);
      result.textContent = `宽度 ${value.toFixed(4)} mm`;
    } else if (k === 'angle') {
      value = measureAngle(points[0], points[1], points[2]);
      result.textContent = `角度 ${value.toFixed(2)}°`;
    } else {
      const r2 = measureCurve(points, depthAt, cal, stepMicron);
      value = r2.planar; value3d = r2.length3d;
      result.textContent = `曲线平面 ${value.toFixed(4)} mm ｜ 三维 ${value3d.toFixed(4)} mm`;
    }
    const base = buildMeasurement(k, label.value || k, points, cv.width, cv.height, value, value3d);
    const rec: Measurement = { id: newId('m'), stackId, createdAt: Date.now(), ...base };
    await db.saveMeasurement(rec);
    measurements.push(rec);
    points.length = 0;
    refreshList();
    toast('量测已保存并挂回批次/栈', 'success');
  }

  const listEl = h('div', {});
  async function refreshList() {
    const all = await db.listMeasurements(stackId);
    listEl.replaceChildren(h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, '名称'), h('th', {}, '类型'), h('th', {}, '值'), h('th', {}, '三维值'), h('th', {}))),
      h('tbody', {}, ...all.map((m) => h('tr', {},
        h('td', {}, m.label), h('td', {}, m.kind),
        h('td', { class: 'mono' }, `${m.value.toFixed(4)}${m.kind === 'angle' ? '°' : ' mm'}`),
        h('td', { class: 'mono' }, m.value3d !== undefined ? `${m.value3d.toFixed(4)} mm` : '—'),
        h('td', {}, h('button', {
          class: 'small danger', onclick: async () => { await db.deleteMeasurement(m.id); refreshList(); },
        }, '删')),
      ))),
    ));
  }
  refreshList();

  wrap.append(
    h('div', { class: 'row' }, field('类型', kind), h('div', { style: 'flex:1' }, field('名称', label)), reset),
    h('div', { class: 'canvas-wrap', style: 'margin-top:10px' }, cv),
    result,
    h('h3', {}, '本栈量测记录'), listEl,
  );
  return wrap;
}

function comparisonCard(): HTMLElement {
  const wrap = h('div', { class: 'card' }, h('h2', {}, '同种参考标本并排比对'));
  const allStacks = store.stacks;
  if (allStacks.length < 1) {
    wrap.append(h('div', { class: 'muted' }, '需要至少一个图层栈。'));
    return wrap;
  }
  const selA = h('select', {}, ...allStacks.map((s) => h('option', { value: s.id }, `${s.specimenId} ${s.viewpoint} ${s.lensMagnification}×`))) as HTMLSelectElement;
  const selB = h('select', {}, ...allStacks.map((s) => h('option', { value: s.id }, `${s.specimenId} ${s.viewpoint} ${s.lensMagnification}×`))) as HTMLSelectElement;
  const opinion = h('textarea', { rows: 2, placeholder: '鉴定意见：形态差异、齿/刻点比对结论…' });
  const save = h('button', {}, '保存比对意见（挂回批次）');

  save.addEventListener('click', async () => {
    const a = allStacks.find((s) => s.id === selA.value)!;
    const b = allStacks.find((s) => s.id === selB.value)!;
    const msA = await db.listMeasurements(a.id);
    await db.saveComparison({
      id: newId('cmp'),
      stackIdA: a.id, stackIdB: b.id,
      viewpoint: a.viewpoint,
      magnification: a.lensMagnification,
      opinion: opinion.value,
      measureIds: msA.map((m) => m.id),
      createdAt: Date.now(),
    });
    toast('比对意见已保存', 'success');
  });

  // 视角/倍率统一提示：选择不同倍率时给出警告（自动统一在合成图重采样阶段实现）
  const warn = h('div', { class: 'muted' });
  const checkMag = () => {
    const a = allStacks.find((s) => s.id === selA.value);
    const b = allStacks.find((s) => s.id === selB.value);
    warn.textContent = a && b && Math.abs(a.lensMagnification - b.lensMagnification) > 0.01
      ? `⚠ 倍率不同（${a.lensMagnification}× vs ${b.lensMagnification}×），系统将按像素当量重采样统一显示`
      : '两边倍率一致，视角按所选机位对齐';
  };
  selA.addEventListener('change', checkMag);
  selB.addEventListener('change', checkMag);
  checkMag();

  wrap.append(
    h('div', { class: 'grid cols-2' }, field('本标本', selA), field('参考标本', selB)),
    warn,
    h('div', { style: 'margin-top:8px' }, h('label', {}, '鉴定意见'), opinion),
    h('div', { class: 'row', style: 'margin-top:8px' }, save),
  );
  return wrap;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
