// 页面 1：截获批次登记（日期/货物/寄主植物）与列表
import type { Batch } from '../../types';
import { db, newId } from '../../lib/storage';
import { store } from '../../state/store';
import { h } from '../dom';
import { navigate, toast } from '../../main';

export function batchesPage(): HTMLElement {
  const root = h('div', {});
  const render = () => {
    root.replaceChildren(
      h('h1', {}, '截获批次'),
      formCard(),
      h('h2', {}, `已登记批次（${store.batches.length}）`),
      tableCard(),
    );
  };

  function formCard(): HTMLElement {
    const code = h('input', { placeholder: '如 HC2026-1127-03', value: `HC${new Date().getFullYear()}-${String(Date.now()).slice(-5)}` });
    const date = h('input', { type: 'date', value: new Date().toISOString().slice(0, 10) });
    const cargo = h('input', { placeholder: '如 进境木材集装箱 / 原木 / 木质包装' });
    const host = h('input', { placeholder: '如 松属 Pinus sp. / 云杉 Picea sp.' });
    const species = h('input', { placeholder: '初步鉴定（可后补）' });
    const notes = h('textarea', { rows: 2, placeholder: '备注' });
    const submit = h('button', {}, '登记批次');
    submit.addEventListener('click', async () => {
      if (!cargo.value.trim() || !host.value.trim()) {
        toast('来源货物与寄主植物必填', 'error'); return;
      }
      const batch: Batch = {
        id: newId('batch'),
        code: code.value.trim(),
        interceptDate: date.value,
        cargo: cargo.value.trim(),
        hostPlant: host.value.trim(),
        species: species.value.trim() || undefined,
        notes: notes.value.trim() || undefined,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      await db.saveBatch(batch);
      await store.refreshBatches();
      toast('批次已登记', 'success');
      store.selectBatch(batch.id);
      navigate('#/stacks');
    });
    return h('div', { class: 'card' },
      h('h2', {}, '新批次登记'),
      h('div', { class: 'grid cols-3' },
        field('批次号', code), field('截获日期', date), field('初步鉴定种', species),
        field('来源货物', cargo), field('寄主植物', host), field('备注', notes),
      ),
      h('div', { class: 'row', style: 'margin-top:12px' }, submit),
    );
  }

  function tableCard(): HTMLElement {
    if (store.batches.length === 0) {
      return h('div', { class: 'card muted' }, '暂无批次，先在上方登记。');
    }
    return h('div', { class: 'card' },
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, '批次号'), h('th', {}, '截获日期'), h('th', {}, '来源货物'),
          h('th', {}, '寄主植物'), h('th', {}, '种名'), h('th', {}, '图层栈'), h('th', {}, '操作'),
        )),
        h('tbody', {}, ...store.batches.map((b) => {
          const stackCount = store.stacks?.filter((s) => s.batchId === b.id).length;
          return h('tr', {},
            h('td', { class: 'mono' }, b.code),
            h('td', {}, b.interceptDate),
            h('td', {}, b.cargo),
            h('td', {}, b.hostPlant),
            h('td', {}, b.species ?? '—'),
            h('td', {}, String(stackCount ?? '·')),
            h('td', {}, h('div', { class: 'row' },
              h('button', {
                class: 'small', onclick: () => { store.selectBatch(b.id); navigate('#/stacks'); },
              }, '图层栈'),
              h('button', {
                class: 'small ghost', onclick: async () => {
                  const sp = prompt('复核种名', b.species ?? '');
                  if (sp !== null) {
                    await db.saveBatch({ ...b, species: sp || undefined, updatedAt: Date.now() });
                    await store.refreshBatches();
                  }
                },
              }, '种名'),
              h('button', {
                class: 'small danger', onclick: async () => {
                  if (!confirm(`删除批次 ${b.code}？其下图层栈与合成结果将一并删除。`)) return;
                  await db.deleteBatch(b.id);
                  await store.refreshBatches();
                  toast('已删除', 'success');
                },
              }, '删除'),
            )),
          );
        })),
      ),
    );
  }

  store.subscribe(render);
  render();
  return root;
}

function field(label: string, input: HTMLElement): HTMLElement {
  return h('div', {}, h('label', {}, label), input);
}
