import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadCatalog, loadDetail } from '../app/jem-rescues/data.ts';

const manifest = {
  schemaVersion: 1, assetBaseUrl: 'https://assets.jorgechang.xyz/seed1-batch4/',
  classNames: { 0: 'cat', 1: 'dog' }, catalog: 'catalog.json', detailChunkSize: 500,
  detailFiles: { 0: 'details/000.json', 1: 'details/001.json' },
  counts: { rescued: 1, regressed: 1, both_correct: 0, both_wrong: 0 },
  meta: { method: "decoder_jacobian_vector_product", complete: true, total: 2, baselineAlpha: 0, hybridAlpha: 0.1, confidencePredictionsVerified: true },
};
// Reordered columns ensure mapping follows the published schema rather than fixed offsets.
const catalog = { columns: ['kind', 'datasetIndex', 'trueLabel', 'filename', 'baselinePred', 'hybridPred', 'confidence0', 'confidence01'],
  rows: [['rescued', 499, 0, 'a.JPEG', 1, 0, .7, .8], ['regressed', 500, 1, 'b.JPEG', 1, 0, .9, .6]] };

test('catalog mapping, chunk boundary, cached reads, and recovery after a failed request', async () => {
  const original = global.fetch;
  const calls = [];
  let failed = false;
  global.fetch = async url => {
    calls.push(url);
    if (url.endsWith('manifest.json')) return Response.json(manifest);
    if (url.endsWith('catalog.json')) return Response.json(catalog);
    if (url.endsWith('001.json') && !failed) { failed = true; return new Response('', { status: 503 }); }
    const index = url.endsWith('000.json') ? 0 : 1;
    const r = catalog.rows[index];
    return Response.json({ rows: [{ kind: r[0], datasetIndex: r[1], trueLabel: r[2], baselinePred: r[4], hybridPred: r[5], maps: { true: {} } }] });
  };
  try {
    const data = await loadCatalog();
    assert.equal(data.rows[0].trueClassName, 'cat');
    assert.equal(data.rows[0].crop, 'crops/00499.webp');
    assert.equal((await loadDetail(data.rows[0], data)).datasetIndex, 499);
    await loadDetail(data.rows[0], data);
    assert.equal(calls.filter(x => x.endsWith('000.json')).length, 1);
    await assert.rejects(loadDetail(data.rows[1], data), /503/);
    assert.equal((await loadDetail(data.rows[1], data)).datasetIndex, 500);
  } finally { global.fetch = original; }
});

test('reject inconsistent counts, duplicate IDs, and unverified confidence', async () => {
  const original = global.fetch;
  try {
    for (const mutation of ['counts', 'duplicate', 'confidence']) {
      const m = structuredClone(manifest), c = structuredClone(catalog);
      if (mutation === 'counts') m.counts.rescued = 4;
      if (mutation === 'duplicate') c.rows[1][1] = 499;
      if (mutation === 'confidence') m.meta.confidencePredictionsVerified = false;
      global.fetch = async url => Response.json(url.endsWith('manifest.json') ? m : c);
      await assert.rejects(loadCatalog());
    }
  } finally { global.fetch = original; }
});

test('analysis switching uses separate detail caches and rejects a mismatched method', async () => {
  const original = global.fetch;
  const inputManifest = structuredClone(manifest);
  inputManifest.assetBaseUrl = 'https://assets.jorgechang.xyz/seed1-input-gradients/';
  inputManifest.meta.method = 'input_pixel_gradient';
  const calls = [];
  global.fetch = async url => {
    calls.push(url);
    if (url.endsWith('manifest.json')) return Response.json(inputManifest);
    if (url.endsWith('catalog.json')) return Response.json(catalog);
    return Response.json({ rows: [{ datasetIndex: 499, kind: 'rescued', trueLabel: 0, baselinePred: 1, hybridPred: 0,
      maps: { true: { baseline: 'maps/input-only.webp' } } }] });
  };
  try {
    const input = await loadCatalog('input');
    assert.equal(input.analysis, 'input');
    assert.equal((await loadDetail(input.rows[0], input)).maps.true.baseline, 'maps/input-only.webp');
    assert.ok(calls.includes('/jem-input-data/details/000.json'));
    await assert.rejects(loadCatalog('decoder'), /Incompatible/);
  } finally { global.fetch = original; }
});
