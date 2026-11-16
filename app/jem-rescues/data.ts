export const analyses = {
  input: { label: "Input gradients", metadataRoot: "/jem-input-data/", assetBaseUrl: "https://assets.jorgechang.xyz/seed1-input-gradients/", method: "input_pixel_gradient" },
  decoder: { label: "Decoder projections", metadataRoot: "/jem-data/", assetBaseUrl: "https://assets.jorgechang.xyz/seed1-batch4/", method: "decoder_jacobian_vector_product" },
} as const;
export type Analysis = keyof typeof analyses;
export type Kind = "rescued" | "regressed" | "both_correct" | "both_wrong";
export type ImageRow = {
  datasetIndex: number; filename: string; kind: Kind; trueLabel: number;
  baselinePred: number; hybridPred: number; confidence0: number | null; confidence01: number | null;
  trueClassName: string; baselineClassName: string; hybridClassName: string; crop: string;
};
export type Target = "true" | "margin";
export type MapSet = { baseline: string; hybrid: string; difference: string; signedDifference: string };
export type Stats = { cosine: number; baselineNorm: number; hybridNorm: number; differenceNorm: number;
  targetClass: number; competitorClass: number | null };
export type Detail = ImageRow & { maps: Partial<Record<Target, MapSet>>; stats: Partial<Record<Target, Stats>> };
type Manifest = {
  schemaVersion: number; assetBaseUrl: string; classNames: Record<string, string>;
  catalog: string; detailChunkSize: number; detailFiles: Record<string, string>;
  counts: Record<Kind, number>;
  meta: { method: string; complete: boolean; total: number; processed: number; baselineAlpha: number; hybridAlpha: number;
    confidencePredictionsVerified: boolean; normalization: string };
};
export type RescueData = {
  analysis: Analysis; rows: ImageRow[]; assetBaseUrl: string; manifest: Manifest;
  meta: { nRescued: number; nRegressed: number; nEvaluated: number; nCrops: number; source: string };
};
async function readJson(path: string, analysis: Analysis) {
  if (!/^(manifest\.json|catalog\.json|details\/\d{3}\.json)$/.test(path)) throw new Error("Invalid metadata path");
  const response = await fetch(analyses[analysis].metadataRoot + path);
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
  return response.json();
}
export async function loadCatalog(analysis: Analysis = "decoder"): Promise<RescueData> {
  const manifest: Manifest = await readJson("manifest.json", analysis);
  if (manifest.schemaVersion !== 1 || !manifest.meta.complete || !manifest.meta.confidencePredictionsVerified ||
      manifest.meta.baselineAlpha !== 0 || manifest.meta.hybridAlpha !== 0.1 || manifest.detailChunkSize !== 500 ||
      manifest.assetBaseUrl !== analyses[analysis].assetBaseUrl || manifest.meta.method !== analyses[analysis].method) throw new Error("Incompatible analysis manifest");
  const catalog: { columns: string[]; rows: unknown[][] } = await readJson(manifest.catalog, analysis);
  const required = ["datasetIndex", "filename", "kind", "trueLabel", "baselinePred", "hybridPred", "confidence0", "confidence01"];
  if (!Array.isArray(catalog.columns) || required.some(k => !catalog.columns.includes(k)) ||
      !Array.isArray(catalog.rows) || catalog.rows.length !== manifest.meta.total) throw new Error("Incomplete prediction catalog");
  const ids = new Set<number>();
  const counts: Record<Kind, number> = { rescued: 0, regressed: 0, both_correct: 0, both_wrong: 0 };
  const rows = catalog.rows.map(values => {
    const r = Object.fromEntries(catalog.columns.map((key, i) => [key, values[i]])) as ImageRow;
    if (!Number.isInteger(r.datasetIndex) || r.datasetIndex < 0 || ids.has(r.datasetIndex) ||
        !(r.kind in counts) || typeof r.filename !== "string" ||
        [r.trueLabel, r.baselinePred, r.hybridPred].some(v => !Number.isInteger(v) || !manifest.classNames[String(v)]) ||
        [r.confidence0, r.confidence01].some(v => typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1))
      throw new Error("Invalid prediction row");
    ids.add(r.datasetIndex); counts[r.kind]++;
    return { ...r, trueClassName: manifest.classNames[r.trueLabel], baselineClassName: manifest.classNames[r.baselinePred],
      hybridClassName: manifest.classNames[r.hybridPred], crop: `crops/${String(r.datasetIndex).padStart(5, "0")}.webp` };
  });
  if (Object.keys(counts).some(k => counts[k as Kind] !== manifest.counts[k as Kind])) throw new Error("Catalog counts do not match manifest");
  return { analysis, rows, manifest, assetBaseUrl: manifest.assetBaseUrl, meta: {
    nRescued: counts.rescued, nRegressed: counts.regressed, nEvaluated: rows.length, nCrops: rows.length,
    source: `${analyses[analysis].label} · ImageNet validation · seed 1 · batch 4 · Cloudflare R2` } };
}
const chunks = new Map<string, Promise<{ rows: Detail[] }>>();
export async function loadDetail(row: ImageRow, data: RescueData): Promise<Detail> {
  const chunk = Math.floor(row.datasetIndex / data.manifest.detailChunkSize);
  const path = data.manifest.detailFiles[String(chunk)];
  if (!path) throw new Error("Missing detail chunk");
  const cacheKey = `${data.analysis}:${path}`;
  let request = chunks.get(cacheKey);
  if (!request) {
    request = readJson(path, data.analysis).catch(error => { chunks.delete(cacheKey); throw error; });
    chunks.set(cacheKey, request);
    if (chunks.size > 12) chunks.delete(chunks.keys().next().value!);
  }
  const detail = (await request).rows.find(r => r.datasetIndex === row.datasetIndex);
  if (!detail || detail.kind !== row.kind || detail.trueLabel !== row.trueLabel ||
      detail.baselinePred !== row.baselinePred || detail.hybridPred !== row.hybridPred || !detail.maps.true)
    throw new Error("Missing or inconsistent image details");
  return detail;
}
