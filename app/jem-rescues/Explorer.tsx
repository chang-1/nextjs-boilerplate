"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import styles from "./explorer.module.css";

type Kind = "rescued" | "regressed";
type ImageRow = {
  datasetIndex: number;
  filename: string;
  kind: Kind;
  trueLabel: number;
  trueClassName: string;
  baselinePred: number;
  baselineClassName: string;
  hybridPred: number;
  hybridClassName: string;
  confidence0: number | null;
  confidence01: number | null;
  crop: string | null;
};
type RescueData = {
  schemaVersion: 2;
  meta: {
    dataset: string;
    seed: number;
    baselineAlpha: number;
    hybridAlpha: number;
    nEvaluated: number;
    nChanged: number;
    nRescued: number;
    nRegressed: number;
    nCrops: number;
    source: string;
  };
  rows: ImageRow[];
};
type Status = "Both" | "Rescued" | "Regressed";
type Tab = "Gallery" | "Class counts" | "Confusions";
type Sort = "ImageNet order" | "Class" | "Random" | "Hybrid confidence high to low" | "Hybrid confidence low to high";
const number = (n: number) => n.toLocaleString("en-US");
const confidence = (n: number | null) => n === null || n === undefined ? "" : ` (${(n * 100).toFixed(1)}%)`;
const numeric = (v: number) => String(v).padStart(4, "0");
const deterministicRandom = (id: number) => ((Math.imul(id + 42, 2654435761) >>> 0) / 4294967296);

function downloadCsv(name: string, rows: Record<string, string | number | null>[]) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const escape = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [keys.map(escape).join(","), ...rows.map(r => keys.map(k => escape(r[k])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Explorer() {
  const [data, setData] = useState<RescueData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("Both");
  const [trueClass, setTrueClass] = useState<string>("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("ImageNet order");
  const [pageSize, setPageSize] = useState(24);
  const [page, setPage] = useState(1);
  const [tab, setTab] = useState<Tab>("Gallery");
  const search = useDeferredValue(query.trim().toLowerCase());

  useEffect(() => {
    let valid = true;
    fetch("/jem-analysis/results.json")
      .then(async response => {
        if (!response.ok) throw new Error(`Could not load /jem-analysis/results.json (HTTP ${response.status})`);
        return response.json() as Promise<RescueData>;
      })
      .then(json => {
        if (json.schemaVersion !== 2 || json.meta?.baselineAlpha !== 0 ||
            json.meta.hybridAlpha !== 0.1 || !Array.isArray(json.rows)) {
          throw new Error("Incompatible data. Use pack_existing_streamlit_data.py on your existing Streamlit CSV.");
        }
        if (valid) setData(json);
      })
      .catch(e => { if (valid) setError(e instanceof Error ? e.message : String(e)); });
    return () => { valid = false; };
  }, []);

  const classes = useMemo(() => {
    if (!data) return [];
    const map = new Map<number, string>();
    for (const r of data.rows) map.set(r.trueLabel, r.trueClassName);
    return Array.from(map.entries()).sort((a, b) => a[1].localeCompare(b[1]));
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const rows = data.rows.filter(r => {
      if (status !== "Both" && r.kind !== status.toLowerCase()) return false;
      if (trueClass !== "all" && r.trueLabel !== Number(trueClass)) return false;
      if (search && ![r.filename, r.trueClassName, r.baselineClassName, r.hybridClassName,
                        `${r.datasetIndex}`].some(x => x.toLowerCase().includes(search))) return false;
      return true;
    });
    if (sort === "Class") rows.sort((a, b) => a.trueClassName.localeCompare(b.trueClassName) || a.datasetIndex - b.datasetIndex);
    else if (sort === "Random") rows.sort((a, b) => deterministicRandom(a.datasetIndex) - deterministicRandom(b.datasetIndex));
    else if (sort.startsWith("Hybrid confidence")) {
      rows.sort((a, b) => (a.confidence01 === null ? 1 : 0) - (b.confidence01 === null ? 1 : 0) ||
        (sort.endsWith("high to low") ? (b.confidence01 ?? 0) - (a.confidence01 ?? 0) : (a.confidence01 ?? 0) - (b.confidence01 ?? 0)));
    } else rows.sort((a, b) => a.datasetIndex - b.datasetIndex);
    return rows;
  }, [data, status, trueClass, search, sort]);

  const classRows = useMemo(() => {
    const map = new Map<number, {id: number, name: string, rescued: number, regressed: number}>();
    for (const r of data?.rows ?? []) {
      let item = map.get(r.trueLabel);
      if (!item) { item = {id: r.trueLabel, name: r.trueClassName, rescued: 0, regressed: 0}; map.set(r.trueLabel, item); }
      item[r.kind] += 1;
    }
    return Array.from(map.values()).map(r => ({...r, net: r.rescued - r.regressed}))
      .sort((a, b) => b.net - a.net || b.rescued - a.rescued);
  }, [data]);

  const confusions = useMemo(() => {
    const map = new Map<string, {kind: string, trueClass: string, wrongClass: string, count: number}>();
    for (const r of data?.rows ?? []) {
      const wrongClass = r.kind === "rescued" ? r.baselineClassName : r.hybridClassName;
      const k = JSON.stringify([r.kind, r.trueClassName, wrongClass]);
      const item = map.get(k) ?? {kind: r.kind, trueClass: r.trueClassName, wrongClass, count: 0};
      item.count++;
      map.set(k, item);
    }
    return [...map.values()].sort((a, b) => b.count - a.count);
  }, [data]);

  useEffect(() => { setPage(1); }, [status, trueClass, query, sort, pageSize]);
  const nPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const activePage = Math.min(page, nPages);
  const pageRows = filtered.slice((activePage - 1) * pageSize, activePage * pageSize);
  const rescued = data?.meta.nRescued ?? 0;
  const regressed = data?.meta.nRegressed ?? 0;
  const maxCount = Math.max(1, ...classRows.slice(0, 30).map(x => Math.max(x.rescued, x.regressed)));

  return <div className={styles.site}>
    <aside className={styles.sidebar}>
      <a className={styles.back} href="/">← Research homepage</a>
      <div className={styles.sidebarTitle}>JEM rescue explorer</div>
      <div className={styles.sideDivider} />
      <div className={styles.sideHeading}>Data source</div>
      <div className={styles.fixedField}><span>Baseline model</span><strong>α = 0.0</strong></div>
      <div className={styles.fixedField}><span>Hybrid model</span><strong>α = 0.1</strong></div>
      <div className={styles.fixedField}><span>Seed</span><strong>1</strong></div>
      <div className={styles.fixedField}><span>Image view</span><strong>Evaluation crop only</strong></div>
      <div className={styles.sideDivider} />
      <div className={styles.sideHeading}>Gallery filters</div>
      <label className={styles.fieldLabel}>Status</label>
      <div className={styles.statusOptions}>
        {(["Both", "Rescued", "Regressed"] as Status[]).map(s =>
          <button type="button" key={s} className={`${styles.statusButton} ${status === s ? styles.selected : ""}`} onClick={() => setStatus(s)}>{s}</button>
        )}
      </div>
      <label className={styles.fieldLabel} htmlFor="true-class">True class</label>
      <select id="true-class" className={styles.control} value={trueClass} onChange={e => setTrueClass(e.target.value)}>
        <option value="all">All classes</option>
        {classes.map(([id, name]) => <option value={id} key={id}>{numeric(id)} — {name}</option>)}
      </select>
      <label className={styles.fieldLabel} htmlFor="search">Search prediction or filename</label>
      <input id="search" className={styles.control} placeholder="e.g. leopard or ILSVRC..." value={query} onChange={e => setQuery(e.target.value)} />
      <label className={styles.fieldLabel} htmlFor="sort">Sort</label>
      <select id="sort" className={styles.control} value={sort} onChange={e => setSort(e.target.value as Sort)}>
        {(["ImageNet order", "Class", "Random", "Hybrid confidence high to low", "Hybrid confidence low to high"] as Sort[])
          .map(s => <option value={s} key={s}>{s}</option>)}
      </select>
      <label className={styles.fieldLabel} htmlFor="page-size">Images per page</label>
      <select id="page-size" className={styles.control} value={pageSize} onChange={e => setPageSize(Number(e.target.value))}>
        {[12,24,36,48].map(s => <option key={s} value={s}>{s}</option>)}
      </select>
      <div className={styles.sideNote}>This page uses precomputed predictions and evaluation crops. No model inference runs on Vercel.</div>
    </aside>

    <main className={styles.main}>
      <h1>🔬 JEM rescued vs. regressed images</h1>
      <p className={styles.subtitle}>Rescued: hybrid α=0.1 correct / baseline α=0 wrong. Regressed: baseline correct / hybrid wrong. Images are 224 × 224 evaluation crops.</p>
      {error && <section className={styles.error}><strong>Unable to load Streamlit data</strong><p>{error}</p><p>Upload the generated <code>results.json</code> and <code>crops/</code> to <code>public/jem-analysis/</code>.</p></section>}
      {!data && !error && <div className={styles.loading}>Loading saved predictions…</div>}
      {data && <>
        <div className={styles.metrics}>
          <div className={styles.metric}><span>Rescued</span><strong>{number(rescued)}</strong></div>
          <div className={styles.metric}><span>Regressed</span><strong>{number(regressed)}</strong></div>
          <div className={styles.metric}><span>Net gain</span><strong>{rescued-regressed >= 0 ? "+" : ""}{number(rescued-regressed)}</strong></div>
          <div className={styles.metric}><span>Hybrid alpha</span><strong>0.1</strong></div>
        </div>
        <div className={styles.metadata}>{number(data.meta.nEvaluated)} evaluation images • {number(data.rows.length)} changed in correctness • {number(data.meta.nCrops)} available crops</div>
        <div className={styles.tabs} role="tablist" aria-label="Results">
          {(["Gallery", "Class counts", "Confusions"] as Tab[]).map(t =>
            <button type="button" key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={tab === t ? styles.activeTab : ""}>
              {t === "Gallery" ? "🖼️ " : t === "Class counts" ? "📊 " : "🧩 "}{t}
            </button>
          )}
        </div>
        {tab === "Gallery" && <section>
          <div className={styles.tabTitle}><h2>{number(filtered.length)} matching images</h2>
            <button className={styles.download} onClick={() => downloadCsv("matching_examples.csv", filtered as unknown as Record<string, string | number | null>[])}>Download matching CSV</button>
          </div>
          <div className={styles.pager}>
            <button type="button" disabled={activePage <= 1} onClick={() => setPage(p => Math.max(1, p - 1))}>Previous</button>
            <span>Page {activePage} of {nPages}</span>
            <button type="button" disabled={activePage >= nPages} onClick={() => setPage(p => Math.min(nPages, p + 1))}>Next</button>
          </div>
          <div className={styles.gallery}>
            {pageRows.map(r => <article className={styles.imageCard} key={r.datasetIndex}>
              <div className={r.kind === "rescued" ? styles.rescuedTag : styles.regressedTag}>{r.kind === "rescued" ? "✅ Rescued" : "❌ Regressed"}</div>
              {r.crop ? <img src={`/jem-analysis/${r.crop}`} alt={`${r.trueClassName} validation image`} width={224} height={224} loading="lazy" className={styles.photo} />
                : <div className={styles.noPhoto}>Evaluation crop not uploaded</div>}
              <div className={styles.imageDetails}>
                <strong>True: {r.trueClassName} (#{r.trueLabel})</strong>
                <p>α=0.0: {r.baselineClassName}{confidence(r.confidence0)}</p>
                <p>α=0.1: {r.hybridClassName}{confidence(r.confidence01)}</p>
                <small title={r.filename}>{r.filename}</small>
              </div>
            </article>)}
          </div>
          {filtered.length === 0 && <p>No examples match these filters.</p>}
          <div className={styles.pager}>
            <button type="button" disabled={activePage <= 1} onClick={() => setPage(p => Math.max(1, p - 1))}>Previous</button>
            <span>Page {activePage} of {nPages}</span>
            <button type="button" disabled={activePage >= nPages} onClick={() => setPage(p => Math.min(nPages, p + 1))}>Next</button>
          </div>
        </section>}
        {tab === "Class counts" && <section>
          <div className={styles.tabTitle}><h2>Class rescue minus regression counts</h2>
            <button className={styles.download} onClick={() => downloadCsv("class_counts.csv", classRows as unknown as Record<string, string | number | null>[])}>Download class counts</button>
          </div>
          <p className={styles.subtitle}>Classes with the largest positive and negative rescue–regression balance.</p>
          <div className={styles.legend}><span className={styles.rescueSquare} /> Rescued <span className={styles.regressionSquare} /> Regressed</div>
          <div className={styles.barChart}>{classRows.slice(0, 30).map(r =>
            <div className={styles.barGroup} key={r.id}>
              <span title={r.name}>{r.name}</span>
              <div className={styles.barTracks}>
                <div className={styles.rescueBar} style={{width: `${100 * r.rescued / maxCount}%`}} title={`${r.rescued} rescued`} />
                <div className={styles.regressionBar} style={{width: `${100 * r.regressed / maxCount}%`}} title={`${r.regressed} regressed`} />
              </div><b>{r.net >= 0 ? "+" : ""}{r.net}</b>
            </div>
          )}</div>
          <div className={styles.tableScroll}><table><thead><tr><th>Class ID</th><th>True class</th><th>Rescued</th><th>Regressed</th><th>Net gain</th></tr></thead>
          <tbody>{classRows.map(r => <tr key={r.id}><td>{r.id}</td><td>{r.name}</td><td>{r.rescued}</td><td>{r.regressed}</td><td>{r.net > 0 ? `+${r.net}` : r.net}</td></tr>)}</tbody></table></div>
        </section>}
        {tab === "Confusions" && <section>
          <div className={styles.tabTitle}><h2>Incorrect predictions on changed examples</h2>
            <button className={styles.download} onClick={() => downloadCsv("confusions.csv", confusions.slice(0, 200) as unknown as Record<string, string | number | null>[])}>Download confusion CSV</button>
          </div>
          <div className={styles.tableScroll}><table><thead><tr><th>Outcome</th><th>True class</th><th>Wrong prediction</th><th>Count</th></tr></thead>
          <tbody>{confusions.slice(0, 200).map((r,i) => <tr key={i}><td>{r.kind}</td><td>{r.trueClass}</td><td>{r.wrongClass}</td><td>{r.count}</td></tr>)}</tbody></table></div>
        </section>}
        <details className={styles.diagnostics}><summary>🔧 Diagnostics</summary>
          <p>Source: {data.meta.source}</p><p>Changed correctness rows: {data.rows.length}</p>
          <p>Image crops supplied: {data.meta.nCrops} of {data.rows.length}</p>
        </details>
      </>}
    </main>
  </div>;
}
