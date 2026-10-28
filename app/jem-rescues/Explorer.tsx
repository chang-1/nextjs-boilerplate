"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import styles from "./explorer.module.css";

type PredictionRow = {
  id: string;
  y: number;
  p: number[]; // predicted ImageNet category ID per alpha
  c: number[]; // probability of each model's predicted top-1 category
  r: number[]; // ground-truth category rank (1 is top-1)
  q: number[]; // probability assigned to the ground-truth category
  thumb?: string; // optional relative path, e.g. thumbs/00042.webp
};

type ExportFile = {
  schemaVersion: 1;
  meta: {
    dataset: string;
    seed: number;
    alphas: number[];
    nImages: number;
    thumbnailCount: number;
    classNames?: string[];
    createdAt?: string;
    checkpoints?: Record<string, string>;
  };
  rows: PredictionRow[];
};

type Outcome =
  | "rescued"
  | "lost"
  | "any"
  | "majority"
  | "unanimous"
  | "rank-up"
  | "rank-down"
  | "both-correct"
  | "both-wrong";

type Ordering = "gain" | "drop" | "rank" | "index";

const PAGE_SIZE = 24;
const ALPHA_TOLERANCE = 1e-6;
const fmt = (num: number) => num.toLocaleString("en-US");
const pct = (num: number, total: number) => `${total ? ((100 * num) / total).toFixed(2) : "0.00"}%`;
const classId = (x: number) => String(x).padStart(3, "0");
const alphaText = (x: number) => x.toFixed(1);

function labelFor(data: ExportFile, id: number): string {
  return data.meta.classNames?.[id] || `ImageNet class ${classId(id)}`;
}
function getStatus(row: PredictionRow, selected: number): {
  rescued: boolean;
  lost: boolean;
  rankUp: boolean;
  rankDown: boolean;
  bothCorrect: boolean;
  bothWrong: boolean;
  rescueCount: number;
} {
  const baseCorrect = row.p[0] === row.y;
  const candidate = selected < 0 ? row.p.slice(1) : [row.p[selected]];
  const candidateRanks = selected < 0 ? row.r.slice(1) : [row.r[selected]];
  const rescueCount = row.p.slice(1).filter((p) => p === row.y).length;
  const candidateCorrect = candidate.some((p) => p === row.y);
  const allCandidateCorrect = candidate.every((p) => p === row.y);
  const allCandidateWrong = candidate.every((p) => p !== row.y);
  return {
    rescued: !baseCorrect && candidateCorrect,
    lost: baseCorrect && !allCandidateCorrect,
    rankUp: candidateRanks.some((r) => r < row.r[0]),
    rankDown: selected < 0 ? candidateRanks.every((r) => r > row.r[0]) : candidateRanks[0] > row.r[0],
    bothCorrect: baseCorrect && allCandidateCorrect,
    bothWrong: !baseCorrect && allCandidateWrong,
    rescueCount,
  };
}

function getDisplayIndex(row: PredictionRow, selected: number): number {
  if (selected >= 0) return selected;
  // In the collective view, show the hybrid with the highest ground-truth probability.
  let best = 1;
  for (let j = 2; j < row.q.length; j += 1) {
    if (row.q[j] > row.q[best]) best = j;
  }
  return best;
}

function Stat({ label, value, detail, variant = "default" }: {
  label: string; value: string; detail: string; variant?: "default" | "good" | "bad";
}) {
  return <div className={styles.stat}>
    <div className={styles.statLabel}>{label}</div>
    <div className={`${styles.statValue} ${variant === "good" ? styles.good : variant === "bad" ? styles.bad : ""}`}>{value}</div>
    <div className={styles.statDetail}>{detail}</div>
  </div>;
}

export default function Explorer() {
  const [data, setData] = useState<ExportFile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alpha, setAlpha] = useState("0.3");
  const [outcome, setOutcome] = useState<Outcome>("rescued");
  const [sort, setSort] = useState<Ordering>("gain");
  const [search, setSearch] = useState("");
  const [classFilter, setClassFilter] = useState<number | null>(null);
  const [thumbOnly, setThumbOnly] = useState(true);
  const [page, setPage] = useState(1);
  const deferredSearch = useDeferredValue(search.trim().toLowerCase());

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch("/jem-analysis/results.json");
        if (!response.ok) throw new Error(`Results not found (HTTP ${response.status}).`);
        const json: unknown = await response.json();
        if (!json || typeof json !== "object" || !("schemaVersion" in json) || !("rows" in json) || !("meta" in json)) {
          throw new Error("Invalid results.json structure.");
        }
        const parsed = json as ExportFile;
        if (parsed.schemaVersion !== 1 || !Array.isArray(parsed.rows) || !Array.isArray(parsed.meta.alphas)) {
          throw new Error("Unsupported results.json schema. Expected version 1.");
        }
        if (!parsed.meta.alphas.includes(0) || parsed.rows.some((r) => !Array.isArray(r.p) || r.p.length !== parsed.meta.alphas.length)) {
          throw new Error("Invalid alpha ordering or prediction arrays in results.json.");
        }
        if (active) {
          setData(parsed);
          setAlpha(alphaText(parsed.meta.alphas.find((a) => Math.abs(a - 0.3) < ALPHA_TOLERANCE) ?? parsed.meta.alphas[1]));
          setThumbOnly(parsed.meta.thumbnailCount > 0);
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, []);

  const selectedIdx = useMemo(() => {
    if (!data || alpha === "any") return -1;
    const idx = data.meta.alphas.findIndex((a) => Math.abs(a - Number(alpha)) < ALPHA_TOLERANCE);
    return idx > 0 ? idx : -1;
  }, [alpha, data]);

  const modelStats = useMemo(() => {
    if (!data) return [];
    return data.meta.alphas.map((a, i) => {
      const correct = data.rows.reduce((n, r) => n + Number(r.p[i] === r.y), 0);
      const top5 = data.rows.reduce((n, r) => n + Number(r.r[i] <= 5), 0);
      const rescued = i === 0 ? 0 : data.rows.reduce((n, r) => n + Number(r.p[0] !== r.y && r.p[i] === r.y), 0);
      const lost = i === 0 ? 0 : data.rows.reduce((n, r) => n + Number(r.p[0] === r.y && r.p[i] !== r.y), 0);
      return { alpha: a, index: i, correct, top5, rescued, lost, net: rescued - lost };
    });
  }, [data]);

  const selectedCounts = useMemo(() => {
    if (!data) return { rescued: 0, lost: 0, majority: 0, unanimous: 0 };
    let rescued = 0, lost = 0, majority = 0, unanimous = 0;
    for (const row of data.rows) {
      const status = getStatus(row, selectedIdx);
      if (status.rescued) rescued += 1;
      if (status.lost) lost += 1;
      if (row.p[0] !== row.y && status.rescueCount >= 3) majority += 1;
      if (row.p[0] !== row.y && status.rescueCount === data.meta.alphas.length - 1) unanimous += 1;
    }
    return { rescued, lost, majority, unanimous };
  }, [data, selectedIdx]);

  const byClass = useMemo(() => {
    if (!data) return [];
    const map = new Map<number, { id: number; n: number; rescued: number; lost: number; baseline: number }>();
    for (const row of data.rows) {
      const record = map.get(row.y) || { id: row.y, n: 0, rescued: 0, lost: 0, baseline: 0 };
      const status = getStatus(row, selectedIdx);
      record.n += 1;
      record.rescued += Number(status.rescued);
      record.lost += Number(status.lost);
      record.baseline += Number(row.p[0] === row.y);
      map.set(row.y, record);
    }
    return [...map.values()].map((v) => ({ ...v, net: v.rescued - v.lost })).sort((a, b) => b.net - a.net || b.rescued - a.rescued);
  }, [data, selectedIdx]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const list = data.rows.filter((row) => {
      if (thumbOnly && !row.thumb) return false;
      if (classFilter !== null && row.y !== classFilter) return false;
      if (deferredSearch && !row.id.toLowerCase().includes(deferredSearch) && !labelFor(data, row.y).toLowerCase().includes(deferredSearch) && !classId(row.y).includes(deferredSearch)) return false;
      const s = getStatus(row, selectedIdx);
      switch (outcome) {
        case "rescued": return s.rescued;
        case "lost": return s.lost;
        case "majority": return row.p[0] !== row.y && s.rescueCount >= 3;
        case "unanimous": return row.p[0] !== row.y && s.rescueCount === data.meta.alphas.length - 1;
        case "rank-up": return s.rankUp;
        case "rank-down": return s.rankDown;
        case "both-correct": return s.bothCorrect;
        case "both-wrong": return s.bothWrong;
        default: return true;
      }
    });
    if (sort !== "index") list.sort((a, b) => {
      const i = getDisplayIndex(a, selectedIdx), j = getDisplayIndex(b, selectedIdx);
      if (sort === "rank") return (b.r[0] - b.r[j]) - (a.r[0] - a.r[i]);
      const gainA = a.q[i] - a.q[0], gainB = b.q[j] - b.q[0];
      return sort === "drop" ? gainA - gainB : gainB - gainA;
    });
    return list;
  }, [data, classFilter, deferredSearch, thumbOnly, selectedIdx, outcome, sort]);

  useEffect(() => { setPage(1); }, [alpha, outcome, sort, search, classFilter, thumbOnly]);
  const maxPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const activePage = Math.min(page, maxPages);
  const visible = filtered.slice((activePage - 1) * PAGE_SIZE, activePage * PAGE_SIZE);
  const baseline = modelStats[0];
  const hybrid = selectedIdx < 0 ? null : modelStats[selectedIdx];
  const alphaSelectedLabel = selectedIdx < 0 ? "Any of the 5 hybrids" : `α = ${alphaText(Number(alpha))}`;
  const datasetSize = data?.rows.length ?? 0;

  return <main className={styles.shell}>
    <div className={styles.container}>
      <nav className={styles.nav}><a href="/" className={styles.navBack}>← JEM research plots</a><span className={styles.navRight}>JEM / IMAGENET-1K / SEED 1</span></nav>
      <header className={styles.header}>
        <div className={styles.eyebrow}>REPRESENTATIONAL ANALYSIS / CLASSIFICATION BEHAVIOR</div>
        <h1>Which images do hybrid <em>JEMs rescue?</em></h1>
        <p>Compare a discriminative baseline (α = 0.0) with five hybrid objectives. Explore newly correct predictions, regressions, and category-level effects on identical ImageNet validation examples.</p>
      </header>

      {loading && <div className={styles.notice}>Loading ImageNet prediction results…</div>}
      {error && <div className={styles.notice} role="alert"><strong>No prediction export found.</strong><p>{error}</p><p>Run <code>scripts/export_jem_classification.py</code> on OSCAR and copy <code>results.json</code> into <code>public/jem-analysis/</code>. The site does not run your PyTorch checkpoints.</p></div>}
      {data && <>
        <div className={styles.metadata}><span className={styles.liveDot}/> {fmt(datasetSize)} validation images <span>·</span> {data.meta.alphas.length} models <span>·</span> Seed {data.meta.seed} <span>·</span> {fmt(data.meta.thumbnailCount)} exported thumbnails</div>
        <section className={styles.panel} aria-label="Model performance comparison">
          <div className={styles.sectionTop}><div><span className={styles.sectionKicker}>01 / MODEL PERFORMANCE</span><h2>Classification across α</h2></div><span className={styles.sectionHint}>Correct predictions / {fmt(datasetSize)}</span></div>
          <div className={styles.performanceGrid}>
            <div className={styles.chartPanel}>
              <div className={styles.smallTitle}>Top-1 accuracy by objective weight</div>
              <div className={styles.barList}>
                {modelStats.map((m) => <button key={m.alpha} className={`${styles.barRow} ${(selectedIdx === m.index || (m.index === 0 && selectedIdx === -1)) ? styles.barRowActive : ""}`} onClick={() => { if (m.index > 0) setAlpha(alphaText(m.alpha)); }} title={m.index === 0 ? "Baseline" : `Inspect alpha ${m.alpha}`}>
                  <span className={styles.barLabel}>α={alphaText(m.alpha)}</span>
                  <span className={styles.barTrack}><span className={`${styles.barFill} ${m.index === 0 ? styles.baselineFill : ""}`} style={{ width: `${(m.correct / datasetSize) * 100}%` }} /></span>
                  <span className={styles.barValue}>{pct(m.correct, datasetSize)}</span>
                </button>)}
              </div>
            </div>
            <div className={styles.chartPanel}>
              <div className={styles.smallTitle}>Rescues versus losses <span className={styles.muted}>(relative to α=0)</span></div>
              <div className={styles.tradeoffList}>
                {modelStats.slice(1).map((m) => <button key={m.alpha} className={styles.tradeoffRow} onClick={() => setAlpha(alphaText(m.alpha))}>
                  <span className={styles.barLabel}>α={alphaText(m.alpha)}</span>
                  <span className={styles.tradeoffTrack}>
                    <span className={styles.rescueFill} style={{ width: `${100 * m.rescued / Math.max(1, ...modelStats.map((x) => x.rescued + x.lost)) }%` }}/>
                    <span className={styles.lossFill} style={{ width: `${100 * m.lost / Math.max(1, ...modelStats.map((x) => x.rescued + x.lost)) }%` }}/>
                  </span>
                  <span className={`${styles.netValue} ${m.net >= 0 ? styles.good : styles.bad}`}>{m.net >= 0 ? "+" : ""}{fmt(m.net)}</span>
                </button>)}
              </div>
              <div className={styles.chartLegend}><span><i className={styles.legendGood}/> Rescued</span><span><i className={styles.legendBad}/> Lost</span><span>Right: net change</span></div>
            </div>
          </div>
        </section>

        <section className={styles.statsRow} aria-label="Selected comparison metrics">
          <Stat label="BASELINE TOP-1" value={pct(baseline?.correct ?? 0, datasetSize)} detail={`α=0.0 · Top-5 ${pct(baseline?.top5 ?? 0, datasetSize)}`} />
          <Stat label={selectedIdx < 0 ? "ANY HYBRID RESCUES" : "HYBRID TOP-1"} value={selectedIdx < 0 ? fmt(selectedCounts.rescued) : pct(hybrid?.correct ?? 0, datasetSize)} detail={selectedIdx < 0 ? "Baseline errors fixed by ≥1 hybrid" : `${alphaSelectedLabel} · Top-5 ${pct(hybrid?.top5 ?? 0, datasetSize)}`} />
          <Stat label="NEWLY CORRECT" value={`+${fmt(selectedCounts.rescued)}`} detail="Baseline wrong → hybrid correct" variant="good" />
          <Stat label="REGRESSIONS" value={`−${fmt(selectedCounts.lost)}`} detail={selectedIdx < 0 ? "Baseline correct → ≥1 hybrid wrong" : "Baseline correct → hybrid wrong"} variant="bad" />
          <Stat label={selectedIdx < 0 ? "ALL FIVE RESCUE" : "NET CLASSIFICATIONS"} value={selectedIdx < 0 ? fmt(selectedCounts.unanimous) : `${selectedCounts.rescued - selectedCounts.lost >= 0 ? "+" : ""}${fmt(selectedCounts.rescued - selectedCounts.lost)}`} detail={selectedIdx < 0 ? `${fmt(selectedCounts.majority)} rescued by ≥3 hybrids` : "Rescued − lost"} variant={selectedIdx < 0 || selectedCounts.rescued >= selectedCounts.lost ? "good" : "bad"}/>
        </section>

        <section id="top" className={styles.panel} aria-label="Image explorer">
          <div className={styles.sectionTop}><div><span className={styles.sectionKicker}>02 / IMAGE-LEVEL EXPLORER</span><h2>Inspect individual predictions</h2></div><span className={styles.sectionHint}>{fmt(filtered.length)} matching examples</span></div>
          <div className={styles.filters}>
            <label className={styles.field}><span>Compared model</span><select value={alpha} onChange={(e) => setAlpha(e.target.value)}><option value="any">Any hybrid (union)</option>{data.meta.alphas.slice(1).map((a) => <option key={a} value={alphaText(a)}>α = {alphaText(a)}</option>)}</select></label>
            <label className={styles.field}><span>Outcome</span><select value={outcome} onChange={(e) => setOutcome(e.target.value as Outcome)}><option value="rescued">Rescued (baseline wrong)</option><option value="lost">Lost (baseline correct)</option><option value="majority">Rescued by 3+ hybrids</option><option value="unanimous">Rescued by all 5</option><option value="rank-up">Ground-truth rank improved</option><option value="rank-down">Ground-truth rank worsened</option><option value="both-correct">Both correct</option><option value="both-wrong">Both wrong</option><option value="any">All images</option></select></label>
            <label className={styles.field}><span>Sort by</span><select value={sort} onChange={(e) => setSort(e.target.value as Ordering)}><option value="gain">Largest true-class probability gain</option><option value="drop">Largest true-class probability drop</option><option value="rank">Largest rank improvement</option><option value="index">Dataset order</option></select></label>
            <label className={styles.field}><span>Image or class search</span><input type="search" placeholder="e.g. tabby, 281, val_0001" value={search} onChange={(e) => setSearch(e.target.value)} /></label>
          </div>
          <div className={styles.filterFooter}>
            <label className={styles.checkbox}><input type="checkbox" checked={thumbOnly} onChange={(e) => setThumbOnly(e.target.checked)} /> Only show images with exported thumbnails</label>
            {classFilter !== null && <button className={styles.clearTag} onClick={() => setClassFilter(null)}>Class: {labelFor(data, classFilter)} ×</button>}
          </div>
          <div className={styles.imageGrid}>
            {visible.map((row) => {
              const h = getDisplayIndex(row, selectedIdx);
              const originalCorrect = row.p[0] === row.y;
              const hybridCorrect = row.p[h] === row.y;
              const delta = row.q[h] - row.q[0];
              return <article className={styles.imageCard} key={row.id}>
                <div className={styles.imageFrame}>
                  {row.thumb ? <Image src={`/jem-analysis/${row.thumb}`} alt={`ImageNet image ${row.id}: ${labelFor(data, row.y)}`} fill sizes="(max-width: 640px) 90vw, (max-width: 1050px) 45vw, 22vw" unoptimized className={styles.image} /> : <div className={styles.noImage}><span>Image not exported</span><small>Class ID {row.y}</small></div>}
                  <span className={`${styles.imageStatus} ${!originalCorrect && hybridCorrect ? styles.rescueChip : originalCorrect && !hybridCorrect ? styles.lossChip : styles.neutralChip}`}>{!originalCorrect && hybridCorrect ? "RESCUED" : originalCorrect && !hybridCorrect ? "LOST" : hybridCorrect ? "CORRECT" : "INCORRECT"}</span>
                </div>
                <div className={styles.cardBody}>
                  <div className={styles.cardId} title={row.id}>{row.id}</div>
                  <div className={styles.groundTruth}>GT: <strong>{labelFor(data, row.y)}</strong></div>
                  <div className={styles.predictionRow}><span>α = 0.0</span><strong className={originalCorrect ? styles.good : styles.bad}>{labelFor(data, row.p[0])}</strong></div>
                  <div className={styles.predictionRow}><span>α = {alphaText(data.meta.alphas[h])}</span><strong className={hybridCorrect ? styles.good : styles.bad}>{labelFor(data, row.p[h])}</strong></div>
                  <div className={styles.cardFooter}><span>GT rank {row.r[0]} → {row.r[h]}</span><span className={delta >= 0 ? styles.good : styles.bad}>{delta >= 0 ? "+" : ""}{(100 * delta).toFixed(1)} pp GT prob.</span></div>
                </div>
              </article>;
            })}
          </div>
          {filtered.length === 0 && <div className={styles.empty}>No images match these filters. Try unchecking the thumbnail restriction or choosing another outcome.</div>}
          <div className={styles.pagination}><span>Showing {filtered.length ? fmt((activePage - 1) * PAGE_SIZE + 1) : 0}–{fmt(Math.min(activePage * PAGE_SIZE, filtered.length))} of {fmt(filtered.length)}</span><div><button disabled={activePage <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>Previous</button><span>Page {activePage} / {maxPages}</span><button disabled={activePage >= maxPages} onClick={() => setPage((p) => Math.min(maxPages, p + 1))}>Next</button></div></div>
        </section>

        <section className={styles.panel} aria-label="ImageNet class breakdown">
          <div className={styles.sectionTop}><div><span className={styles.sectionKicker}>03 / CLASS-LEVEL ANALYSIS</span><h2>Where do hybrids help?</h2></div><span className={styles.sectionHint}>Click a class to inspect its images</span></div>
          <div className={styles.classTableScroll}><table className={styles.classTable}><thead><tr><th>ImageNet class</th><th>Images</th><th>Rescued</th><th>Lost</th><th>Net gain</th></tr></thead><tbody>{byClass.slice(0, 30).map((item) => <tr key={item.id} onClick={() => { setClassFilter(item.id); setOutcome("rescued"); document.getElementById("top")?.scrollIntoView({ behavior: "smooth" }); }}><td><span className={styles.classNumber}>{classId(item.id)}</span> {labelFor(data, item.id)}</td><td>{item.n}</td><td className={styles.good}>+{item.rescued}</td><td className={styles.bad}>−{item.lost}</td><td className={item.net >= 0 ? styles.good : styles.bad}>{item.net > 0 ? "+" : ""}{item.net}</td></tr>)}</tbody></table></div>
        </section>
        <div className={styles.footnote}>Definitions: <strong>rescued</strong> = α=0 wrong, hybrid correct; <strong>lost</strong> = α=0 correct, hybrid wrong. In “any hybrid” mode, rescues and losses are set unions and must not be interpreted as one model’s net gain. Ground-truth probability refers to the softmax probability of the labeled class; it is not a calibrated confidence measure. ImageNet images may have redistribution restrictions—publish thumbnails only when authorized.</div>
      </>}
    </div>
  </main>;
}
