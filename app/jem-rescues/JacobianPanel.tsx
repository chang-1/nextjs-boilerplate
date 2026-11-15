"use client";
import { useEffect, useRef, useState } from "react";
import { loadDetail, type Detail, type ImageRow, type RescueData, type Target } from "./data";
import styles from "./explorer.module.css";

function MapImage({ src, label }: { src: string; label: string }) {
  const [failed, setFailed] = useState(false);
  return <figure className={styles.mapFigure}><figcaption>{label}</figcaption>
    {failed ? <p role="alert">Image unavailable. <a href={src} target="_blank" rel="noreferrer">Open file</a></p> :
      <a href={src} target="_blank" rel="noreferrer" title={`Open ${label}`}><img src={src} alt={label} width={224} height={224} onError={() => setFailed(true)} /></a>}
  </figure>;
}
export default function JacobianPanel({ row, data, onClose }: { row: ImageRow; data: RescueData; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [target, setTarget] = useState<Target>("true");
  useEffect(() => {
    const node = dialog.current!;
    const previous = document.activeElement as HTMLElement | null;
    node.showModal();
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { node.close(); document.body.style.overflow = oldOverflow; previous?.focus(); };
  }, []);
  useEffect(() => {
    let active = true;
    loadDetail(row, data).then(d => { if (active) setDetail(d); })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : String(e)); });
    return () => { active = false; };
  }, [row, data, attempt]);
  const maps = detail?.maps[target];
  const stats = detail?.stats[target];
  const name = (id: number) => data.manifest.classNames[id] ?? String(id);
  const metric = (value: number) => Number.isFinite(value) ? value.toPrecision(4) : "Unavailable";
  return <dialog ref={dialog} className={styles.modal} aria-labelledby="jacobian-title" onCancel={onClose}>
    <header className={styles.modalHeader}><div><h2 id="jacobian-title">Jacobian comparison · {row.trueClassName}</h2><p>{row.filename}</p></div>
      <button className={styles.download} onClick={onClose} autoFocus>Close</button></header>
    <p className={styles.subtitle}>Baseline: {row.baselineClassName} ({((row.confidence0 ?? 0) * 100).toFixed(1)}%) · Hybrid: {row.hybridClassName} ({((row.confidence01 ?? 0) * 100).toFixed(1)}%)</p>
    {error ? <div className={styles.error} role="alert">{error} <button className={styles.download} onClick={() => { setError(""); setAttempt(n => n + 1); }}>Retry</button></div> :
      !detail ? <p role="status">Loading Jacobian details…</p> : <>
        <div className={styles.targetControls} aria-label="Attribution target">
          <button className={`${styles.statusButton} ${target === "true" ? styles.selected : ""}`} aria-pressed={target === "true"} onClick={() => setTarget("true")}>Ground-truth logit</button>
          <button className={`${styles.statusButton} ${target === "margin" ? styles.selected : ""}`} aria-pressed={target === "margin"} disabled={!detail.maps.margin} onClick={() => setTarget("margin")}>Ground-truth − competitor margin</button>
        </div>
        {!detail.maps.margin && <p className={styles.subtitle}>Margin maps are available for rescued and regressed images only.</p>}
        {stats && <p className={styles.subtitle}>Target: {name(stats.targetClass)}{stats.competitorClass !== null ? ` minus ${name(stats.competitorClass)}. Both models use this same competitor.` : "."}</p>}
        {maps && <div className={styles.mapGrid}>
          <MapImage key={detail.crop} src={new URL(detail.crop, data.assetBaseUrl).href} label="Evaluation crop" />
          {([['baseline', 'Baseline · α = 0'], ['hybrid', 'Hybrid · α = 0.1'], ['difference', 'Difference magnitude'], ['signedDifference', 'Signed difference']] as const).map(([key, label]) =>
            <MapImage key={maps[key]} src={new URL(maps[key], data.assetBaseUrl).href} label={label} />)}
        </div>}
        {stats && <dl className={styles.mapStats}>{([['Cosine similarity', stats.cosine], ['Baseline norm', stats.baselineNorm], ['Hybrid norm', stats.hybridNorm], ['Difference norm', stats.differenceNorm]] as const).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{metric(value)}</dd></div>)}</dl>}
        <div className={styles.mapExplanation}><p>These maps show decoder-Jacobian projections of latent logit gradients into image space. Magnitude overlays share a per-image 99th-percentile scale; colors are not comparable in absolute strength across images.</p>
          <p>Difference magnitude measures the projected hybrid-minus-baseline gradient. Signed difference shows its channel mean: <strong className={styles.negative}>blue is negative</strong>, <strong className={styles.positive}>red is positive</strong>. It does not indicate whether a prediction improved.</p>
          <p>Cosine similarity compares gradient direction (−1 opposite, 0 orthogonal, +1 aligned). Norms measure gradient magnitude. Click any image to open it at its original size.</p></div>
      </>}
  </dialog>;
}
