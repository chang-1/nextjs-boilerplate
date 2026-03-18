"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), { ssr: false });

type Row = {
  model_family?: string;
  exp?: string;
  latent_dim?: number;
  alpha?: number;
  alpha_str?: string;

  human_corr?: number;
  n_seeds?: number;

  gloss_acc_mean?: number;
  skybox_acc_mean?: number;
  bumpheight_r2_mean?: number;
};

const SIZE_MAP: Record<number, number> = {
  10: 6,
  64: 10,
  100: 15,
  500: 21,
  1000: 27,
  2000: 32, // added
};

function sizeForLatent(d?: number) {
  if (!d) return 10;
  return SIZE_MAP[d] ?? Math.min(34, Math.max(6, Math.round(6 + 10 * Math.log10(d))));
}

// Styles
const JEM_BG_OPACITY = 0.21;
const JEM_FG_OPACITY = 0.95;
const PIX_OPACITY = 0.95;

const JEM_OUTLINE = { width: 0.6, color: "rgba(255,255,255,1.0)" };
const JEM_OUTLINE_BLACK = { width: 0.6, color: "rgba(0,0,0,1.0)" }; // alpha=0.0 only
const PIX_OUTLINE = { width: 1.0, color: "rgba(0,0,0,1.0)" };

const PIX_RED = "rgba(220,20,60,1.0)";
const HIGHLIGHT_BLUE = "rgba(30,144,255,1.0)";

// Plotly qualitative palette (px.colors.qualitative.Plotly)
const PLOTLY_QUAL_PLOTLY = [
  "#636EFA",
  "#EF553B",
  "#00CC96",
  "#AB63FA",
  "#FFA15A",
  "#19D3F3",
  "#FF6692",
  "#B6E880",
  "#FF97FF",
  "#FECB52",
];

const ALPHA_OVERRIDES: Record<string, string> = {
  "0.0": "#636EFA",
};

function buildAlphaColorMap(alphaLevels: string[]) {
  const cmap: Record<string, string> = {};
  alphaLevels.forEach((a, i) => {
    cmap[a] = PLOTLY_QUAL_PLOTLY[i % PLOTLY_QUAL_PLOTLY.length];
  });
  for (const [a, c] of Object.entries(ALPHA_OVERRIDES)) {
    if (a in cmap) cmap[a] = c;
  }
  return cmap;
}

// Put these files in /public
const DATASETS = [
  { id: "w10d28", label: "w10d28", path: "/w10d28.json" },
  { id: "w8d22", label: "w8d22", path: "/w8d22.json" },
] as const;

type DatasetId = (typeof DATASETS)[number]["id"];
type XMetric = "gloss_acc_mean" | "skybox_acc_mean" | "bumpheight_r2_mean";
type JemMode = "all" | "alpha" | "latent";
type PixLatent = "off" | "all" | number;

export default function Page() {
  // dataset selector
  const [datasetId, setDatasetId] = useState<DatasetId>("w10d28");
  const datasetPath = DATASETS.find((d) => d.id === datasetId)!.path;

  // loaded data
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);

    fetch(datasetPath)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch ${datasetPath}: ${r.status}`);
        return r.json();
      })
      .then((json) => {
        if (cancelled) return;
        setRows(json as Row[]);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setErr(String(e));
        setRows([]);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [datasetPath]);

  // controls
  const [xMetric, setXMetric] = useState<XMetric>("bumpheight_r2_mean");

  const [jemMode, setJemMode] = useState<JemMode>("all");
  const [jemAlpha, setJemAlpha] = useState<string>("0.0");
  const [jemLatent, setJemLatent] = useState<number>(64);

  const [pixLatent, setPixLatent] = useState<PixLatent>("all");

  // split + option lists
  const { jemRows, pixRows, alphaLevels, jemLatents, pixLatents } = useMemo(() => {
    const jem = rows.filter((r) => ((r.model_family || "") + "").toLowerCase().includes("jem"));
    const pix = rows.filter((r) => ((r.model_family || "") + "").toLowerCase().includes("pixel"));

    const alphas = Array.from(new Set(jem.map((r) => r.alpha_str).filter(Boolean) as string[])).sort(
      (a, b) => parseFloat(a) - parseFloat(b)
    );

    const jLat = Array.from(new Set(jem.map((r) => r.latent_dim).filter(Boolean) as number[])).sort((a, b) => a - b);
    const pLat = Array.from(new Set(pix.map((r) => r.latent_dim).filter(Boolean) as number[])).sort((a, b) => a - b);

    return { jemRows: jem, pixRows: pix, alphaLevels: alphas, jemLatents: jLat, pixLatents: pLat };
  }, [rows]);

  // keep selections valid when dataset changes
  useEffect(() => {
    if (alphaLevels.length && !alphaLevels.includes(jemAlpha)) setJemAlpha(alphaLevels[0]);
    if (jemLatents.length && !jemLatents.includes(jemLatent)) setJemLatent(jemLatents[0]);
    if (pixLatents.length && typeof pixLatent === "number" && !pixLatents.includes(pixLatent)) setPixLatent("all");
    if (!pixLatents.length && pixLatent !== "off") setPixLatent("off");
  }, [alphaLevels, jemLatents, pixLatents]); // eslint-disable-line react-hooks/exhaustive-deps

  const alphaLabel = (a: string) => {
    if (a === "0.0") return "0.0 (discriminative)";
    if (a === "1.0") return "1.0 (generative)";
    return a;
  };

  const metricLabel = (m: XMetric) => m.replace("_mean", "");

  // title
  const titleParts: string[] = [`dataset=${datasetId}`];
  if (jemMode === "alpha") titleParts.push(`alpha=${alphaLabel(jemAlpha)}`);
  if (jemMode === "latent") titleParts.push(`JEM latent=${jemLatent}`);
  if (pixLatent === "off") titleParts.push("PixelVAE=off");
  if (pixLatent === "all") titleParts.push("PixelVAE=all latents");
  if (typeof pixLatent === "number") titleParts.push(`PixelVAE latent=${pixLatent}`);

  const title = `${metricLabel(xMetric)} vs human correlation — ${titleParts.join(", ")}`;

  // traces
  const traces = useMemo(() => {
    const out: any[] = [];
    if (!jemRows.length && !pixRows.length) return out;

    let jemFg = jemRows;
    if (jemMode === "alpha") jemFg = jemRows.filter((r) => r.alpha_str === jemAlpha);
    if (jemMode === "latent") jemFg = jemRows.filter((r) => r.latent_dim === jemLatent);

    // JEM background
    out.push({
      name: "JEM (bg)",
      type: "scatter",
      mode: "markers",
      x: jemRows.map((r) => (r as any)[xMetric]),
      y: jemRows.map((r) => r.human_corr),
      marker: {
        symbol: "circle",
        color: "rgba(170,170,170,1.0)",
        opacity: JEM_BG_OPACITY,
        size: jemRows.map((r) => sizeForLatent(r.latent_dim)),
        line: JEM_OUTLINE,
      },
      hoverinfo: "skip",
      showlegend: false,
    });

    // JEM foreground
    if (jemMode === "alpha") {
      out.push({
        name: `JEM selected alpha=${alphaLabel(jemAlpha)}`,
        type: "scatter",
        mode: "markers",
        x: jemFg.map((r) => (r as any)[xMetric]),
        y: jemFg.map((r) => r.human_corr),
        marker: {
          symbol: "circle",
          color: HIGHLIGHT_BLUE,
          opacity: JEM_FG_OPACITY,
          size: jemFg.map((r) => sizeForLatent(r.latent_dim)),
          line: jemAlpha === "0.0" ? JEM_OUTLINE_BLACK : JEM_OUTLINE,
        },
        text: jemFg.map((r) => `model=JEM<br>exp=${r.exp}<br>latent=${r.latent_dim}<br>alpha=${r.alpha_str}`),
        hoverinfo: "text+x+y",
      });
    } else {
      const alphasHere = Array.from(new Set(jemFg.map((r) => r.alpha_str).filter(Boolean) as string[])).sort(
        (a, b) => parseFloat(a) - parseFloat(b)
      );
      const cmap = buildAlphaColorMap(alphasHere);

      for (const a of alphasHere) {
        const dd = jemFg.filter((r) => r.alpha_str === a);
        out.push({
          name: `JEM alpha=${alphaLabel(a)}`,
          type: "scatter",
          mode: "markers",
          x: dd.map((r) => (r as any)[xMetric]),
          y: dd.map((r) => r.human_corr),
          marker: {
            symbol: "circle",
            color: cmap[a],
            opacity: JEM_FG_OPACITY,
            size: dd.map((r) => sizeForLatent(r.latent_dim)),
            line: a === "0.0" ? JEM_OUTLINE_BLACK : JEM_OUTLINE,
          },
          text: dd.map((r) => `model=JEM<br>exp=${r.exp}<br>latent=${r.latent_dim}<br>alpha=${r.alpha_str}`),
          hoverinfo: "text+x+y",
        });
      }
    }

    // PixelVAE overlay
    const addPix = (subset: Row[], name: string) => {
      out.push({
        name,
        type: "scatter",
        mode: "markers",
        x: subset.map((r) => (r as any)[xMetric]),
        y: subset.map((r) => r.human_corr),
        marker: {
          symbol: "hexagon",
          color: PIX_RED,
          opacity: PIX_OPACITY,
          size: subset.map((r) => sizeForLatent(r.latent_dim)),
          line: PIX_OUTLINE,
        },
        text: subset.map((r) => `model=PixelVAE<br>exp=${r.exp}<br>latent=${r.latent_dim}`),
        hoverinfo: "text+x+y",
      });
    };

    if (pixLatent === "all") {
      const dims = Array.from(new Set(pixRows.map((r) => r.latent_dim).filter(Boolean) as number[])).sort((a, b) => a - b);
      for (const d of dims) addPix(pixRows.filter((r) => r.latent_dim === d), `PixelVAE latent=${d}`);
    } else if (typeof pixLatent === "number") {
      addPix(pixRows.filter((r) => r.latent_dim === pixLatent), `PixelVAE latent=${pixLatent}`);
    }

    return out;
  }, [jemRows, pixRows, xMetric, jemMode, jemAlpha, jemLatent, pixLatent]);

  return (
    <div style={{ padding: 18, display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <label>
          Dataset:&nbsp;
          <select value={datasetId} onChange={(e) => setDatasetId(e.target.value as DatasetId)}>
            {DATASETS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          X metric:&nbsp;
          <select value={xMetric} onChange={(e) => setXMetric(e.target.value as XMetric)}>
            <option value="gloss_acc_mean">gloss_acc</option>
            <option value="skybox_acc_mean">skybox_acc</option>
            <option value="bumpheight_r2_mean">bumpheight_r2</option>
          </select>
        </label>

        <label>
          JEM mode:&nbsp;
          <select value={jemMode} onChange={(e) => setJemMode(e.target.value as JemMode)}>
            <option value="all">All</option>
            <option value="alpha">Pick alpha</option>
            <option value="latent">Pick latent</option>
          </select>
        </label>

        <label style={{ opacity: jemMode === "alpha" ? 1 : 0.4 }}>
          alpha:&nbsp;
          <select disabled={jemMode !== "alpha"} value={jemAlpha} onChange={(e) => setJemAlpha(e.target.value)}>
            {alphaLevels.map((a) => (
              <option key={a} value={a}>
                {alphaLabel(a)}
              </option>
            ))}
          </select>
        </label>

        <label style={{ opacity: jemMode === "latent" ? 1 : 0.4 }}>
          JEM latent:&nbsp;
          <select
            disabled={jemMode !== "latent"}
            value={jemLatent}
            onChange={(e) => setJemLatent(parseInt(e.target.value))}
          >
            {jemLatents.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>

        <label>
          PixelVAE:&nbsp;
          <select
            value={String(pixLatent)}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "off" || v === "all") setPixLatent(v);
              else setPixLatent(parseInt(v));
            }}
          >
            <option value="off">off</option>
            <option value="all">all latents</option>
            {pixLatents.map((d) => (
              <option key={d} value={String(d)}>
                {d}
              </option>
            ))}
          </select>
        </label>

        {loading && <span style={{ opacity: 0.7 }}>loading…</span>}
        {err && <span style={{ color: "crimson" }}>{err}</span>}
      </div>

      <Plot
        data={traces}
        layout={{
          title,
          xaxis: { title: metricLabel(xMetric) },
          yaxis: { title: "human_corr" },
          margin: { l: 60, r: 20, t: 70, b: 50 },
          legend: { title: { text: "Legend" } },
        }}
        style={{ width: "100%", height: "74vh" }}
        config={{ responsive: true }}
      />
    </div>
  );
}
