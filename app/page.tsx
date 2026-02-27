"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import data from "./data/agg.json";

const Plot = dynamic(() => import("react-plotly.js").then(m => m.default), { ssr: false });

type Row = {
  model_family?: string;
  exp?: string;
  latent_dim?: number;
  alpha?: number;
  alpha_str?: string;
  human_corr?: number;
  gloss_acc_mean?: number;
  skybox_acc_mean?: number;
  bumpheight_r2_mean?: number;
};

const SIZE_MAP: Record<number, number> = { 10: 4, 64: 8, 100: 13, 500: 19, 1000: 26 };

export default function Page() {
  const rows = data as Row[];

  const [xMetric, setXMetric] = useState<"gloss_acc_mean" | "skybox_acc_mean" | "bumpheight_r2_mean">(
    "bumpheight_r2_mean"
  );

  const [jemMode, setJemMode] = useState<"all" | "alpha" | "latent">("all");
  const [jemAlpha, setJemAlpha] = useState<string>("0.0");
  const [jemLatent, setJemLatent] = useState<number>(64);

  const [pixLatent, setPixLatent] = useState<"off" | "all" | number>("all");

  const { jemRows, pixRows, alphaLevels, jemLatents, pixLatents } = useMemo(() => {
    const jem = rows.filter((r) => (r.model_family || "").toLowerCase().includes("jem"));
    const pix = rows.filter((r) => (r.model_family || "").toLowerCase().includes("pixel"));

    const alphas = Array.from(new Set(jem.map((r) => r.alpha_str).filter(Boolean) as string[])).sort(
      (a, b) => parseFloat(a) - parseFloat(b)
    );
    const jLat = Array.from(new Set(jem.map((r) => r.latent_dim).filter(Boolean) as number[])).sort(
      (a, b) => a - b
    );
    const pLat = Array.from(new Set(pix.map((r) => r.latent_dim).filter(Boolean) as number[])).sort(
      (a, b) => a - b
    );

    return { jemRows: jem, pixRows: pix, alphaLevels: alphas, jemLatents: jLat, pixLatents: pLat };
  }, [rows]);

  const alphaLabel = (a: string) => {
    if (a === "0.0") return "0.0 (discriminative)";
    if (a === "1.0") return "1.0 (generative)";
    return a;
  };

  const titleParts: string[] = [];
  if (jemMode === "alpha") titleParts.push(`alpha=${alphaLabel(jemAlpha)}`);
  if (jemMode === "latent") titleParts.push(`JEM latent=${jemLatent}`);
  if (pixLatent === "off") titleParts.push("PixelVAE=off");
  if (pixLatent === "all") titleParts.push("PixelVAE=all latents");
  if (typeof pixLatent === "number") titleParts.push(`PixelVAE latent=${pixLatent}`);

  const title = `${xMetric} vs human_corr` + (titleParts.length ? ` — ${titleParts.join(", ")}` : "");

  // ----- build traces -----
  const traces = useMemo(() => {
    const out: any[] = [];

    // Filter JEM foreground according to mode
    let jemFg = jemRows;
    if (jemMode === "alpha") jemFg = jemRows.filter((r) => r.alpha_str === jemAlpha);
    if (jemMode === "latent") jemFg = jemRows.filter((r) => r.latent_dim === jemLatent);

    // Background = all JEM dim gray
    out.push({
      name: "JEM (bg)",
      type: "scatter",
      mode: "markers",
      x: jemRows.map((r) => (r as any)[xMetric]),
      y: jemRows.map((r) => r.human_corr),
      marker: {
        symbol: "circle",
        color: "rgba(170,170,170,1.0)",
        opacity: 0.21,
        size: jemRows.map((r) => SIZE_MAP[r.latent_dim || 0] ?? 10),
        line: { width: 0.6, color: "rgba(255,255,255,1.0)" },
      },
      hoverinfo: "skip",
      showlegend: false,
    });

    // JEM foreground (color by alpha when in "all" or "latent"; single color in alpha mode)
    if (jemMode === "alpha") {
      out.push({
        name: `JEM selected alpha=${alphaLabel(jemAlpha)}`,
        type: "scatter",
        mode: "markers",
        x: jemFg.map((r) => (r as any)[xMetric]),
        y: jemFg.map((r) => r.human_corr),
        marker: {
          symbol: "circle",
          color: "rgba(30,144,255,1.0)",
          opacity: 0.95,
          size: jemFg.map((r) => SIZE_MAP[r.latent_dim || 0] ?? 10),
          line: { width: 0.6, color: "rgba(255,255,255,1.0)" },
        },
        text: jemFg.map((r) => `exp=${r.exp}<br>latent=${r.latent_dim}<br>alpha=${r.alpha_str}`),
        hoverinfo: "text+x+y",
      });
    } else {
      // one trace per alpha (so legend toggles)
      const alphas = Array.from(new Set(jemFg.map((r) => r.alpha_str).filter(Boolean) as string[])).sort(
        (a, b) => parseFloat(a) - parseFloat(b)
      );
      const colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
      ];
      const cmap: Record<string, string> = {};
      alphas.forEach((a, i) => (cmap[a] = colors[i % colors.length]));

      for (const a of alphas) {
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
            opacity: 0.95,
            size: dd.map((r) => SIZE_MAP[r.latent_dim || 0] ?? 10),
            line: { width: 0.6, color: "rgba(255,255,255,1.0)" },
          },
          text: dd.map((r) => `exp=${r.exp}<br>latent=${r.latent_dim}<br>alpha=${r.alpha_str}`),
          hoverinfo: "text+x+y",
        });
      }
    }

    // PixelVAE overlay: red circles with black outline
    const addPix = (subset: Row[], name: string) => {
      out.push({
        name,
        type: "scatter",
        mode: "markers",
        x: subset.map((r) => (r as any)[xMetric]),
        y: subset.map((r) => r.human_corr),
        marker: {
          symbol: "circle",
          color: "rgba(220,20,60,1.0)",
          opacity: 0.95,
          size: subset.map((r) => SIZE_MAP[r.latent_dim || 0] ?? 10),
          line: { width: 1.0, color: "rgba(0,0,0,1.0)" },
        },
        text: subset.map((r) => `exp=${r.exp}<br>latent=${r.latent_dim}<br>alpha=${r.alpha_str}`),
        hoverinfo: "text+x+y",
      });
    };

    if (pixLatent === "all") {
      // one trace per latent for easy toggling
      const dims = Array.from(new Set(pixRows.map((r) => r.latent_dim).filter(Boolean) as number[])).sort((a, b) => a - b);
      for (const d of dims) {
        addPix(pixRows.filter((r) => r.latent_dim === d), `PixelVAE latent=${d}`);
      }
    } else if (typeof pixLatent === "number") {
      addPix(pixRows.filter((r) => r.latent_dim === pixLatent), `PixelVAE latent=${pixLatent}`);
    }

    return out;
  }, [jemRows, pixRows, xMetric, jemMode, jemAlpha, jemLatent, pixLatent]);

  return (
    <div style={{ padding: 18, display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <label>
          X metric:&nbsp;
          <select value={xMetric} onChange={(e) => setXMetric(e.target.value as any)}>
            <option value="gloss_acc_mean">gloss_acc_mean</option>
            <option value="skybox_acc_mean">skybox_acc_mean</option>
            <option value="bumpheight_r2_mean">bumpheight_r2_mean</option>
          </select>
        </label>

        <label>
          JEM mode:&nbsp;
          <select value={jemMode} onChange={(e) => setJemMode(e.target.value as any)}>
            <option value="all">All</option>
            <option value="alpha">Pick alpha</option>
            <option value="latent">Pick latent</option>
          </select>
        </label>

        <label style={{ opacity: jemMode === "alpha" ? 1 : 0.4 }}>
          alpha:&nbsp;
          <select
            disabled={jemMode !== "alpha"}
            value={jemAlpha}
            onChange={(e) => setJemAlpha(e.target.value)}
          >
            {alphaLevels.map((a) => (
              <option key={a} value={a}>{alphaLabel(a)}</option>
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
              <option key={d} value={d}>{d}</option>
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
              <option key={d} value={String(d)}>{d}</option>
            ))}
          </select>
        </label>
      </div>

      <Plot
        data={traces}
        layout={{
          title,
          xaxis: { title: xMetric },
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
