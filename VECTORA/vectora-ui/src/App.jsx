import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

// ════════════════════════════════════════════════════════════
// CONSTANTS & HELPERS
// ════════════════════════════════════════════════════════════
const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";
const DIMS = 16;
const COL = {
  cs: "#00d9ff",
  math: "#b388ff",
  food: "#ffb74d",
  sports: "#69f0ae",
  doc: "#a6e3a1",
  default: "#90a4ae",
};
const DIM_COL = Array(4)
  .fill("#00d9ff")
  .concat(Array(4).fill("#b388ff"))
  .concat(Array(4).fill("#ffb74d"))
  .concat(Array(4).fill("#69f0ae"));

const KW = {
  cs: [
    "algorithm",
    "data",
    "tree",
    "graph",
    "array",
    "linked",
    "hash",
    "stack",
    "queue",
    "sort",
    "binary",
    "dynamic",
    "programming",
    "recursion",
    "complexity",
    "pointer",
    "node",
    "search",
    "insert",
    "bfs",
    "dfs",
    "heap",
    "trie",
  ],
  math: [
    "calculus",
    "matrix",
    "probability",
    "theorem",
    "integral",
    "derivative",
    "linear",
    "algebra",
    "equation",
    "function",
    "prime",
    "modular",
    "combinatorics",
    "permutation",
    "eigenvalue",
    "statistics",
    "proof",
  ],
  food: [
    "food",
    "pizza",
    "sushi",
    "ramen",
    "pasta",
    "recipe",
    "cook",
    "eat",
    "restaurant",
    "dish",
    "ingredient",
    "flavor",
    "spice",
    "noodle",
    "bread",
    "croissant",
    "taco",
    "fish",
    "rice",
    "soup",
  ],
  sports: [
    "sport",
    "basketball",
    "football",
    "tennis",
    "chess",
    "swim",
    "game",
    "play",
    "score",
    "team",
    "athlete",
    "competition",
    "match",
    "tournament",
    "olympic",
    "dribble",
    "tackle",
    "serve",
  ],
};

function textToEmbedding(text) {
  const t = text.toLowerCase(),
    ws = t.split(/\s+/);
  const s = { cs: 0, math: 0, food: 0, sports: 0 };
  for (const w of ws) {
    for (const [cat, kws] of Object.entries(KW)) {
      for (const kw of kws) {
        if (w.includes(kw) || kw.startsWith(w)) {
          s[cat] += 0.35;
          break;
        }
      }
    }
  }
  const mx = Math.max(...Object.values(s), 0.01);
  const n = (v) => Math.min((v / mx) * 0.88, 0.94);
  const jitter = () => (Math.random() - 0.5) * 0.04;
  const emb = new Array(16).fill(0.08);
  const fill = (i, score) => {
    if (score < 0.01) return;
    const b = n(score);
    emb[i] = Math.max(0.05, b + jitter());
    emb[i + 1] = Math.max(0.05, b + jitter());
    emb[i + 2] = Math.max(0.05, b * 0.92 + jitter());
    emb[i + 3] = Math.max(0.05, b * 0.87 + jitter());
  };
  fill(0, s.cs);
  fill(4, s.math);
  fill(8, s.food);
  fill(12, s.sports);
  return emb;
}

function pca2D(embs) {
  const n = embs.length,
    d = embs[0].length;
  if (n < 2) return embs.map(() => [0, 0]);
  const mean = new Array(d).fill(0);
  for (const e of embs) for (let i = 0; i < d; i++) mean[i] += e[i] / n;
  const X = embs.map((e) => e.map((v, i) => v - mean[i]));
  function powerIter(X, excl) {
    let v = new Array(d).fill(0).map(() => Math.random() - 0.5);
    if (excl) {
      let dot = v.reduce((s, vi, i) => s + vi * excl[i], 0);
      v = v.map((vi, i) => vi - dot * excl[i]);
    }
    let nrm = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
    v = v.map((vi) => vi / nrm);
    for (let it = 0; it < 200; it++) {
      const Xv = X.map((xi) => xi.reduce((s, xij, j) => s + xij * v[j], 0));
      const nv = new Array(d).fill(0);
      for (let k = 0; k < n; k++)
        for (let j = 0; j < d; j++) nv[j] += X[k][j] * Xv[k];
      if (excl) {
        let dot = nv.reduce((s, vi, i) => s + vi * excl[i], 0);
        for (let i = 0; i < d; i++) nv[i] -= dot * excl[i];
      }
      nrm = Math.sqrt(nv.reduce((s, vi) => s + vi * vi, 0));
      if (nrm < 1e-10) break;
      const prev = v.slice();
      v = nv.map((vi) => vi / nrm);
      if (v.reduce((s, vi, i) => s + (vi - prev[i]) ** 2, 0) < 1e-12) break;
    }
    return v;
  }
  const pc1 = powerIter(X, null),
    pc2 = powerIter(X, pc1);
  return X.map((x) => [
    x.reduce((s, v, i) => s + v * pc1[i], 0),
    x.reduce((s, v, i) => s + v * pc2[i], 0),
  ]);
}
