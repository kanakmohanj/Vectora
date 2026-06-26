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

// ════════════════════════════════════════════════════════════
// TYPEWRITER COMPONENT (Prevents full app re-renders)
// ════════════════════════════════════════════════════════════
const Typewriter = ({ text, speed = 18 }) => {
  const [displayed, setDisplayed] = useState("");
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    let i = 0;
    setDisplayed("");
    setIsTyping(true);
    const timer = setInterval(() => {
      if (i >= text.length) {
        clearInterval(timer);
        setIsTyping(false);
        return;
      }
      const chunk = text.slice(i, i + 3);
      setDisplayed((prev) => prev + chunk);
      i += 3;
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);

  return <span className={isTyping ? "typing" : ""}>{displayed}</span>;
};

// ════════════════════════════════════════════════════════════
// MAIN APP COMPONENT
// ════════════════════════════════════════════════════════════
export default function App() {
  // UI State
  const [activeTab, setActiveTab] = useState("search");

  // DB State
  const [allItems, setAllItems] = useState([]);
  const [geminiStatus, setGeminiStatus] = useState({
    up: false,
    checking: true,
  });
  const [hnswLayers, setHnswLayers] = useState(null);

  // Search State
  const [selAlgo, setSelAlgo] = useState("hnsw");
  const [metric, setMetric] = useState("cosine");
  const [topK, setTopK] = useState(5);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchLatency, setSearchLatency] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [currentEmb, setCurrentEmb] = useState(new Array(16).fill(0));

  // Doc/RAG State
  const [docs, setDocs] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [isInsertingDoc, setIsInsertingDoc] = useState(false);
  const [docInsertStatus, setDocInsertStatus] = useState("");

  // Refs for Canvas Animation (Mutable State)
  const canvasRef = useRef(null);
  const vecCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const pcaState = useRef({
    points: [],
    bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
  });
  const searchState = useRef({ hitIds: new Set(), queryPt: null });
  const mouseState = useRef({ hoverItem: null, x: 0, y: 0 });

  // ─── INITIALIZATION ───
  const loadItems = useCallback(async () => {
    try {
      const r = await fetch(API + "/items");
      const items = await r.json();
      setAllItems(items);

      if (items.length >= 2) {
        const coords = pca2D(items.map((v) => v.embedding));
        const pts = items.map((item, i) => ({
          x: coords[i][0],
          y: coords[i][1],
          item,
        }));
        let x0 = Infinity,
          x1 = -Infinity,
          y0 = Infinity,
          y1 = -Infinity;
        for (const p of pts) {
          x0 = Math.min(x0, p.x);
          x1 = Math.max(x1, p.x);
          y0 = Math.min(y0, p.y);
          y1 = Math.max(y1, p.y);
        }
        const px = (x1 - x0) * 0.18 || 0.1,
          py = (y1 - y0) * 0.18 || 0.1;
        pcaState.current = {
          points: pts,
          bounds: {
            minX: x0 - px,
            maxX: x1 + px,
            minY: y0 - py,
            maxY: y1 + py,
          },
        };
      } else {
        pcaState.current.points = [];
      }
    } catch (e) {
      console.error("Failed to load items");
    }
  }, []);

  const loadHNSW = useCallback(async () => {
    try {
      const r = await fetch(API + "/hnsw-info");
      setHnswLayers(await r.json());
    } catch (e) {}
  }, []);

  const loadDocList = useCallback(async () => {
    try {
      const r = await fetch(API + "/doc/list");
      setDocs(await r.json());
    } catch (e) {}
  }, []);

  const checkGeminiStatus = useCallback(async () => {
    try {
      const r = await fetch(API + "/status");
      const d = await r.json();
      // Adjust this based on your actual C++ struct naming
      setGeminiStatus({
        up: d.geminiAvailable || false,
        data: d,
        checking: false,
      });
    } catch (e) {
      setGeminiStatus({ up: false, checking: false });
    }
  }, []);

  useEffect(() => {
    loadItems().then(loadHNSW);
    checkGeminiStatus();
  }, [loadItems, loadHNSW, checkGeminiStatus]);

  // ─── CANVAS RENDER LOOP ───
  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    let pulse = 0;

    const resize = () => {
      const r = cvs.parentElement.getBoundingClientRect();
      cvs.width = r.width;
      cvs.height = r.height;
    };
    window.addEventListener("resize", resize);
    resize();

    const w2c = (wx, wy, bounds) => {
      const P = 70,
        W = cvs.width,
        H = cvs.height;
      const rx = bounds.maxX - bounds.minX || 1,
        ry = bounds.maxY - bounds.minY || 1;
      return [
        P + ((wx - bounds.minX) / rx) * (W - 2 * P),
        H - P - ((wy - bounds.minY) / ry) * (H - 2 * P),
      ];
    };

    const drawFrame = () => {
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.fillStyle = "#f5f6fa";
      ctx.fillRect(0, 0, cvs.width, cvs.height);

      // Grid
      ctx.strokeStyle = "#dcdde1";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 8; i++) {
        const tx = 70 + (i / 8) * (cvs.width - 140),
          ty = 70 + (i / 8) * (cvs.height - 140);
        ctx.beginPath();
        ctx.moveTo(tx, 70);
        ctx.lineTo(tx, cvs.height - 70);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(70, ty);
        ctx.lineTo(cvs.width - 70, ty);
        ctx.stroke();
      }

      ctx.fillStyle = "#718093";
      ctx.font = "14px Fira Code,monospace";
      ctx.fillText("PC₁ →", cvs.width / 2 - 40, cvs.height - 18);
      ctx.save();
      ctx.translate(18, cvs.height / 2 + 50);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("PC₂ →", 0, 0);
      ctx.restore();
      ctx.fillStyle = "#718093";
      ctx.font = "15px Fira Code,monospace";
      ctx.fillText("2D PCA Projection  ·  Semantic Space", 80, 28);

      const { points, bounds } = pcaState.current;
      const { hitIds, queryPt } = searchState.current;
      const { hoverItem } = mouseState.current;

      // Draw Query Links
      if (queryPt && hitIds.size > 0) {
        const [qx, qy] = w2c(queryPt.x, queryPt.y, bounds);
        for (const pt of points) {
          if (!hitIds.has(pt.item.id)) continue;
          const [px, py] = w2c(pt.x, pt.y, bounds);
          ctx.strokeStyle = "rgba(108,99,255,0.18)";
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(px, py);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Draw Nodes
      for (const pt of points) {
        const [cx, cy] = w2c(pt.x, pt.y, bounds);
        const col = COL[pt.item.category] || COL.default;
        const isHit = hitIds.has(pt.item.id);
        const r = isHit ? 10 : 7;

        if (isHit) {
          const pr = r + 7 + Math.sin(pulse) * 3.5;
          ctx.beginPath();
          ctx.arc(cx, cy, pr, 0, 2 * Math.PI);
          ctx.strokeStyle = col + "55";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
        grd.addColorStop(0, col + (isHit ? "bb" : "88"));
        grd.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(cx, cy, r * 3, 0, 2 * Math.PI);
        ctx.fillStyle = grd;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fillStyle = col;
        ctx.fill();

        if (hoverItem && hoverItem.id === pt.item.id) {
          ctx.beginPath();
          ctx.arc(cx, cy, r + 5, 0, 2 * Math.PI);
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // Draw Query Point
      if (queryPt) {
        const [qx, qy] = w2c(queryPt.x, queryPt.y, bounds);
        ctx.save();
        ctx.translate(qx, qy);
        ctx.shadowColor = "#333";
        ctx.shadowBlur = 18;
        ctx.beginPath();
        for (let i = 0; i < 10; i++) {
          const a = (i * Math.PI) / 5 - Math.PI / 2,
            rr = i % 2 === 0 ? 13 : 5;
          if (i === 0) ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
          else ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
        }
        ctx.closePath();
        ctx.fillStyle = "#fff";
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.restore();
        ctx.fillStyle = "#2f3640";
        ctx.font = "13px Fira Code,monospace";
        ctx.fillText("query", qx + 16, qy + 4);
      }

      if (points.length === 0) {
        ctx.fillStyle = "#1a1a38";
        ctx.font = "13px Fira Code,monospace";
        ctx.textAlign = "center";
        ctx.fillText("Connecting to VectorDB…", cvs.width / 2, cvs.height / 2);
        ctx.textAlign = "left";
      }

      pulse += 0.05;
      animationRef.current = requestAnimationFrame(drawFrame);
    };

    animationRef.current = requestAnimationFrame(drawFrame);
    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  const handleCanvasMouseMove = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left,
      my = e.clientY - rect.top;

    let best = 18;
    let hovered = null;
    const { points, bounds } = pcaState.current;

    const w2c = (wx, wy) => {
      const P = 70,
        W = canvasRef.current.width,
        H = canvasRef.current.height;
      const rx = bounds.maxX - bounds.minX || 1,
        ry = bounds.maxY - bounds.minY || 1;
      return [
        P + ((wx - bounds.minX) / rx) * (W - 2 * P),
        H - P - ((wy - bounds.minY) / ry) * (H - 2 * P),
      ];
    };

    for (const pt of points) {
      const [cx, cy] = w2c(pt.x, pt.y);
      const d = Math.hypot(mx - cx, my - cy);
      if (d < best) {
        best = d;
        hovered = pt.item;
      }
    }

    mouseState.current = { hoverItem: hovered, x: e.clientX, y: e.clientY };

    // 🔥 Direct DOM manipulation for zero-lag tooltip 🔥
    const tip = document.getElementById("tip");
    if (tip) {
      if (hovered) {
        tip.style.display = "block";
        tip.style.left = e.clientX + 14 + "px";
        tip.style.top = e.clientY - 8 + "px";
        const col = COL[hovered.category] || COL.default;
        tip.innerHTML = `<span style="color:${col}; font-weight: 600;">[${hovered.category.toUpperCase()}]</span><br/>${hovered.metadata}`;
      } else {
        tip.style.display = "none";
      }
    }
  };

  const handleCanvasMouseLeave = () => {
    mouseState.current.hoverItem = null;
    const tip = document.getElementById("tip");
    if (tip) tip.style.display = "none";
  };

  // ─── VECTOR CHART RENDER LOOP ───
  useEffect(() => {
    const vc = vecCanvasRef.current;
    if (!vc || activeTab !== "search") return;
    const W = vc.parentElement.clientWidth;
    vc.width = W;
    const vx = vc.getContext("2d");
    vx.clearRect(0, 0, W, 76);
    vx.fillStyle = "#f5f6fa";
    vx.fillRect(0, 0, W, 76);
    const bw = (W - 4) / DIMS;
    for (let i = 0; i < DIMS; i++) {
      const h = currentEmb[i] * 58,
        x = 2 + i * bw,
        col = DIM_COL[i];
      vx.shadowColor = col;
      vx.shadowBlur = 5;
      vx.fillStyle = col + "aa";
      vx.fillRect(x + 1, 63 - h, bw - 2, h);
    }
    vx.shadowBlur = 0;
    vx.font = "11px monospace";
    vx.textAlign = "center";
    [
      ["CS", 0],
      ["MATH", 4],
      ["FOOD", 8],
      ["SPORT", 12],
    ].forEach(([lbl, gi], i) => {
      vx.fillStyle = Object.values(COL)[i] + "77";
      vx.fillText(lbl, 2 + (gi + 1.5) * bw, 74);
    });
    vx.textAlign = "left";
  }, [currentEmb, activeTab]);

  // ─── ACTIONS ───
  const runSearch = async () => {
    const text = searchQuery.trim();
    if (!text) return;
    const emb = textToEmbedding(text);
    setCurrentEmb(emb);

    try {
      const r = await fetch(
        `${API}/search?v=${emb.join(",")}&k=${topK}&metric=${metric}&algo=${selAlgo}`,
      );
      const data = await r.json();
      const results = data.results || [];
      setSearchResults(results);
      setSearchLatency(data.latencyUs || 0);

      const newHitIds = new Set(results.map((r) => r.id));
      let queryPt = null;
      if (results.length > 0) {
        let sx = 0,
          sy = 0,
          sw = 0;
        for (let i = 0; i < Math.min(3, results.length); i++) {
          const pt = pcaState.current.points.find(
            (p) => p.item.id === results[i].id,
          );
          if (pt) {
            const w = 1 / (i + 1);
            sx += pt.x * w;
            sy += pt.y * w;
            sw += w;
          }
        }
        if (sw > 0)
          queryPt = {
            x: sx / sw + (Math.random() - 0.5) * 0.015,
            y: sy / sw + (Math.random() - 0.5) * 0.015,
          };
      }
      searchState.current = { hitIds: newHitIds, queryPt };
    } catch (e) {
      alert("Cannot reach server — is it running on :8080?");
    }
  };

  const runBenchmark = async () => {
    const text = searchQuery || "binary tree";
    const algos = ["hnsw", "kdtree", "bruteforce"];
    let resultsText = `🚀 BENCHMARK RESULTS 🚀\nQuery: "${text}"\nMetric: ${metric}\n\n`;

    try {
      console.log("Starting benchmark...");

      // 🔥 FIX: Convert the text to a vector embedding FIRST
      const emb = textToEmbedding(text);
      const vString = emb.join(",");

      for (const algo of algos) {
        // 🔥 FIX: Use "?v=" to pass the vector string, matching runSearch exactly
        const res = await fetch(
          `${API}/search?v=${vString}&k=${topK}&metric=${metric}&algo=${algo}`,
        );

        if (!res.ok) throw new Error(`${algo} search failed`);

        const data = await res.json();

        const formatAlgo = algo
          .toUpperCase()
          .replace("BRUTEFORCE", "BRUTE FORCE");
        resultsText += `• ${formatAlgo}: ${data.latencyUs} µs\n`;
      }

      alert(resultsText);
    } catch (e) {
      console.error("Benchmark Error:", e);
      alert("Benchmark failed to run. Make sure your C++ backend is running!");
    }
  };

  const addVector = async (e) => {
    e.preventDefault();
    const meta = e.target.meta.value.trim();
    const cat = e.target.cat.value;
    if (!meta) return;
    const emb = textToEmbedding(meta + " " + cat);
    try {
      await fetch(API + "/insert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metadata: meta, category: cat, embedding: emb }),
      });
      e.target.meta.value = "";
      loadItems().then(loadHNSW);
    } catch (_) {}
  };

  const deleteDemoItem = async (id) => {
    try {
      await fetch(`${API}/delete/${id}`, { method: "DELETE" });
      setSearchResults((prev) => prev.filter((r) => r.id !== id));
      searchState.current.hitIds.delete(id);
      loadItems().then(loadHNSW);
    } catch (_) {}
  };

  const insertDocument = async (e) => {
    e.preventDefault();
    const title = e.target.docTitle.value.trim();
    const text = e.target.docText.value.trim();
    if (!title || !text)
      return setDocInsertStatus("⚠ Need both a title and text.");

    setIsInsertingDoc(true);
    setDocInsertStatus("Calling Gemini API for embeddings...");

    try {
      const r = await fetch(API + "/doc/insert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text }),
      });
      const d = await r.json();
      if (d.error) {
        setDocInsertStatus(`✗ Error: ${d.error}`);
      } else {
        setDocInsertStatus(
          `✓ Inserted ${d.chunks} chunk(s) · ${d.dims}D embeddings`,
        );
        e.target.reset();

        // Push a fake 16D vector so it renders on our map
        const emb16 = textToEmbedding(title + " " + text);
        await fetch(API + "/insert", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            metadata: title,
            category: "doc",
            embedding: emb16,
          }),
        });

        loadItems().then(loadHNSW);
        loadDocList();
        checkGeminiStatus();
      }
    } catch (_) {
      setDocInsertStatus("✗ Server error");
    }
    setIsInsertingDoc(false);
  };

  const deleteDocument = async (docId, docTitle) => {
    try {
      // 1. Delete from the real Document DB (RAG)
      await fetch(`${API}/doc/delete/${docId}`, { method: "DELETE" });

      // 2. Find the corresponding fake 16D vector in the visualizer and delete it
      const fakePoint = allItems.find(
        (item) => item.category === "doc" && item.metadata === docTitle,
      );
      if (fakePoint) {
        await fetch(`${API}/delete/${fakePoint.id}`, { method: "DELETE" });
      }

      // 3. Refresh all UI states
      loadDocList();
      loadItems().then(loadHNSW); // This removes it from the graph!
      checkGeminiStatus();
    } catch (e) {
      console.error("Failed to delete document", e);
    }
  };

  const askAI = async (e) => {
    e.preventDefault();
    const question = e.target.ragQuestion.value.trim();
    const k = e.target.ragK.value;
    if (!question) return;

    setIsAsking(true);
    const newChat = [...chatHistory, { role: "user", text: question }];
    setChatHistory(newChat);
    e.target.ragQuestion.value = "";

    // Background Map Update
    fetch(API + "/doc/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: parseInt(k) }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.contexts?.length > 0) {
          const hitIds = new Set();
          let sx = 0,
            sy = 0,
            sw = 0;
          data.contexts.forEach((ctx, i) => {
            const pt = pcaState.current.points.find(
              (p) =>
                p.item.category === "doc" &&
                ctx.title.startsWith(p.item.metadata),
            );
            if (pt) {
              hitIds.add(pt.item.id);
              const w = 1 / (i + 1);
              sx += pt.x * w;
              sy += pt.y * w;
              sw += w;
            }
          });
          const queryPt =
            sw > 0
              ? {
                  x: sx / sw + (Math.random() - 0.5) * 0.015,
                  y: sy / sw + (Math.random() - 0.5) * 0.015,
                }
              : null;
          searchState.current = { hitIds, queryPt };
        }
      })
      .catch(() => {});

    try {
      const r = await fetch(API + "/doc/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: parseInt(k) }),
      });
      const d = await r.json();

      if (d.error) {
        setChatHistory([...newChat, { role: "error", text: d.error }]);
      } else {
        setChatHistory([
          ...newChat,
          { role: "ai", text: d.answer, model: d.model, contexts: d.contexts },
        ]);
      }
    } catch (err) {
      setChatHistory([
        ...newChat,
        { role: "error", text: "Server error — is the backend running?" },
      ]);
    }
    setIsAsking(false);
  };

  // ─── RENDERERS ───
  const hoverItem = mouseState.current.hoverItem;

  return (
    <>
      <header>
        <h1>VECTORA</h1>
        {/* Make the branding badges dynamic based on the selected algorithm state */}
        <span className={`badge ${selAlgo === "hnsw" ? "hl" : ""}`}>HNSW</span>
        <span className={`badge ${selAlgo === "kdtree" ? "hl" : ""}`}>
          KD-TREE
        </span>
        <span className={`badge ${selAlgo === "bruteforce" ? "hl" : ""}`}>
          BRUTE FORCE
        </span>

        <span className={`badge ${geminiStatus.up ? "ok" : "err"}`}>
          GEMINI {geminiStatus.up ? "✓" : "✗"}
        </span>
        <span id="statsLabel">
          {allItems.length} vectors · {DIMS} dims
        </span>
        <div
          style={{
            position: "absolute",
            top: "20px",
            right: "30px",
            fontSize: "13px",
            fontWeight: "500",
            letterSpacing: "0.5px",
            color: "var(--text-muted, #888)",
            zIndex: 1000,
            userSelect: "none",
          }}
        >
          made with <span style={{ color: "#ff4757" }}>❤️</span> by Kanak
        </div>
      </header>

      <div className="layout">
        {/* ══ LEFT PANEL ══ */}
        <div className="left-panel">
          <div>
            <div className="sec">Query (Demo Vectors)</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runSearch()}
                placeholder="binary tree, sushi…"
              />
              <button className="btn-p" onClick={runSearch}>
                ⚡ SEARCH
              </button>
            </div>
          </div>
          <div>
            <div className="sec">Algorithm</div>
            <div className="algo-row">
              {["hnsw", "kdtree", "bruteforce"].map((a) => (
                <div
                  key={a}
                  className={`algo-btn ${selAlgo === a ? "on" : ""}`}
                  onClick={() => setSelAlgo(a)}
                >
                  {a.toUpperCase().replace("BRUTEFORCE", "BRUTE")}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="sec">Distance Metric</div>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="cosine">Cosine Similarity</option>
              <option value="euclidean">Euclidean Distance</option>
              <option value="manhattan">Manhattan Distance</option>
            </select>
          </div>
          <div>
            <div className="sec">
              Top-K: <span>{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />
          </div>
          <div>
            <div className="sec">Category Legend</div>
            <div className="legend">
              <div className="leg-row">
                <div
                  className="dot"
                  style={{ background: COL.cs, boxShadow: `0 0 5px ${COL.cs}` }}
                ></div>
                CS / Algorithms
              </div>
              <div className="leg-row">
                <div
                  className="dot"
                  style={{
                    background: COL.math,
                    boxShadow: `0 0 5px ${COL.math}`,
                  }}
                ></div>
                Mathematics
              </div>
              <div className="leg-row">
                <div
                  className="dot"
                  style={{
                    background: COL.food,
                    boxShadow: `0 0 5px ${COL.food}`,
                  }}
                ></div>
                Food & Cooking
              </div>
              <div className="leg-row">
                <div
                  className="dot"
                  style={{
                    background: COL.sports,
                    boxShadow: `0 0 5px ${COL.sports}`,
                  }}
                ></div>
                Sports & Games
              </div>
              <div className="leg-row">
                <div
                  className="dot"
                  style={{
                    background: COL.doc,
                    boxShadow: `0 0 5px ${COL.doc}`,
                  }}
                ></div>
                Documents (RAG)
              </div>
            </div>
          </div>
          <div>
            <div className="sec">Insert Demo Vector</div>
            <form
              onSubmit={addVector}
              style={{ display: "flex", flexDirection: "column", gap: 6 }}
            >
              <input
                type="text"
                name="meta"
                placeholder="Description…"
                required
              />
              <select name="cat">
                <option value="cs">CS / Algorithms</option>
                <option value="math">Mathematics</option>
                <option value="food">Food & Cooking</option>
                <option value="sports">Sports & Games</option>
              </select>
              <button type="submit" className="btn-s">
                + INSERT
              </button>
            </form>
          </div>
          <div>
            <div className="sec">Benchmark</div>
            <button className="btn-s" onClick={runBenchmark}>
              ▶ COMPARE ALL ALGOS
            </button>
          </div>
        </div>

        {/* ══ CENTER: SCATTER PLOT ══ */}
        <div className="center-panel">
          <canvas
            ref={canvasRef}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
          />
        </div>

        {/* ══ RIGHT PANEL ══ */}
        <div className="right-panel">
          <div className="tabs">
            <div
              className={`tab ${activeTab === "search" ? "on" : ""}`}
              onClick={() => setActiveTab("search")}
            >
              SEARCH
            </div>
            <div
              className={`tab ${activeTab === "docs" ? "on" : ""}`}
              onClick={() => {
                setActiveTab("docs");
                loadDocList();
              }}
            >
              DOCUMENTS
            </div>
            <div
              className={`tab ${activeTab === "rag" ? "on" : ""}`}
              onClick={() => setActiveTab("rag")}
            >
              ASK AI
            </div>
          </div>

          {/* TAB: SEARCH */}
          <div className={`tab-content ${activeTab === "search" ? "on" : ""}`}>
            <div>
              <div className="sec">Search Latency</div>
              <div className="lat-big">
                {searchLatency !== null
                  ? searchLatency < 1000
                    ? `${searchLatency} μs`
                    : `${(searchLatency / 1000).toFixed(2)} ms`
                  : "—"}
              </div>
              <div className="lat-sub">
                {searchLatency !== null
                  ? `${selAlgo.toUpperCase()} · ${metric} · k=${topK}`
                  : "No query yet"}
              </div>
            </div>
            <div>
              <div className="sec">Top Matches</div>
              <div className="results">
                {searchResults.length === 0 ? (
                  <div style={{ color: "var(--muted)", fontSize: 11 }}>
                    Run a search to see results…
                  </div>
                ) : (
                  searchResults.map((r, i) => {
                    const col = COL[r.category] || COL.default;
                    return (
                      <div key={r.id} className="rcard">
                        <div className="rrank">#{i + 1} NEAREST</div>
                        <div className="rmeta">{r.metadata}</div>
                        <div className="rfoot">
                          <span
                            className="rcat"
                            style={{
                              background: `${col}18`,
                              color: col,
                              border: `1px solid ${col}44`,
                            }}
                          >
                            {r.category.toUpperCase()}
                          </span>
                          <span className="rdist">
                            dist: {r.distance.toFixed(5)}
                          </span>
                          <button
                            className="del"
                            onClick={() => deleteDemoItem(r.id)}
                          >
                            ✕
                          </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
            <div>
              <div className="sec">Query Embedding (16D)</div>
              <canvas
                ref={vecCanvasRef}
                height="76"
                style={{
                  background: "var(--bg)",
                  borderRadius: 5,
                  width: "100%",
                  display: "block",
                }}
              ></canvas>
            </div>

            {benchmarks && (
              <div>
                <div className="sec">Algorithm Comparison</div>
                <div className="bench">
                  {[
                    {
                      lbl: "Brute Force",
                      us: benchmarks.bruteforceUs,
                      col: "#f38ba8",
                    },
                    { lbl: "KD-Tree", us: benchmarks.kdtreeUs, col: "#89dceb" },
                    { lbl: "HNSW", us: benchmarks.hnswUs, col: "#b388ff" },
                  ].map((b) => {
                    const mx = Math.max(
                      benchmarks.bruteforceUs,
                      benchmarks.kdtreeUs,
                      benchmarks.hnswUs,
                      1,
                    );
                    const pct = Math.max((b.us / mx) * 100, 2);
                    return (
                      <div className="brow" key={b.lbl}>
                        <div className="blabel">
                          <span style={{ color: b.col }}>{b.lbl}</span>
                          <span style={{ color: "var(--muted)" }}>
                            {b.us < 1000
                              ? `${b.us} μs`
                              : `${(b.us / 1000).toFixed(2)} ms`}
                          </span>
                        </div>
                        <div className="btrack">
                          <div
                            className="bfill"
                            style={{ width: `${pct}%`, background: b.col }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div>
              <div className="sec">HNSW Graph Layers</div>
              <div className="layers">
                {!hnswLayers ? (
                  <div style={{ color: "var(--muted)", fontSize: 11 }}>
                    Loading…
                  </div>
                ) : (
                  hnswLayers.nodesPerLayer.map((cnt, lyr) => {
                    const maxN = hnswLayers.nodesPerLayer[0] || 1;
                    const pct = Math.max((cnt / maxN) * 100, 2);
                    const edg = hnswLayers.edgesPerLayer[lyr] || 0;
                    return (
                      <div className="lrow" key={lyr}>
                        <div className="lnum">L{lyr}</div>
                        <div className="ltrack">
                          <div
                            className="lfill"
                            style={{ width: `${pct}%` }}
                          ></div>
                        </div>
                        <div className="lcount">
                          {cnt}n · {edg}e
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* TAB: DOCUMENTS */}
          <div className={`tab-content ${activeTab === "docs" ? "on" : ""}`}>
            <div>
              <div className="sec">Gemini API Status</div>
              <div
                className={`gemini-status ${geminiStatus.up ? "ok" : "err"}`}
              >
                {geminiStatus.checking ? (
                  "Checking..."
                ) : geminiStatus.up ? (
                  <>
                    <span style={{ color: "var(--green)" }}>
                      ● API Accessible
                    </span>
                    <br />
                    Embed Model:{" "}
                    <span style={{ color: "var(--accent)" }}>
                      {geminiStatus.data.embedModel || "gemini-text-embedding"}
                    </span>
                    <br />
                    Gen Model:{" "}
                    <span style={{ color: "var(--accent)" }}>
                      {geminiStatus.data.genModel || "gemini-1.5-flash"}
                    </span>
                    <br />
                    Doc Chunks:{" "}
                    <span style={{ color: "var(--text)" }}>
                      {geminiStatus.data.docCount}
                    </span>
                  </>
                ) : (
                  <>
                    <span style={{ color: "var(--red)" }}>
                      ● API Offline or Invalid Key
                    </span>
                    <br />
                    <br />
                    <span style={{ color: "var(--muted)" }}>
                      Ensure your .env file contains a valid Google API Key and
                      your C++ backend is running on port 8080.
                    </span>
                  </>
                )}
              </div>
            </div>
            <div>
              <div className="sec">Insert Document</div>
              <form
                onSubmit={insertDocument}
                style={{ display: "flex", flexDirection: "column", gap: 8 }}
              >
                <input
                  type="text"
                  name="docTitle"
                  placeholder="Document title / topic…"
                  required
                />
                <textarea
                  name="docText"
                  placeholder="Paste your notes or textbook excerpt here... Long text is automatically chunked and embedded via Gemini."
                  required
                ></textarea>
                <button
                  type="submit"
                  className="btn-g"
                  disabled={isInsertingDoc}
                >
                  {isInsertingDoc ? "Embedding…" : "⚡ EMBED & INSERT"}
                </button>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>
                  {docInsertStatus}
                </div>
              </form>
            </div>
            <div>
              <div className="sec">Stored Documents ({docs.length})</div>
              <div className="doc-list">
                {docs.length === 0 ? (
                  <div style={{ color: "var(--muted)", fontSize: 11 }}>
                    No documents yet.
                  </div>
                ) : (
                  docs.map((d) => (
                    <div className="dcard" key={d.id}>
                      <div className="dcard-title">{d.title}</div>
                      <div className="dcard-preview">{d.preview}</div>
                      <div className="dcard-foot">
                        <span className="dcard-words">{d.words} words</span>
                        <button
                          className="del"
                          onClick={() => deleteDocument(d.id, d.title)}
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* TAB: ASK AI (RAG) */}
          <div className={`tab-content ${activeTab === "rag" ? "on" : ""}`}>
            <div>
              <div className="sec">Ask a Question</div>
              <form
                onSubmit={askAI}
                style={{ display: "flex", flexDirection: "column", gap: 8 }}
              >
                <textarea
                  name="ragQuestion"
                  rows="3"
                  placeholder="What is dynamic programming?"
                  required
                ></textarea>
                <div style={{ display: "flex", gap: 6 }}>
                  <select name="ragK" style={{ width: "auto", flexShrink: 0 }}>
                    <option value="2">Top 2</option>
                    <option value="3" defaultValue>
                      Top 3
                    </option>
                    <option value="5">Top 5</option>
                  </select>
                  <button
                    type="submit"
                    className="btn-g"
                    style={{ flex: 1 }}
                    disabled={isAsking}
                  >
                    {isAsking ? "Thinking…" : "🤖 ASK AI"}
                  </button>
                </div>
              </form>
            </div>
            <div>
              <div className="sec">Conversation</div>
              <div className="chat-history">
                {chatHistory.length === 0 ? (
                  <div style={{ color: "var(--muted)", fontSize: 11 }}>
                    Ask a question about your inserted documents…
                  </div>
                ) : (
                  chatHistory.map((msg, i) =>
                    msg.role === "user" ? (
                      <div key={i} className="chat-q">
                        {msg.text}
                      </div>
                    ) : msg.role === "error" ? (
                      <div key={i} className="chat-a">
                        <div className="chat-a-label">ERROR</div>
                        <div
                          className="chat-a-text"
                          style={{ color: "var(--red)" }}
                        >
                          {msg.text}
                        </div>
                      </div>
                    ) : (
                      <div key={i} className="chat-a">
                        <div className="chat-a-label">
                          🤖 {msg.model || "gemini"}
                        </div>
                        <div className="chat-a-text">
                          <Typewriter text={msg.text} />
                        </div>
                        {msg.contexts && msg.contexts.length > 0 && (
                          <div className="chat-ctx">
                            <div className="chat-ctx-label">
                              RETRIEVED CONTEXT ({msg.contexts.length} chunks)
                            </div>
                            {msg.contexts.map((c, ci) => (
                              <details key={ci} style={{ display: "inline" }}>
                                <summary className="ctx-chip">
                                  #{ci + 1} {c.title} · {c.distance.toFixed(3)}
                                </summary>
                                <div className="ctx-expand">{c.text}</div>
                              </details>
                            ))}
                          </div>
                        )}
                      </div>
                    ),
                  )
                )}
                {isAsking && (
                  <div className="thinking">
                    <div className="spinner"></div>Retrieving context &
                    generating answer…
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* TOOLTIP */}
      <div id="tip"></div>
    </>
  );
}
