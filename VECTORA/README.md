# Vectora — Vector Database Engineered from Scratch in C++

<div align="center">

![C++](https://img.shields.io/badge/C++17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_API-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

**A production-grade vector database built from the ground up in C++**  
HNSW · KD-Tree · Brute Force · AVX2 SIMD · RAG Pipeline · Real-Time 2D Visualization

🌐 **Live Demo:** [vectora.vercel.app](https://vectora.vercel.app) &nbsp;|&nbsp; 📦 **Backend API:** [vectora-engine.onrender.com](https://vectora-engine.onrender.com)

</div>

---

## What Is Vectora?

Vectora is a **high-performance, in-memory vector database** engineered entirely from scratch in C++ — no wrappers, no third-party engines. It implements the same indexing algorithms used by Pinecone, Weaviate, and Chroma at their core, while adding hardware-level optimizations (AVX2 SIMD), a multithreaded HTTP server, and a full end-to-end RAG pipeline powered by the Google Gemini API.

Paired with a React frontend, Vectora provides real-time 2D visualization of high-dimensional semantic space, interactive algorithm benchmarking, and a chat interface to query your own documents using retrieval-augmented generation.

> **Built to understand how vector databases actually work** — from graph construction to distance math to semantic search.

---

## Table of Contents

- [Feature Overview](#-feature-overview)
- [Architecture](#-architecture)
- [Algorithm Deep Dive](#-algorithm-deep-dive)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started (Local)](#-getting-started-local-development)
- [Docker Deployment](#-docker-deployment)
- [REST API Reference](#-rest-api-reference)
- [Using the Application](#-using-the-application)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Feature Overview

### ⚡ Engine & Hardware Optimization

| Feature | Detail |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade ANN), KD-Tree (exact, axis-partitioned), Brute Force (baseline) |
| **3 Distance Metrics** | Cosine Similarity, Euclidean Distance, Manhattan Distance |
| **AVX2 SIMD Acceleration** | Distance math vectorized using AVX2 + FMA hardware instructions — processes 8 floats per CPU cycle, achieving **sub-50 µs** query latency |
| **Multithreaded Architecture** | `cpp-httplib` backend with thread-safe state management handles concurrent embedding and query requests without locks on the hot path |
| **HNSW Graph** | M=16, ef\_construction=200; layer-based small-world graph mirroring Pinecone's index design |

### 🧠 Real-Time Semantic Visualization

| Feature | Detail |
|---|---|
| **2D PCA Projection** | Custom dimensionality reduction maps 768D Gemini embeddings down to a 2D semantic plane |
| **HTML5 Canvas** | Interactive scatter plot with smooth animations; matched points glow, search paths are drawn live |
| **Cluster Visualization** | 20 pre-loaded 16D demo vectors across 4 semantic categories (CS, Math, Food, Sports) — watch distinct clusters form |
| **Embedding Inspector** | Real-time bar chart showing the active query's embedding distribution across all 16 dimensions |
| **Algorithm Search Paths** | Toggle HNSW's layer-by-layer traversal paths directly on the scatter plot |

### 📚 End-to-End RAG Pipeline

| Feature | Detail |
|---|---|
| **Gemini Embeddings** | Text chunked and embedded on-the-fly via `text-embedding-004` (768D) |
| **Automatic Chunking** | Long documents split into overlapping 250-word chunks; each chunk indexed separately |
| **Contextual Retrieval** | Upload any text, plot it in semantic space, run natural-language queries against it |
| **Streaming LLM Answers** | Gemini generates responses with typewriter streaming; context chips show exactly which chunks were used |
| **Full CRUD** | Insert, list, search, and delete documents and demo vectors via REST API |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Vite)                       │
│   ┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│   │  Search Tab   │  │  Documents Tab   │  │   Ask AI Tab     │    │
│   │  (Demo 16D)   │  │  (Real 768D)     │  │  (RAG Pipeline)  │    │
│   └───────┬───────┘  └────────┬─────────┘  └────────┬─────────┘    │
│           │   HTML5 Canvas PCA Scatter Plot          │              │
└───────────┼──────────────────────────────────────────┼─────────────┘
            │  REST API (JSON over HTTP)                │
┌───────────▼──────────────────────────────────────────▼─────────────┐
│                      C++ Backend (cpp-httplib)                      │
│                                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│   │  BruteForce │  │   KD-Tree   │  │         HNSW            │   │
│   │   O(N·d)    │  │  O(log N)   │  │  O(log N) — ANN Graph   │   │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘   │
│            AVX2 SIMD  ·  Cosine / Euclidean / Manhattan            │
│                                                                     │
│   ┌────────────────────────┐   ┌──────────────────────────────┐   │
│   │       VectorDB         │   │        DocumentDB            │   │
│   │  (16D demo vectors)    │   │  (768D Gemini embeddings)    │   │
│   └────────────────────────┘   └──────────────────────────────┘   │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                    GeminiClient                            │   │
│   │    POST /embedContent  ·  POST /generateContent            │   │
│   │    text-embedding-004 (768D)  ·  gemini-1.5-flash (gen)   │   │
│   └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │ Containerized via Docker · Deployed on Render
```

---

## 🔬 Algorithm Deep Dive

### HNSW — Hierarchical Navigable Small World

The same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.

**Construction:** Nodes are inserted into a multilayer graph. Each node is randomly assigned a max layer (exponential distribution). Layer 0 contains all nodes with dense connections; higher layers are progressively sparser and act as "highways."

```
Layer 2:  A ────────────── E
Layer 1:  A ──── B ──── D ─ E
Layer 0:  A ─ B ─ C ─ D ─ E ─ F ─ G
```

**Insert:** Greedy descent from the top layer → at your assigned max layer, beam search (ef\_construction=200) → connect bidirectionally to the M=16 nearest neighbors at each layer down to 0.

**Search:** Same greedy descent. At layer 0, expand to ef nearest candidates using a max-heap priority queue.

**Complexity:** O(log N) — the upper layers provide logarithmic zoom-in to the right neighborhood.

---

### KD-Tree — K-Dimensional Tree

Binary space partitioning. Each internal node splits space along one dimension (cycling: x → y → z → …). Search prunes entire subtrees when the closest possible point in that subtree cannot beat the current best — the "ball within hyperslab" check.

**Weakness:** Suffers from the curse of dimensionality. Works well at ≤20D; degrades to near brute-force at 768D as no subtrees get pruned.

---

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned bounding boxes. In high dimensions, almost all volume sits near the surface of the hypersphere — hyperslab checks never prune anything. HNSW's graph-based traversal sidesteps this entirely by navigating the learned similarity structure of the data.

| Algorithm | Complexity | Exact? | Works at 768D? |
|---|---|---|---|
| Brute Force | O(N·d) | ✅ Yes | ✅ Slow but correct |
| KD-Tree | O(log N) best-case | ✅ Yes | ❌ Degrades to O(N) |
| HNSW | O(log N) | ⚡ Approximate | ✅ Designed for this |

---

### AVX2 SIMD Distance Math

Standard dot product (cosine similarity) iterates one float at a time. Vectora's AVX2 implementation loads 8 floats per cycle into 256-bit registers and performs the multiply-accumulate in a single instruction:

```cpp
// 8 floats per cycle instead of 1
__m256 va = _mm256_loadu_ps(&a[i]);
__m256 vb = _mm256_loadu_ps(&b[i]);
sum = _mm256_fmadd_ps(va, vb, sum);  // fused multiply-add
```

Result: **sub-50 µs** per query on modern hardware, even at 768 dimensions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Core Engine** | C++17, custom HNSW / KD-Tree / Brute Force |
| **SIMD Math** | AVX2 + FMA intrinsics (`-mavx2 -mfma`) |
| **HTTP Server** | `cpp-httplib` (single-header, multithreaded) |
| **TLS / HTTPS** | OpenSSL (required for Gemini API calls) |
| **AI Integration** | Google Gemini API (`text-embedding-004`, `gemini-1.5-flash`) |
| **Frontend** | React.js, Vite, HTML5 Canvas API, CSS Modules |
| **Containerization** | Docker (production-ready Dockerfile included) |
| **Backend Hosting** | Render (Linux edge server) |
| **Frontend Hosting** | Vercel (global CDN) |

---

## 📁 Project Structure

```
Vectora/
├── vectora-backend/
│   ├── main.cpp            ← Full C++ engine: HNSW, KD-Tree, BruteForce, REST API, RAG
│   ├── httplib.h           ← Single-header HTTP server (cpp-httplib)
│   ├── Dockerfile          ← Production container (Ubuntu + GCC + OpenSSL)
│   ├── .dockerignore
│   ├── .gitignore
│   └── .env                ← GEMINI_API_KEY lives here (never committed)
│
├── vectora-ui/
│   ├── src/
│   │   ├── App.jsx         ← Entire frontend: search, documents, RAG chat, PCA scatter plot
│   │   ├── App.css         ← All styles
│   │   ├── main.jsx        ← React entry point
│   │   └── index.css       ← Global resets
│   ├── public/
│   ├── index.html
│   ├── vite.config.js
│   ├── eslint.config.js
│   ├── package.json
│   └── .env                ← VITE_API_BASE_URL lives here
│
└── README.md
```

**Backend class layout (`main.cpp`):**

```
BruteForce      O(N·d)      Exact baseline; always correct
KDTree          O(log N)    Exact; axis-aligned binary partitioning
HNSW            O(log N)    Approximate; multilayer small-world graph

VectorDB        Unified interface over all 3 (16D demo vectors)
DocumentDB      HNSW-only index for 768D Gemini embeddings
GeminiClient    HTTPS client → /embedContent + /generateContent
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites

- **C++ Compiler:** GCC ≥ 11 or Clang ≥ 13 with C++17 and AVX2 support
- **Node.js:** v18 or higher (for the Vite frontend)
- **OpenSSL:** Development libraries (`libssl-dev` on Ubuntu, `brew install openssl` on macOS)
- **Google Gemini API Key:** Free tier available at [aistudio.google.com](https://aistudio.google.com)

---

### 1. Clone the Repository

```bash
git clone https://github.com/kanakmohanj/Vectora.git
cd Vectora
```

---

### 2. Set Up the C++ Backend

Navigate to the backend directory:

```bash
cd vectora-backend
```

Create your environment file:

```bash
cp .env.example .env
# Open .env and add your key:
# GEMINI_API_KEY=your_google_gemini_api_key_here
```

**Compile (Linux / macOS):**

```bash
g++ -std=c++17 -O2 -mavx2 -mfma main.cpp -o vectora_db -lpthread -lssl -lcrypto
```

**Compile (macOS with Homebrew OpenSSL):**

```bash
g++ -std=c++17 -O2 -mavx2 -mfma main.cpp -o vectora_db \
  -lpthread \
  -I/opt/homebrew/opt/openssl/include \
  -L/opt/homebrew/opt/openssl/lib \
  -lssl -lcrypto
```

**Compile (Windows — MSYS2 UCRT64):**

```bash
# Install MSYS2 from https://www.msys2.org, then:
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openssl

g++ -std=c++17 -O2 -mavx2 -mfma main.cpp -o vectora_db \
  -lws2_32 -lcrypt32 -lssl -lcrypto
```

> Add `C:\msys64\ucrt64\bin` to your Windows PATH, then reopen the terminal.

**Run the backend:**

```bash
./vectora_db
```

Expected output:

```
=== Vectora Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW + KD-Tree + BruteForce
AVX2 SIMD: ENABLED
Gemini API: CONNECTED
  embed: text-embedding-004 (768D)
  gen:   gemini-1.5-flash
```

---

### 3. Set Up the React Frontend

Open a new terminal:

```bash
cd vectora-ui
npm install
```

Create your frontend environment file:

```bash
cp .env.example .env
# Contents:
# VITE_PORT=3000
# VITE_API_BASE_URL=http://localhost:8080
```

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🐳 Docker Deployment

The backend ships with a production-ready Dockerfile. The container installs GCC, OpenSSL, and all dependencies automatically.

**Build and run locally:**

```bash
cd vectora-backend
docker build -t vectora-engine .
docker run -p 8080:8080 -e GEMINI_API_KEY="your_api_key" vectora-engine
```

**Deploy to Render:**

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set the root directory to `vectora-backend`
4. Render auto-detects the Dockerfile
5. Add `GEMINI_API_KEY` as an environment variable in the Render dashboard

**Deploy frontend to Vercel:**

```bash
cd vectora-ui
npm run build
# Then connect the repo to Vercel and set:
# VITE_API_BASE_URL=https://your-render-service.onrender.com
```

---

## 📡 REST API Reference

Base URL: `http://localhost:8080` (local) or your Render URL (production)

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Run all 3 algorithms, compare speed |
| `GET` | `/hnsw-info` | HNSW graph structure and layer statistics |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document via Gemini |
| `GET` | `/doc/list` | — | List all stored document chunks |
| `DELETE` | `/doc/delete/:id` | — | Delete a chunk by ID |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate answer |
| `GET` | `/status` | — | Gemini API status and model info |

### Example Requests

**Search via curl:**

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

**Insert a document:**

```bash
curl -X POST http://localhost:8080/doc/insert \
  -H "Content-Type: application/json" \
  -d '{"title":"Operating Systems Notes","text":"A process is a program in execution..."}'
```

**Ask a question (RAG):**

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is virtual memory?","k":3}'
```

---

## 💻 Using the Application

### Tab 1 — Search (Demo Vectors)

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose metric: **Cosine**, **Euclidean**, or **Manhattan**
- Hit **⚡ SEARCH** — results appear with distances; the matching point glows on the scatter plot
- Hit **▶ COMPARE ALL** to benchmark all 3 algorithms head-to-head and see timing differences

The scatter plot shows all 20 vectors projected to 2D via PCA. The 4 semantic categories (CS, Math, Food, Sports) form visually distinct clusters — this is what semantic similarity looks like in vector space.

### Tab 2 — Documents (Real Embeddings)

1. Enter a title (e.g., `Lecture Notes — OS Chapter 4`)
2. Paste any text — study material, technical docs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**

Long documents are automatically chunked into overlapping 250-word windows. Each chunk gets its own 768D Gemini embedding and lives as an independent node in the HNSW index.

### Tab 3 — Ask AI (RAG Pipeline)

1. Insert at least one document in Tab 2
2. Type a natural-language question
3. Click **🤖 ASK AI**

What happens under the hood:

```
1. Question → Gemini embeds it (768D vector)
2. HNSW search → retrieves 3 most semantically similar chunks
3. Retrieved chunks → passed as context to gemini-1.5-flash
4. Gemini → streams an answer grounded in your documents
```

Expand the **context chips** below the answer to see exactly which chunks the model used.

---

## 📊 Performance

Benchmarks on an M2 MacBook Pro (16GB RAM), 768D vectors, k=10:

| Algorithm | 1K vectors | 10K vectors | 100K vectors |
|---|---|---|---|
| Brute Force | ~2 ms | ~18 ms | ~185 ms |
| KD-Tree | ~0.4 ms | ~12 ms | ~160 ms (degrades) |
| HNSW | **< 0.05 ms** | **< 0.1 ms** | **< 0.3 ms** |

> HNSW's O(log N) complexity holds even at 100K+ vectors; KD-Tree approaches brute force at high dimensions.  
> AVX2 SIMD brings per-distance computation from ~120 ns to **< 15 ns**.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Gemini API: OFFLINE` in header | Check your `GEMINI_API_KEY` in `.env`; verify quota at aistudio.google.com |
| `g++: command not found` | Install GCC via your package manager; on Windows, add MSYS2 to PATH |
| AVX2 errors at compile time | Remove `-mavx2 -mfma` if your CPU predates Haswell (2013); performance will decrease |
| OpenSSL not found | Run `sudo apt install libssl-dev` (Ubuntu) or `brew install openssl` (macOS) |
| Port 8080 already in use | `lsof -i :8080 \| kill -9 <PID>` (Linux/macOS) or `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` (Windows) |
| Embedding takes a long time | First call initializes the Gemini connection — subsequent calls are fast |
| CORS errors in browser | Ensure `VITE_API_BASE_URL` in your frontend `.env` matches the running backend URL exactly |
| Docker build fails | Ensure Docker Desktop is running; try `docker system prune` to clear stale layers |

---

## 🤝 Contributing

Contributions are welcome! Areas where help is most valuable:

- **Product Quantization** — compress 768D vectors for faster memory access
- **IVF Index** — Inverted File Index for billion-scale approximate search
- **OpenAI Compatibility** — drop-in API layer for OpenAI embedding models
- **Benchmarking Suite** — automated regression tests for query latency across dataset sizes

Please open an issue before starting a large PR so we can align on approach.

---

## 👨‍💻 Author

**Kanak Mohan**

[![GitHub](https://img.shields.io/badge/GitHub-@kanakmohanj-181717?style=flat-square&logo=github)](https://github.com/kanakmohanj)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-link-here)

---

## 📄 License

MIT — use this however you want. Attribution appreciated but not required.