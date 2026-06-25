#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>
#include <immintrin.h>
// // --- BINARY I/O HELPERS ---
// template<typename T>
// void writeBinary(std::ofstream& out, const T& val) {
//     out.write(reinterpret_cast<const char*>(&val), sizeof(T));
// }

// template<typename T>
// void readBinary(std::ifstream& in, T& val) {
//     in.read(reinterpret_cast<char*>(&val), sizeof(T));
// }

// void writeString(std::ofstream& out, const std::string& str) {
//     size_t len = str.size();
//     writeBinary(out, len);
//     if (len > 0) out.write(str.data(), len);
// }

// void readString(std::ifstream& in, std::string& str) {
//     size_t len;
//     readBinary(in, len);
//     str.resize(len);
//     if (len > 0) in.read(&str[0], len);
// }

// void writeFloatVector(std::ofstream& out, const std::vector<float>& vec) {
//     size_t len = vec.size();
//     writeBinary(out, len);
//     if (len > 0) out.write(reinterpret_cast<const char*>(vec.data()), len * sizeof(float));
// }

// void readFloatVector(std::ifstream& in, std::vector<float>& vec) {
//     size_t len;
//     readBinary(in, len);
//     vec.resize(len);
//     if (len > 0) in.read(reinterpret_cast<char*>(vec.data()), len * sizeof(float));
// }
static const int DIMS = 16;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// =====================================================================
//  DATA TYPES
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

// float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
//     float s = 0;
//     for (int i = 0; i < (int)a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
//     return std::sqrt(s);
// }

// float cosine(const std::vector<float>& a, const std::vector<float>& b) {
//     float dot=0, na=0, nb=0;
//     for (int i = 0; i < (int)a.size(); i++) {
//         dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
//     }
//     if (na < 1e-9f || nb < 1e-9f) return 1.0f;
//     return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
// }

// float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
//     float s = 0;
//     for (int i = 0; i < (int)a.size(); i++) s += std::abs(a[i]-b[i]);
//     return s;
// }

// 1. SIMD Euclidean (L2) Distance
float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    size_t size = a.size();
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    // Process 8 dimensions per clock cycle
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum256 = _mm256_add_ps(sum256, sq);
    }

    // Extract the 8 partial sums
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum256);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // Tail case (for dimensions not perfectly divisible by 8)
    for (; i < size; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    
    return std::sqrt(s);
}

// 2. SIMD Cosine Distance
float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    size_t size = a.size();
    size_t i = 0;
    
    // We need to track three separate sums simultaneously
    __m256 dot256 = _mm256_setzero_ps();
    __m256 na256  = _mm256_setzero_ps();
    __m256 nb256  = _mm256_setzero_ps();

    // Process 8 dimensions per clock cycle
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        dot256 = _mm256_add_ps(dot256, _mm256_mul_ps(va, vb));
        na256  = _mm256_add_ps(na256,  _mm256_mul_ps(va, va));
        nb256  = _mm256_add_ps(nb256,  _mm256_mul_ps(vb, vb));
    }

    // Extract all partial sums
    alignas(32) float tmp_dot[8], tmp_na[8], tmp_nb[8];
    _mm256_storeu_ps(tmp_dot, dot256);
    _mm256_storeu_ps(tmp_na, na256);
    _mm256_storeu_ps(tmp_nb, nb256);

    float dot = 0, na = 0, nb = 0;
    for (int j = 0; j < 8; ++j) {
        dot += tmp_dot[j];
        na  += tmp_na[j];
        nb  += tmp_nb[j];
    }

    // Tail case
    for (; i < size; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

// 3. SIMD Manhattan (L1) Distance
float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    size_t size = a.size();
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    
    // BITWISE HACK: float absolute value mask (0x7FFFFFFF clears the sign bit)
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    // Process 8 dimensions per clock cycle
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_and_ps(diff, sign_mask); // Fast absolute value
        
        sum256 = _mm256_add_ps(sum256, abs_diff);
    }

    // Extract the 8 partial sums
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum256);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // Tail case
    for (; i < size; ++i) {
        s += std::abs(a[i] - b[i]);
    }
    
    return s;
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }
};

// =====================================================================
//  KD-TREE
// =====================================================================

struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return; destroy(n->left); destroy(n->right); delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
        else                              n->right = ins(n->right, v, d+1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        float dn = dist(q, n->item.emb);
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop();
        }
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  VECTOR DATABASE  (demo 16D index)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
    // // --- ADD THESE INSIDE class VectorDB ---

    // bool save_to_disk(const std::string& filename) {
    //     std::lock_guard<std::mutex> lk(mu); // Exclusive lock while saving
    //     std::ofstream out(filename, std::ios::binary);
    //     if (!out) return false;

    //     writeBinary(out, nextId);
    //     size_t count = store.size();
    //     writeBinary(out, count);

    //     for (const auto& [id, item] : store) {
    //         writeBinary(out, item.id);
    //         writeString(out, item.metadata);
    //         writeString(out, item.category);
    //         writeFloatVector(out, item.emb);
    //     }
    //     return true;
    // }

    // bool load_from_disk(const std::string& filename) {
    //     std::lock_guard<std::mutex> lk(mu); // Exclusive lock while loading
    //     std::ifstream in(filename, std::ios::binary);
    //     if (!in) return false;

    //     // Clear existing data before loading
    //     store.clear();
    //     // Assuming your BruteForce/KDTree/HNSW have a .clear() or reset method
    //     // If not, you may need to recreate them or just rely on this running at startup
        
    //     readBinary(in, nextId);
    //     size_t count;
    //     readBinary(in, count);

    //     for (size_t i = 0; i < count; ++i) {
    //         VectorItem item;
    //         readBinary(in, item.id);
    //         readString(in, item.metadata);
    //         readString(in, item.category);
    //         readFloatVector(in, item.emb);
            
    //         store[item.id] = item;
    //         // Rebuild the search indexes
    //         bf.insert(item); 
    //         kdt.insert(item); 
    //         hnsw.insert(item, getDistFn("cosine")); 
    //     }
    //     return true;
    // }
};

// =====================================================================
//  JSON HELPERS
// =====================================================================

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        try { v.push_back(std::stof(t)); } catch (...) {}
    return v;
}

// Extract a JSON string field value (handles basic escape sequences)
std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    if (p >= body.size() || body[p] != '"') return "";
    p++;
    std::string result;
    while (p < body.size()) {
        if (body[p] == '"') break;
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            switch (body[p]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[p]; break;
            }
        } else {
            result += body[p];
        }
        p++;
    }
    return result;
}

// Extract a JSON integer field value
int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    try { return std::stoi(body.substr(p)); } catch (...) { return def; }
}

bool parseBody(const std::string& b, std::string& meta,
               std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat  = extractStr(b, "category");
    auto extractArr = [&](const std::string& key) -> std::vector<float> {
        size_t p = b.find('"' + key + '"');
        if (p == std::string::npos) return {};
        p = b.find('[', p);
        if (p == std::string::npos) return {};
        size_t e = b.find(']', p);
        if (e == std::string::npos) return {};
        return parseVec(b.substr(p + 1, e - p - 1));
    };
    emb = extractArr("embedding");
    return !meta.empty() && !emb.empty();
}

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

std::vector<std::string> chunkText(const std::string& text,
                                   int chunkWords = 250, int overlapWords = 30)
{
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);

    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {text};

    std::vector<std::string> chunks;
    int step = chunkWords - overlapWords;
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) { if (j > i) chunk += ' '; chunk += words[j]; }
        chunks.push_back(chunk);
        if (end == (int)words.size()) break;
    }
    return chunks;
}

// // =====================================================================
// //  OLLAMA CLIENT  — wraps local Ollama REST API
// //  Install:  https://ollama.com
// //  Models:   ollama pull nomic-embed-text
// //            ollama pull llama3.2
// // =====================================================================

// class OllamaClient {
//     std::string host;
//     int         port;

//     // Escape a string for embedding inside a JSON string literal
//     std::string esc(const std::string& s) {
//         std::string o;
//         for (char c : s) {
//             if      (c == '"')  o += "\\\"";
//             else if (c == '\\') o += "\\\\";
//             else if (c == '\n') o += "\\n";
//             else if (c == '\r') o += "\\r";
//             else if (c == '\t') o += "\\t";
//             else                o += c;
//         }
//         return o;
//     }

//     // Parse {"embedding":[...]} from Ollama /api/embeddings response
//     std::vector<float> parseEmbedding(const std::string& body) {
//         size_t p = body.find("\"embedding\"");
//         if (p == std::string::npos) return {};
//         p = body.find('[', p);
//         if (p == std::string::npos) return {};
//         // Find matching ]  — embeddings can be large (768+ floats)
//         size_t e = p + 1, depth = 1;
//         while (e < body.size() && depth > 0) {
//             if (body[e] == '[') depth++;
//             else if (body[e] == ']') depth--;
//             e++;
//         }
//         return parseVec(body.substr(p + 1, e - p - 2));
//     }

//     // Parse {"response":"..."} from Ollama /api/generate response
//     std::string parseResponse(const std::string& body) {
//         return extractStr(body, "response");
//     }

// public:
//     std::string embedModel = "nomic-embed-text";
//     std::string genModel   = "llama3.2";

//     OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
//         : host(h), port(p) {}

//     bool isAvailable() {
//         httplib::Client cli(host, port);
//         cli.set_connection_timeout(2, 0);
//         auto res = cli.Get("/api/tags");
//         return res && res->status == 200;
//     }

//     // Returns empty vector if Ollama is not running or model not found
//     std::vector<float> embed(const std::string& text) {
//         httplib::Client cli(host, port);
//         cli.set_connection_timeout(3, 0);
//         cli.set_read_timeout(30, 0);
//         std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
//         auto res = cli.Post("/api/embeddings", body, "application/json");
//         if (!res || res->status != 200) return {};
//         return parseEmbedding(res->body);
//     }

//     // Returns error string if Ollama is unavailable
//     std::string generate(const std::string& prompt) {
//         httplib::Client cli(host, port);
//         cli.set_connection_timeout(3, 0);
//         cli.set_read_timeout(180, 0);   // LLMs can be slow
//         std::string body = "{\"model\":\"" + genModel + "\","
//                            "\"prompt\":\"" + esc(prompt) + "\","
//                            "\"stream\":false}";
//         auto res = cli.Post("/api/generate", body, "application/json");
//         if (!res || res->status != 200)
//             return "ERROR: Ollama unavailable. Run: ollama serve";
//         return parseResponse(res->body);
//     }
// };

// Gemini Client

class GeminiClient {
    std::string apiKey;
    std::string host = "generativelanguage.googleapis.com";

    // Escape a string for embedding inside a JSON string literal
    std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if      (c == '"')  o += "\\\"";
            else if (c == '\\') o += "\\\\";
            else if (c == '\n') o += "\\n";
            else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t";
            else                o += c;
        }
        return o;
    }

    // Parse {"embedding": {"values": [...]}} from Gemini API response
    std::vector<float> parseGeminiEmbedding(const std::string& body) {
        // Gemini nests the array under "values" inside the "embedding" object
        size_t p = body.find("\"values\"");
        if (p == std::string::npos) return {};
        
        p = body.find('[', p);
        if (p == std::string::npos) return {};
        
        // Find matching ] — embeddings are 768 floats
        size_t e = p + 1, depth = 1;
        while (e < body.size() && depth > 0) {
            if (body[e] == '[') depth++;
            else if (body[e] == ']') depth--;
            e++;
        }
        
        // Reusing your existing parseVec implementation
        return parseVec(body.substr(p + 1, e - p - 2));
    }

    // Parse {"text": "..."} from Gemini API response
    std::string parseGeminiResponse(const std::string& body) {
        // Gemini nests the response inside candidates -> content -> parts -> text
        // Assuming your extractStr properly extracts the value for a given JSON key
        return extractStr(body, "text");
    }

public:
    std::string embedModel = "gemini-embedding-2";
    std::string genModel   = "gemini-2.5-flash";

    // Constructor fetches API key from Environment Variables for Docker deployment
    GeminiClient() {
        // 1. First, check OS environment variables (Important for Docker later)
        const char* env_key = std::getenv("GEMINI_API_KEY");
        if (env_key) {
            apiKey = std::string(env_key);
            return;
        } 
        
        // 2. Fallback: Parse the .env file manually
        std::ifstream envFile(".env");
        if (envFile.is_open()) {
            std::string line;
            while (std::getline(envFile, line)) {
                // Ignore comments and empty lines
                if (line.empty() || line[0] == '#') continue;

                // Find the key-value pair
                size_t delimiterPos = line.find('=');
                if (delimiterPos != std::string::npos) {
                    std::string key = line.substr(0, delimiterPos);
                    if (key == "GEMINI_API_KEY") {
                        // Extract the value
                        std::string value = line.substr(delimiterPos + 1);

                        // 1. Remove quotes, carriage returns (\r), and newlines (\n)
                        value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
                        value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
                        value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());

                        // 2. Trim any leading or trailing spaces (just in case)
                        size_t start = value.find_first_not_of(" \t");
                        size_t end = value.find_last_not_of(" \t");
                        if (start != std::string::npos && end != std::string::npos) {
                            apiKey = value.substr(start, end - start + 1);
                        } else {
                            apiKey = value; 
                        }
                        break;
                    }
                }
            }
            envFile.close();
        }

        if (apiKey.empty()) {
            std::cerr << "WARNING: GEMINI_API_KEY not found in system env or .env file!" << std::endl;
        } else {
            std::cout << "Gemini API Key loaded successfully from .env file." << std::endl;
        }
    }

    bool isAvailable() {
        return !apiKey.empty();
    }

    // Returns empty vector if API request fails
    std::vector<float> embed(const std::string& text) {
        if (apiKey.empty()) return {};

        httplib::SSLClient cli(host);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(30, 0);

        // Gemini embed payload structure
        std::string body = "{\"model\":\"models/" + embedModel + "\",\"content\":{\"parts\":[{\"text\":\"" + esc(text) + "\"}]}}";
        std::string path = "/v1beta/models/" + embedModel + ":embedContent?key=" + apiKey;

        auto res = cli.Post(path.c_str(), body, "application/json");
        
        if (!res || res->status != 200) {
            std::cerr << "Gemini Embedding Error: " << (res ? std::to_string(res->status) : "Connection Failed") << "\n";
            return {};
        }
        
        return parseGeminiEmbedding(res->body);
    }

    // Returns error string if API request fails
    std::string generate(const std::string& prompt) {
        if (apiKey.empty()) return "ERROR: GEMINI_API_KEY is missing.";

        httplib::SSLClient cli(host);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(180, 0); 

        // Gemini generate payload structure
        std::string body = "{\"contents\":[{\"parts\":[{\"text\":\"" + esc(prompt) + "\"}]}]}";
        std::string path = "/v1beta/models/" + genModel + ":generateContent?key=" + apiKey;

        auto res = cli.Post(path.c_str(), body, "application/json");
        
        if (!res || res->status != 200) {
            return "ERROR: Gemini API request failed. Status: " + (res ? std::to_string(res->status) : "Connection Failed");
        }
        
        return parseGeminiResponse(res->body);
    }
};


// =====================================================================
//  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
// =====================================================================

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW       hnsw;
    BruteForce bf;       // brute force fallback for small sets
    std::mutex mu;
    int nextId = 1;
    int dims   = 0;      // determined from first inserted embedding

public:
    DocumentDB() : hnsw(16, 200) {}

    // Insert one chunk with its pre-computed embedding
    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (dims == 0) dims = (int)emb.size();
        DocItem item{nextId++, title, text, emb};
        store[item.id] = item;
        VectorItem vi{item.id, title, "doc", emb};
        hnsw.insert(vi, cosine);
        bf.insert(vi);
        return item.id;
    }

    // Semantic search — returns top-k most similar chunks
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 0.7f)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty()) return {};
        auto raw = (store.size() < 10)
                   ? bf.knn(q, k, cosine)
                   : hnsw.knn(q, k, 50, cosine);
        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw)
            if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); hnsw.remove(id); bf.remove(id);
        return true;
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    int getDims() { return dims; }

    //     // --- ADD THESE INSIDE class DocumentDB ---

    // bool save_to_disk(const std::string& filename) {
    //     std::lock_guard<std::mutex> lk(mu);
    //     std::ofstream out(filename, std::ios::binary);
    //     if (!out) return false;

    //     writeBinary(out, nextId);
    //     writeBinary(out, dims);
    //     size_t count = store.size();
    //     writeBinary(out, count);

    //     for (const auto& [id, item] : store) {
    //         writeBinary(out, item.id);
    //         writeString(out, item.title);
    //         writeString(out, item.text);
    //         writeFloatVector(out, item.emb);
    //     }
    //     return true;
    // }

    // bool load_from_disk(const std::string& filename) {
    //     std::lock_guard<std::mutex> lk(mu);
    //     std::ifstream in(filename, std::ios::binary);
    //     if (!in) return false;

    //     store.clear();
        
    //     readBinary(in, nextId);
    //     readBinary(in, dims);
    //     size_t count;
    //     readBinary(in, count);

    //     for (size_t i = 0; i < count; ++i) {
    //         DocItem item;
    //         readBinary(in, item.id);
    //         readString(in, item.title);
    //         readString(in, item.text);
    //         readFloatVector(in, item.emb);
            
    //         store[item.id] = item;
            
    //         VectorItem vi{item.id, item.title, "doc", item.emb};
    //         hnsw.insert(vi, cosine);
    //         bf.insert(vi);
    //     }
    //     return true;
    // }
};

// =====================================================================
//  DEMO DATA  (16D categorical vectors)
// =====================================================================

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");
    // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math",
        {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
}

// =====================================================================
//  HTTP SERVER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    DocumentDB docDB;
    GeminiClient gemini;
    std::shared_mutex db_mutex;
    // std::cout << "Loading databases from disk..." << std::endl;
    // if (db.load_from_disk("vectors.idx")) {
    //     std::cout << "Loaded " << db.size() << " demo vectors." << std::endl;
    // }
    // if (docDB.load_from_disk("docs.idx")) {
    //     std::cout << "Loaded " << docDB.size() << " documents." << std::endl;
    // }

    loadDemo(db);

    // Check Gemini at startup (non-fatal)
    bool geminiUp = gemini.isAvailable();
    std::cout << "=== VectorDB Engine ===" << std::endl;
    std::cout << "http://localhost:8080" << std::endl;
    std::cout << db.size() << " demo vectors | " << DIMS << " dims | HNSW+KD-Tree+BruteForce" << std::endl;
    std::cout << "Gemini: " << (geminiUp ? "ONLINE" : "OFFLINE (install from gemini.com)") << std::endl;
    if (geminiUp) std::cout << "  embed model: " << gemini.embedModel
                            << "  gen model: "   << gemini.genModel << std::endl;

    httplib::Server svr;

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        auto out = [&]() {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            return db.search(q, k, metric, algo);
        }();
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id;
        {
            // 2. Memory Write: Acquire EXCLUSIVE write lock to safely modify the graph
            std::unique_lock<std::shared_mutex> lock(db_mutex);
            id = db.insert(meta, cat, emb, getDistFn("cosine"));
        } // Lock is instantly released here!
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = false;

        {
            // ACQUIRE EXCLUSIVE WRITE LOCK: Block all reads and writes during deletion
            std::unique_lock<std::shared_mutex> lock(db_mutex);
            ok = db.remove(id);
        }
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        { 
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            auto items = db.all();
            ss << '[';
            for (size_t i = 0; i < items.size(); i++) {
                if (i) ss << ',';
                auto& v = items[i];
                ss << "{\"id\":"        << v.id
                << ",\"metadata\":"  << jS(v.metadata)
                << ",\"category\":"  << jS(v.category)
                << ",\"embedding\":" << jVec(v.emb) << '}';
            }
            ss << ']';
        }
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        // ACQUIRE SHARED LOCK: Only lock while actively traversing the graphs
        // This allows multiple recruiters or users to hit the benchmark simultaneously!
        auto b = [&]() {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            return db.benchmark(q, k, metric);
        }(); // Lock instantly releases when benchmarking finishes
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = [&]() {
            // ACQUIRE SHARED LOCK: Just long enough to safely copy the graph structure
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            return db.hnswInfo();
        }();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert  {"title":"...","text":"..."}
    // Chunks the text, embeds each chunk via Ollama, stores in DocumentDB
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;

        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = gemini.embed(chunks[i]);
            if (emb.empty()) {
                res.set_content(
                    "{\"error\":\"Gemini API unavailable or API Key invalid. Please check your .env file.\"}",
                    "application/json");
                return;
            }
            std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            {
                std::unique_lock<std::shared_mutex> lock(db_mutex);
                ids.push_back(docDB.insert(chunkTitle, chunks[i], emb));
            }
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":"    << docDB.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // DELETE /doc/delete/123
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = false;
        {
            std::unique_lock<std::shared_mutex> lock(db_mutex);
            ok = docDB.remove(id);
        }
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    // GET /doc/list
    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss; 
        { 
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            auto docs = docDB.all();
            ss << '[';
            for (size_t i = 0; i < docs.size(); i++) {
                if (i) ss << ',';
                // Truncate text preview to 120 chars
                std::string preview = docs[i].text.substr(0, 120);
                if (docs[i].text.size() > 120) preview += "…";
                ss << "{\"id\":" << docs[i].id
                << ",\"title\":" << jS(docs[i].title)
                << ",\"preview\":" << jS(preview)
                << ",\"words\":"  << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1
                << '}';
            }
            ss << ']';
        }
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/search {"question":"...","k":3}
    // Fast retrieval for the UI visualizer
    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        auto qEmb = gemini.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Gemini unavailable\"}", "application/json"); return;
        }

        //auto hits = docDB.search(qEmb, k);
        // Database Read: Acquire SHARED lock exclusively for the search execution
        auto hits = [&]() {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            return docDB.search(qEmb, k);
        }(); // The lock is instantly released once the search completes

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/ask  {"question":"...","k":3}
    // Full RAG pipeline: embed → retrieve → generate
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        // Step 1: embed the question
        auto qEmb = gemini.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Gemini unavailable\"}", "application/json"); return;
        }

        // Step 2: retrieve top-k relevant chunks
        // auto hits = docDB.search(qEmb, k);
        auto [hits, currentDocCount] = [&](const std::vector<float>& emb) {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            return std::make_pair(docDB.search(emb, k), docDB.size());
        }(qEmb);

        // Step 3: build prompt
        std::ostringstream ctx;
        for (int i = 0; i < (int)hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }
        std::string prompt =
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            "Just answer the question naturally.\n\n"
            "Context:\n" + ctx.str() +
            "Question: " + question + "\n\n"
            "Answer:";

        // Step 4: generate answer
        auto answer = gemini.generate(prompt);

        // Step 5: return everything
        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(gemini.genModel)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << currentDocCount << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = gemini.isAvailable();
        std::ostringstream ss;
        {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            ss << "{\"geminiAvailable\":"  << (up ? "true" : "false")
            << ",\"embedModel\":"       << jS(gemini.embedModel)
            << ",\"genModel\":"         << jS(gemini.genModel)
            << ",\"docCount\":"         << docDB.size()
            << ",\"docDims\":"          << docDB.getDims()
            << ",\"demoDims\":"         << DIMS
            << ",\"demoCount\":"        << db.size() << '}';
        }
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        {
            std::shared_lock<std::shared_mutex> lock(db_mutex);
            ss << "{\"count\":"      << db.size()
            << ",\"dims\":"       << DIMS
            << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
            << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        }
        res.set_content(ss.str(), "application/json");
    });

    // Serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
