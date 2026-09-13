// sched_toy.cpp -- experiment 3.2: can a CPU-assigned ggml split run CONCURRENTLY
// with a CUDA-assigned split using only the public ggml backend API?
//
// Build:
//   g++ -O2 -std=c++17 sched_toy.cpp -o sched_toy \
//     -I<llama.cpp>/ggml/include -L<llama.cpp>/build/bin \
//     -lggml -lggml-base -lggml-cpu -lggml-cuda -lpthread
//
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ---------------------------------------------------------------- parameters
struct Cfg {
    int cpu_k = 1024, cpu_b = 256, cpu_depth = 2;
    int gpu_k = 2048, gpu_b = 2048, gpu_depth = 3;
    int n_threads = 0;      // 0 => hardware_concurrency
    int iters = 20;
    int ov_iters = 500;     // dispatch-overhead iterations
    int tiny_k = 64, tiny_b = 8, tiny_depth = 1;
    int defer_gpu_depth = 100;   // GPU chain depth in the deferral test
    int layers = 32;             // alternating split-pairs per token
    int lay_k = 1024, lay_b = 32, lay_cpu_depth = 1, lay_gpu_depth = 24;
};

// ------------------------------------------------------------------ helpers
static void fill(ggml_tensor * t, unsigned seed) {
    size_t n = ggml_nelements(t);
    std::vector<float> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 0.1f;
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// A self-contained matmul chain living entirely in one backend buffer.
// out = W_d * ... * W_1 * x , all [K,K] x [K,B] -> [K,B]
struct Chain {
    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph         * gf  = nullptr;
    ggml_tensor         * in  = nullptr;
    ggml_tensor         * out = nullptr;
    double gflop = 0.0;

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

static Chain build_chain(ggml_backend_t backend, int K, int B, int depth,
                         unsigned seed, const char * tag) {
    Chain c;
    size_t mem = ggml_tensor_overhead() * (size_t)(2 * depth + 16) + ggml_graph_overhead() + (1u << 20);
    ggml_init_params ip = { mem, nullptr, /*no_alloc*/ true };
    c.ctx = ggml_init(ip);

    c.in = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, B);
    ggml_set_name(c.in, (std::string(tag) + "_in").c_str());

    std::vector<ggml_tensor *> W(depth);
    for (int i = 0; i < depth; i++) {
        W[i] = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, K);
        ggml_set_name(W[i], (std::string(tag) + "_w" + std::to_string(i)).c_str());
    }

    ggml_tensor * x = c.in;
    for (int i = 0; i < depth; i++) {
        x = ggml_mul_mat(c.ctx, W[i], x);           // [K,K] x [K,B] -> [K,B]
        ggml_set_name(x, (std::string(tag) + "_h" + std::to_string(i)).c_str());
    }
    c.out = x;
    ggml_set_name(c.out, (std::string(tag) + "_out").c_str());

    c.gf = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(c.gf, c.out);

    // allocate EVERYTHING (weights + intermediates) in this backend's buffer:
    // no cross-split aliasing, no ggml-alloc reuse hazards.
    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    if (!c.buf) { fprintf(stderr, "alloc failed for %s\n", tag); exit(1); }

    fill(c.in, seed);
    for (int i = 0; i < depth; i++) fill(W[i], seed + 100 + i);

    c.gflop = 2.0 * (double)K * K * B * depth / 1e9;
    return c;
}

// persistent worker thread (what a real integration would use)
class Worker {
    std::thread th;
    std::mutex m;
    std::condition_variable cv_job, cv_done;
    std::function<void()> job;
    bool has_job = false, done = true, quit = false;
public:
    Worker() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                { std::unique_lock<std::mutex> lk(m);
                  cv_job.wait(lk, [this] { return has_job || quit; });
                  if (quit) return;
                  j = job; has_job = false; }
                j();
                { std::lock_guard<std::mutex> lk(m); done = true; }
                cv_done.notify_one();
            }
        });
    }
    void submit(std::function<void()> j) {
        { std::lock_guard<std::mutex> lk(m); job = std::move(j); has_job = true; done = false; }
        cv_job.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return done; });
    }
    ~Worker() {
        { std::lock_guard<std::mutex> lk(m); quit = true; }
        cv_job.notify_all(); th.join();
    }
};

struct Stat { double best, med, mean; };
static Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0; for (double x : v) s += x;
    return { v.front(), v[v.size()/2], s/(double)v.size() };
}
static void report(const char * name, const Stat & s) {
    printf("  %-46s best %8.3f ms   med %8.3f ms   mean %8.3f ms\n", name, s.best, s.med, s.mean);
}

// ============================================================== MAIN
int main(int argc, char ** argv) {
    Cfg cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() { return atoi(argv[++i]); };
        if      (a == "--cpu-k")     cfg.cpu_k = val();
        else if (a == "--cpu-b")     cfg.cpu_b = val();
        else if (a == "--cpu-depth") cfg.cpu_depth = val();
        else if (a == "--gpu-k")     cfg.gpu_k = val();
        else if (a == "--gpu-b")     cfg.gpu_b = val();
        else if (a == "--gpu-depth") cfg.gpu_depth = val();
        else if (a == "--threads")   cfg.n_threads = val();
        else if (a == "--iters")     cfg.iters = val();
        else if (a == "--ov-iters")  cfg.ov_iters = val();
        else if (a == "--defer-gpu-depth") cfg.defer_gpu_depth = val();
        else if (a == "--layers")        cfg.layers = val();
        else if (a == "--lay-k")         cfg.lay_k = val();
        else if (a == "--lay-b")         cfg.lay_b = val();
        else if (a == "--lay-cpu-depth") cfg.lay_cpu_depth = val();
        else if (a == "--lay-gpu-depth") cfg.lay_gpu_depth = val();
    }
    if (cfg.n_threads <= 0) cfg.n_threads = (int)std::thread::hardware_concurrency();

    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, cfg.n_threads);

    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) { fprintf(stderr, "no cuda backend\n"); return 1; }

    char desc[256] = {0};
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    printf("GPU: %s\n", desc);
    printf("CPU backend threads: %d (hw concurrency %u)\n", cfg.n_threads, std::thread::hardware_concurrency());
    printf("CPU chain : K=%d B=%d depth=%d\n", cfg.cpu_k, cfg.cpu_b, cfg.cpu_depth);
    printf("CUDA chain: K=%d B=%d depth=%d\n", cfg.gpu_k, cfg.gpu_b, cfg.gpu_depth);
    printf("iters=%d\n\n", cfg.iters);

    Chain A = build_chain(cpu,  cfg.cpu_k, cfg.cpu_b, cfg.cpu_depth, 1u, "cpuA");
    Chain B = build_chain(cuda, cfg.gpu_k, cfg.gpu_b, cfg.gpu_depth, 2u, "gpuB");
    printf("branch A (CPU)  work: %.3f GFLOP\n", A.gflop);
    printf("branch B (CUDA) work: %.3f GFLOP\n\n", B.gflop);

    const int W = std::max(3, cfg.iters / 4);   // warmup

    // ---------------------------------------------------- 0. isolated timings
    printf("== [0] isolated branch times ==\n");
    std::vector<double> vA, vB, vAt;
    for (int i = 0; i < W + cfg.iters; i++) {
        auto t = clk::now();
        ggml_backend_graph_compute(cpu, A.gf);
        if (i >= W) vA.push_back(ms_since(t));
    }
    for (int i = 0; i < W + cfg.iters; i++) {
        auto t = clk::now();
        ggml_backend_graph_compute(cuda, B.gf);
        if (i >= W) vB.push_back(ms_since(t));
    }
    // does running the CPU backend from a non-main std::thread cost anything?
    {
        Worker w;
        for (int i = 0; i < W + cfg.iters; i++) {
            auto t = clk::now();
            w.submit([&] { ggml_backend_graph_compute(cpu, A.gf); });
            w.wait();
            if (i >= W) vAt.push_back(ms_since(t));
        }
    }
    Stat sA = stats(vA), sB = stats(vB), sAt = stats(vAt);
    report("T_A  (CPU branch, main thread)", sA);
    report("T_A' (CPU branch, from worker thread)", sAt);
    report("T_B  (CUDA branch)", sB);
    printf("  -> CPU %.1f GFLOPS   CUDA %.1f GFLOPS\n",
           A.gflop / (sA.med / 1000.0), B.gflop / (sB.med / 1000.0));
    printf("  -> T_A+T_B = %.3f ms   max(T_A,T_B) = %.3f ms\n\n",
           sA.med + sB.med, std::max(sA.med, sB.med));

    // ---------------------------------------------------- 1. serial baseline
    printf("== [1] manual SERIAL (what sched does today) ==\n");
    std::vector<double> vSer;
    for (int i = 0; i < W + cfg.iters; i++) {
        auto t = clk::now();
        ggml_backend_graph_compute(cpu, A.gf);           // blocking
        ggml_backend_graph_compute_async(cuda, B.gf);
        ggml_backend_synchronize(cuda);
        if (i >= W) vSer.push_back(ms_since(t));
    }
    Stat sSer = stats(vSer);
    report("serial A then B", sSer);
    printf("\n");

    // ------------------------------- 2. concurrent, NO extra thread (reorder)
    // CUDA graph_compute_async really is async -> issue it first, then do the
    // CPU split on this same thread, then sync CUDA.
    printf("== [2] concurrent, async-only (issue CUDA first, CPU on main thread) ==\n");
    std::vector<double> vAsy;
    for (int i = 0; i < W + cfg.iters; i++) {
        auto t = clk::now();
        ggml_backend_graph_compute_async(cuda, B.gf);    // returns immediately
        ggml_backend_graph_compute(cpu, A.gf);           // overlaps with GPU
        ggml_backend_synchronize(cuda);
        if (i >= W) vAsy.push_back(ms_since(t));
    }
    Stat sAsy = stats(vAsy);
    report("async-only overlap", sAsy);
    printf("\n");

    // -------------------------- 3. concurrent, CPU split on a worker std::thread
    printf("== [3] concurrent, CPU split on std::thread ==\n");
    std::vector<double> vSpawn, vPersist;
    for (int i = 0; i < W + cfg.iters; i++) {
        auto t = clk::now();
        ggml_backend_graph_compute_async(cuda, B.gf);
        std::thread th([&] { ggml_backend_graph_compute(cpu, A.gf); });
        ggml_backend_synchronize(cuda);
        th.join();
        if (i >= W) vSpawn.push_back(ms_since(t));
    }
    {
        Worker w;
        for (int i = 0; i < W + cfg.iters; i++) {
            auto t = clk::now();
            ggml_backend_graph_compute_async(cuda, B.gf);
            w.submit([&] { ggml_backend_graph_compute(cpu, A.gf); });
            ggml_backend_synchronize(cuda);
            w.wait();
            if (i >= W) vPersist.push_back(ms_since(t));
        }
    }
    Stat sSpawn = stats(vSpawn), sPer = stats(vPersist);
    report("fresh std::thread per iteration", sSpawn);
    report("persistent worker thread (cv handoff)", sPer);
    printf("\n");

    // ------------------------------------ 4. ggml_backend_sched (mode a)
    printf("== [4] default ggml_backend_sched on the same 2-branch graph ==\n");
    {
        // weights/inputs pre-placed in their backend's buffer; sched infers
        // placement from where the weights live (exactly like llama.cpp).
        size_t memw = ggml_tensor_overhead() * 256 + (1u << 20);
        ggml_init_params ipw = { memw, nullptr, true };
        ggml_context * ctx_wc = ggml_init(ipw);
        ggml_context * ctx_wg = ggml_init(ipw);

        ggml_tensor * inA = ggml_new_tensor_2d(ctx_wc, GGML_TYPE_F32, cfg.cpu_k, cfg.cpu_b);
        std::vector<ggml_tensor *> WA(cfg.cpu_depth);
        for (int i = 0; i < cfg.cpu_depth; i++) WA[i] = ggml_new_tensor_2d(ctx_wc, GGML_TYPE_F32, cfg.cpu_k, cfg.cpu_k);
        ggml_tensor * inB = ggml_new_tensor_2d(ctx_wg, GGML_TYPE_F32, cfg.gpu_k, cfg.gpu_b);
        std::vector<ggml_tensor *> WB(cfg.gpu_depth);
        for (int i = 0; i < cfg.gpu_depth; i++) WB[i] = ggml_new_tensor_2d(ctx_wg, GGML_TYPE_F32, cfg.gpu_k, cfg.gpu_k);

        ggml_backend_buffer_t bwc = ggml_backend_alloc_ctx_tensors(ctx_wc, cpu);
        ggml_backend_buffer_t bwg = ggml_backend_alloc_ctx_tensors(ctx_wg, cuda);
        ggml_backend_buffer_set_usage(bwc, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(bwg, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        fill(inA, 1u); for (int i = 0; i < cfg.cpu_depth; i++) fill(WA[i], 101 + i);
        fill(inB, 2u); for (int i = 0; i < cfg.gpu_depth; i++) fill(WB[i], 102 + i);

        size_t memc = ggml_tensor_overhead() * 512 + ggml_graph_overhead() + (1u << 20);
        ggml_init_params ipc = { memc, nullptr, true };
        ggml_context * ctx_c = ggml_init(ipc);
        ggml_tensor * xa = inA; for (int i = 0; i < cfg.cpu_depth; i++) xa = ggml_mul_mat(ctx_c, WA[i], xa);
        ggml_tensor * xb = inB; for (int i = 0; i < cfg.gpu_depth; i++) xb = ggml_mul_mat(ctx_c, WB[i], xb);
        ggml_cgraph * gf = ggml_new_graph(ctx_c);
        ggml_build_forward_expand(gf, xa);
        ggml_build_forward_expand(gf, xb);

        ggml_backend_t bes[2] = { cuda, cpu };
        ggml_backend_buffer_type_t bts[2] = { ggml_backend_cuda_buffer_type(0), ggml_backend_cpu_buffer_type() };
        ggml_backend_sched_t sched = ggml_backend_sched_new(bes, bts, 2, GGML_DEFAULT_GRAPH_SIZE, false, /*op_offload*/ false);
        ggml_backend_sched_reserve(sched, gf);

        std::vector<double> vSch;
        for (int i = 0; i < W + cfg.iters; i++) {
            ggml_backend_sched_reset(sched);
            auto t = clk::now();
            ggml_backend_sched_graph_compute(sched, gf);
            if (i >= W) vSch.push_back(ms_since(t));
        }
        Stat sSch = stats(vSch);
        printf("  n_splits = %d\n", ggml_backend_sched_get_n_splits(sched));
        report("ggml_backend_sched_graph_compute", sSch);
        printf("  sched wall / (T_A+T_B) = %.3f     sched wall / max(T_A,T_B) = %.2f\n\n",
               sSch.med / (sA.med + sB.med), sSch.med / std::max(sA.med, sB.med));

        ggml_backend_sched_free(sched);
        ggml_free(ctx_c);
        ggml_backend_buffer_free(bwc); ggml_backend_buffer_free(bwg);
        ggml_free(ctx_wc); ggml_free(ctx_wg);
    }

    // ------------------------------------ 5. deferral-shaped topology (mode c)
    printf("== [5] deferral-shaped topology + numerical correctness ==\n");
    {
        // CPU: "layer N experts"  E = expert_chain(x_cpu)
        // CUDA:"layer N+1 attn"   ATT = attn_chain(x_gpu)
        // join on CUDA (one step later): OUT = ATT + upload(E)
        // -- E is produced concurrently with ATT but consumed AFTER it.
        // Both branches produce the SAME shape [cpu_k, cpu_b] (the "residual"),
        // so they can be added. The GPU branch uses a much deeper chain to reach
        // a comparable wall time, since the T4 is ~100x faster than 8 CPU cores.
        Chain E   = build_chain(cpu,  cfg.cpu_k, cfg.cpu_b, cfg.cpu_depth,       7u, "exp");
        Chain ATT = build_chain(cuda, cfg.cpu_k, cfg.cpu_b, cfg.defer_gpu_depth, 8u, "att");
        printf("  expert(CPU) %.3f GFLOP   attn(CUDA) %.3f GFLOP   join shape [%d,%d]\n",
               E.gflop, ATT.gflop, cfg.cpu_k, cfg.cpu_b);

        // join graph on CUDA
        size_t memj = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + (1u << 20);
        ggml_init_params ipj = { memj, nullptr, true };
        ggml_context * ctxj = ggml_init(ipj);
        ggml_tensor * e_gpu = ggml_new_tensor_2d(ctxj, GGML_TYPE_F32, cfg.cpu_k, cfg.cpu_b);
        // consume ATT.out directly (already allocated in the CUDA buffer) so the
        // timing isn't polluted by a redundant D2D copy
        ggml_tensor * outj  = ggml_add(ctxj, ATT.out, e_gpu);
        ggml_cgraph * gj = ggml_new_graph(ctxj);
        ggml_build_forward_expand(gj, outj);
        ggml_backend_buffer_t bufj = ggml_backend_alloc_ctx_tensors(ctxj, cuda);

        size_t nb = ggml_nbytes(outj);
        std::vector<float> ref(nb / 4), got(nb / 4);

        auto run_serial = [&] {
            ggml_backend_graph_compute(cpu,  E.gf);            // experts first
            ggml_backend_tensor_copy(E.out, e_gpu);
            ggml_backend_graph_compute(cuda, ATT.gf);          // then attention
            ggml_backend_graph_compute(cuda, gj);
        };
        auto run_concurrent = [&] {
            ggml_backend_graph_compute_async(cuda, ATT.gf);    // issue GPU attn
            ggml_backend_graph_compute(cpu,  E.gf);            // CPU experts overlap
            ggml_backend_synchronize(cuda);                    // GPU attn done
            ggml_backend_tensor_copy(E.out, e_gpu);            // upload deferred expert out
            ggml_backend_graph_compute(cuda, gj);
        };
        auto run_concurrent_thread = [&](Worker & w) {
            ggml_backend_graph_compute_async(cuda, ATT.gf);
            w.submit([&] { ggml_backend_graph_compute(cpu, E.gf); });
            ggml_backend_synchronize(cuda);
            w.wait();
            ggml_backend_tensor_copy(E.out, e_gpu);
            ggml_backend_graph_compute(cuda, gj);
        };

        run_serial();
        ggml_backend_tensor_get(outj, ref.data(), 0, nb);

        run_concurrent();
        ggml_backend_tensor_get(outj, got.data(), 0, nb);
        size_t bad = 0; double maxabs = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            if (memcmp(&ref[i], &got[i], 4) != 0) bad++;
            maxabs = std::max(maxabs, (double)std::fabs(ref[i] - got[i]));
        }
        printf("  correctness (async-only overlap) : %zu/%zu differing floats, max|diff| = %.3e  -> %s\n",
               bad, ref.size(), maxabs, bad == 0 ? "BIT-IDENTICAL" : "MISMATCH");

        {
            Worker w;
            run_concurrent_thread(w);
            ggml_backend_tensor_get(outj, got.data(), 0, nb);
            bad = 0; maxabs = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                if (memcmp(&ref[i], &got[i], 4) != 0) bad++;
                maxabs = std::max(maxabs, (double)std::fabs(ref[i] - got[i]));
            }
            printf("  correctness (worker-thread overlap): %zu/%zu differing floats, max|diff| = %.3e  -> %s\n",
                   bad, ref.size(), maxabs, bad == 0 ? "BIT-IDENTICAL" : "MISMATCH");
        }

        std::vector<double> vS, vC, vT;
        for (int i = 0; i < W + cfg.iters; i++) { auto t = clk::now(); run_serial();     if (i>=W) vS.push_back(ms_since(t)); }
        for (int i = 0; i < W + cfg.iters; i++) { auto t = clk::now(); run_concurrent(); if (i>=W) vC.push_back(ms_since(t)); }
        { Worker w;
          for (int i = 0; i < W + cfg.iters; i++) { auto t = clk::now(); run_concurrent_thread(w); if (i>=W) vT.push_back(ms_since(t)); } }
        Stat ss = stats(vS), sc = stats(vC), st = stats(vT);
        report("deferral serial", ss);
        report("deferral concurrent (async-only)", sc);
        report("deferral concurrent (worker thread)", st);
        printf("  speedup async-only = %.3fx   worker-thread = %.3fx\n\n", ss.med / sc.med, ss.med / st.med);

        ggml_backend_buffer_free(bufj); ggml_free(ctxj);
        E.free_all(); ATT.free_all();
    }

    // ------------------------------------ 6. per-split-pair dispatch overhead
    printf("== [6] dispatch overhead per split-pair (tiny graphs) ==\n");
    {
        Chain ta = build_chain(cpu,  cfg.tiny_k, cfg.tiny_b, cfg.tiny_depth, 11u, "tinyA");
        Chain tb = build_chain(cuda, cfg.tiny_k, cfg.tiny_b, cfg.tiny_depth, 12u, "tinyB");
        int N = cfg.ov_iters;
        auto bench = [&](const char * name, std::function<void()> f) {
            for (int i = 0; i < 50; i++) f();
            auto t = clk::now();
            for (int i = 0; i < N; i++) f();
            double per = ms_since(t) / N;
            printf("  %-46s %8.4f ms/pair\n", name, per);
            return per;
        };
        double p_a  = bench("tiny: CPU only",           [&]{ ggml_backend_graph_compute(cpu, ta.gf); });
        double p_b  = bench("tiny: CUDA only",          [&]{ ggml_backend_graph_compute(cuda, tb.gf); });
        double p_s  = bench("tiny: serial pair",        [&]{ ggml_backend_graph_compute(cpu, ta.gf);
                                                             ggml_backend_graph_compute(cuda, tb.gf); });
        double p_as = bench("tiny: async-only overlap", [&]{ ggml_backend_graph_compute_async(cuda, tb.gf);
                                                             ggml_backend_graph_compute(cpu, ta.gf);
                                                             ggml_backend_synchronize(cuda); });
        double p_sp = bench("tiny: fresh std::thread",  [&]{ ggml_backend_graph_compute_async(cuda, tb.gf);
                                                             std::thread th([&]{ ggml_backend_graph_compute(cpu, ta.gf); });
                                                             ggml_backend_synchronize(cuda); th.join(); });
        double p_pw;
        { Worker w;
          p_pw = bench("tiny: persistent worker",       [&]{ ggml_backend_graph_compute_async(cuda, tb.gf);
                                                             w.submit([&]{ ggml_backend_graph_compute(cpu, ta.gf); });
                                                             ggml_backend_synchronize(cuda); w.wait(); }); }
        printf("\n  overhead vs serial pair:\n");
        printf("    async-only     : %+8.4f ms/pair\n", p_as - p_s);
        printf("    fresh thread   : %+8.4f ms/pair  (thread spawn+join cost)\n", p_sp - p_s);
        printf("    persistent wrk : %+8.4f ms/pair  (cv handoff cost)\n", p_pw - p_s);
        printf("    [ref] cpu-only %.4f  cuda-only %.4f  serial %.4f ms\n\n", p_a, p_b, p_s);
        ta.free_all(); tb.free_all();
    }

    // ---------------- 7. REAL granularity: n_layers alternating split-pairs/token
    // This is the question that matters: with 16-40 CPU/GPU split pairs per token,
    // does per-pair dispatch overhead swamp the overlap win?
    printf("== [7] per-layer granularity: %d alternating split-pairs per token ==\n", cfg.layers);
    {
        // one small CPU "deferred expert" and one small GPU "attention" per layer,
        // reused across layers (identical work each layer)
        Chain la = build_chain(cpu,  cfg.lay_k, cfg.lay_b, cfg.lay_cpu_depth, 21u, "layA");
        Chain lb = build_chain(cuda, cfg.lay_k, cfg.lay_b, cfg.lay_gpu_depth, 22u, "layB");
        printf("  per-layer: CPU %.4f GFLOP, CUDA %.4f GFLOP (K=%d B=%d, cpu_depth=%d gpu_depth=%d)\n",
               la.gflop, lb.gflop, cfg.lay_k, cfg.lay_b, cfg.lay_cpu_depth, cfg.lay_gpu_depth);

        int L = cfg.layers;
        auto tok_serial = [&] {
            for (int l = 0; l < L; l++) {
                ggml_backend_graph_compute(cpu,  la.gf);
                ggml_backend_graph_compute(cuda, lb.gf);
            }
        };
        auto tok_async = [&] {
            for (int l = 0; l < L; l++) {
                ggml_backend_graph_compute_async(cuda, lb.gf);
                ggml_backend_graph_compute(cpu,  la.gf);
                ggml_backend_synchronize(cuda);
            }
        };
        auto tok_worker = [&](Worker & w) {
            for (int l = 0; l < L; l++) {
                ggml_backend_graph_compute_async(cuda, lb.gf);
                w.submit([&] { ggml_backend_graph_compute(cpu, la.gf); });
                ggml_backend_synchronize(cuda);
                w.wait();
            }
        };
        // isolated per-layer costs
        std::vector<double> pa, pb;
        for (int i = 0; i < 200; i++) { auto t=clk::now(); ggml_backend_graph_compute(cpu, la.gf);  if(i>=50) pa.push_back(ms_since(t)); }
        for (int i = 0; i < 200; i++) { auto t=clk::now(); ggml_backend_graph_compute(cuda, lb.gf); if(i>=50) pb.push_back(ms_since(t)); }
        Stat spa = stats(pa), spb = stats(pb);
        printf("  per-layer isolated: CPU %.4f ms   CUDA %.4f ms\n", spa.med, spb.med);

        std::vector<double> vS, vA2, vW;
        for (int i = 0; i < W + cfg.iters; i++) { auto t=clk::now(); tok_serial(); if(i>=W) vS.push_back(ms_since(t)); }
        for (int i = 0; i < W + cfg.iters; i++) { auto t=clk::now(); tok_async();  if(i>=W) vA2.push_back(ms_since(t)); }
        { Worker w;
          for (int i = 0; i < W + cfg.iters; i++) { auto t=clk::now(); tok_worker(w); if(i>=W) vW.push_back(ms_since(t)); } }
        Stat ssS = stats(vS), ssA = stats(vA2), ssW = stats(vW);
        report("token: serial (all 2L splits in order)", ssS);
        report("token: async-only overlap", ssA);
        report("token: persistent worker thread", ssW);
        double ideal_ser = L * (spa.med + spb.med);
        double ideal_ovl = L * std::max(spa.med, spb.med);
        printf("  ideal serial %.3f ms   ideal perfect-overlap %.3f ms\n", ideal_ser, ideal_ovl);
        printf("  speedup: async-only %.3fx   worker-thread %.3fx\n", ssS.med/ssA.med, ssS.med/ssW.med);
        printf("  overlap efficiency: async-only %.1f%%   worker-thread %.1f%%\n",
               (ssS.med-ssA.med)/(ssS.med-ideal_ovl)*100.0, (ssS.med-ssW.med)/(ssS.med-ideal_ovl)*100.0);
        printf("  per-split-pair overhead vs serial: async-only %+.4f ms  worker %+.4f ms\n\n",
               (ssA.med - ideal_ovl)/L, (ssW.med - ideal_ovl)/L);
        la.free_all(); lb.free_all();
    }

    // ------------------------------------------------------- final summary
    printf("== SUMMARY ==\n");
    printf("  T_A (CPU)   = %.3f ms\n", sA.med);
    printf("  T_B (CUDA)  = %.3f ms\n", sB.med);
    printf("  T_A + T_B   = %.3f ms  (serial ideal)\n", sA.med + sB.med);
    printf("  max(T_A,T_B)= %.3f ms  (perfect overlap ideal)\n", std::max(sA.med, sB.med));
    printf("  measured serial       = %.3f ms\n", sSer.med);
    printf("  measured async-only   = %.3f ms\n", sAsy.med);
    printf("  measured worker-thread= %.3f ms\n", sPer.med);
    {
        double smax = std::max(sA.med, sB.med), ssum = sA.med + sB.med;
        auto eff = [&](double w) { return (ssum - w) / (ssum - smax) * 100.0; };
        printf("  overlap efficiency (100%% = perfect): async-only %.1f%%  worker-thread %.1f%%\n",
               eff(sAsy.med), eff(sPer.med));
    }

    A.free_all(); B.free_all();
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    return 0;
}
