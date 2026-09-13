# Expert-Deferral Quality Smoke Test — Design Notes

Experiment 4.1-smoke. Dev half: patch + runbook only. No measurements run here.

---

## 0. What was recovered, and provenance

The v11 build's source tree is **`/data/projects/revv-home/src/llama.cpp`** on the box (ssh alias
`ollama`). This is not a guess — it was proven by checksum:

```
cc83854f51309bd8efc8c97a529244ba  /data/projects/q27b_on_12gb/llama-server-v11/llama-server
cc83854f51309bd8efc8c97a529244ba  /data/projects/revv-home/src/llama.cpp/build/bin/llama-server
91a3b91a3559d742d0f40cc5c22ab16c  .../llama-server-v11/libllama.so.0.3.0
91a3b91a3559d742d0f40cc5c22ab16c  .../revv-home/src/llama.cpp/build/bin/libllama.so.0.3.0
1c921d19157a95be34d393c7b213dbdd  .../llama-server-v11/libllama-server-impl.so
1c921d19157a95be34d393c7b213dbdd  .../revv-home/src/llama.cpp/build/bin/libllama-server-impl.so
```

All three deployed artifacts are byte-identical to that tree's build output. **No upstream clone
was needed; there is no discrepancy to flag.**

Tree state: `git HEAD = daef7b687` ("vulkan: top_k radix select for k >= 1024 for Qwen 3.8 Flash
Next (#28032)"), plus four *uncommitted* working-tree modifications which are the two patches named
in the `llama-server-v11` wrapper script:

| File | Patch |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/vecdotq.cuh` | `mmvq_iquant_decode.patch` |
| `tools/server/server-context.cpp`, `tools/server/tests/unit/test_slot_save.py` | `pr26004-rebased-daef7b687.patch` |

**None of those four files is touched by our patch**, so there is no conflict surface.

Local working copy: `/Users/yugendren/experiments/q27b_on_12gb/deferral_smoke/llamacpp-v11/`
(source only; `build/` excluded). Its `.git` was renamed to `.git.disabled` because the box repo
uses `objects/info/alternates -> /data/projects/llama.cpp/.git/objects`, which does not exist
locally. The patch is therefore a plain `diff -u` (`-p1`, applies with `git apply` or `patch -p1`).

## 1. Architecture facts, verified not assumed

GGUF header of `/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`, read directly:

```
general.architecture                       = qwen35moe
qwen35moe.block_count                      = 41      <- 40 trunk + 1 MTP (blk.40)
qwen35moe.nextn_predict_layers             = 1
qwen35moe.embedding_length                 = 2048
qwen35moe.full_attention_interval          = 4       <- blocks 3,7,11,... are full-attn; rest GDN
qwen35moe.expert_count                     = 256
qwen35moe.expert_used_count                = 8
qwen35moe.expert_feed_forward_length       = 512
qwen35moe.expert_shared_feed_forward_length= 512
qwen35moe.ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size} = 4,128,16,32,4096
```

So the graph builder is **`llama_model_qwen35moe::graph`** in
`src/models/qwen35moe.cpp` — *not* `qwen3next` and *not* `qwen3moe`. (This fork splits every
architecture into its own file under `src/models/`; `llama-model.cpp:321` only dispatches.)
`hparams.is_recr(il) = (il < 40) && ((il+1) % 4 != 0)` marks 30 gated-delta-net blocks and 10
full-attention blocks; the brief's "~6" was an underestimate but is irrelevant to this patch, which
is FFN-side only and identical on both layer types.

### Residual wiring (pre-norm), per trunk block `il`

```
inpSA          = inpL
cur            = RMSNorm(inpL, attn_norm)
cur            = attn(cur)                       # GDN or full-attn
cur            = cur + inpSA                     # "attn_residual"
ffn_residual   = cur
attn_post_norm = RMSNorm(cur, attn_post_norm)
cur            = build_layer_ffn(attn_post_norm) # = moe_out + shexp_gated
cur            = cur + ffn_residual              # "post_moe"
inpL           = build_cvec(cur, il)
```

**Routed and shared expert outputs ARE summed before joining the residual** —
`qwen35moe.cpp:544` (original): `cur = ggml_add(ctx0, moe_out, ffn_shexp)`. The shared expert is
gated by its own sigmoid scalar gate (`ffn_gate_inp_shexp` -> sigmoid -> elementwise multiply)
*before* that sum. Our patch splits exactly at that `ggml_add`, which is why it is a clean
intervention point: `moe_out` is the complete routed-expert contribution and nothing else.

---

## 2. THE JOIN-POINT CHOICE (and the off-by-one that nearly shipped)

**Chosen: the deferred `moe_out(i)` is added into the residual stream at the TOP of block `i+d+1`
— equivalently, at the END of block `i+d` instead of the end of block `i`.**

Concretely, in the loop:

```
for il in 0..n_layer-1:
    inpL += pending[il]        <-- THE JOIN POINT
    res->t_layer_inp[il] = inpL
    ... attn_norm, attention, post-attn norm, FFN ...
    ...
    defer_push(moe_out(il), il + d + 1)
```

So `moe_out(i)` is **absent from the input of blocks `i+1 .. i+d`** and **present from block
`i+d+1` onward**. Exactly `d` blocks run stale.

### The +1 is load-bearing. Read this before "simplifying" it.

The brief specified the join point as *"after block `i+d-1`'s own contributions, before block
`i+d`'s processing"* — i.e. the top of block `i+d`, without the `+1`. **That is a no-op at `d=1`,
and it under-delivers by one block at every `d`.** I implemented it that way first and the local
runtime check caught it.

Why: stock block `i` ends with

```
cur  = shexp(i) + moe_out(i) + ffn_residual(i)     # "post_moe"
inpL = build_cvec(cur, i)                          # identity when no control vector is loaded
```

and the very next thing that happens is the top of block `i+1`. So "top of block `i+1`" and "end of
block `i`" are the *same point on the residual stream*. Deferring by one and joining at the top of
block `i+1` re-associates the same three summands at the same place — it changes nothing.

This was confirmed empirically, not just argued. On a 2-layer synthetic `qwen35moe` model, the
`i+d` wiring at `d=1` produced logits **bit-identical** to the stock unpatched build (same
FNV-1a-64 hash `0218c1f0d921e0dd`, same sum to 17 significant figures) across two independent
weight seeds. Only `d>=2` diverged.

Had this shipped, the experiment would have measured `d-1` blocks of staleness while reporting `d`,
and a GREEN light at "d=3" would have authorised a scheduler with a 3-block overlap window that had
only ever been validated for 2. That is a silent, dangerous error, which is exactly why the local
runtime check exists.

The brief's *primary* sentence — *"join the residual stream at layer N+d instead of N"* — is the
correct specification and is what is implemented. A contribution normally joins at the end of block
`N`; joining "at layer `N+d`" means joining at the end of block `N+d`. The parenthetical gloss was
the off-by-one.

### Why this point and not an alternative

| # | Join point | Blocks running stale | Verdict |
|---|---|---|---|
| A | top of block `i+d+1` ≡ end of block `i+d` | exactly `d` | **chosen** |
| B | top of block `i+d` ≡ end of block `i+d-1` | `d-1` | rejected — no-op at `d=1` |
| C | into `ffn_residual` of block `i+d+1` (after its attention) | `d` + 1 attention | rejected — non-integer depth |

Point A is the correct model of a real overlapped implementation:

1. **It matches the physical overlap window exactly.** A real scheduler dispatches block `i`'s CPU
   expert work as soon as `attn_post_norm(i)` exists, then runs blocks `i+1 .. i+d` on the GPU, then
   *must* synchronise before block `i+d+1`'s `attn_norm` reads the stream. The CPU therefore gets
   exactly `d` full blocks of GPU time. `d` in this experiment means precisely "number of blocks of
   GPU work available to hide CPU expert latency", which is the number the scheduler designer needs.

2. **It keeps the contribution on the residual highway, not inside a block's internals.** The
   residual stream is the only tensor that persists across block boundaries. Injecting at a block
   boundary means the real implementation needs one fence and one `ggml_add` on the stream, with no
   surgery inside a block. Point C would require reaching into block `i+d+1`'s intermediate
   `ffn_residual`, which is uglier and gives a non-integer effective depth.

Implementation note: A is coded as "top of block `i+d+1`" rather than "end of block `i+d`" so that
the join never interacts with `build_cvec`. The two are identical whenever no control vector is
loaded (the production case); with a control vector they differ in whether the deferred contribution
is subject to block `i+d`'s cvec. Noted in §4.7.

### Conservation

Every contribution is delivered exactly once. This is structurally airtight, not merely tested:

- `defer_push(t, target)` writes to slot `min(target, n_layer)`; since `d >= 1`,
  `target = i + d + 1 > i`, so a contribution can never be pushed into a slot the loop has already
  passed.
- Slot `j < n_layer` is drained at the top of block `j` and set to `nullptr`.
- Slot `n_layer` (the tail slot) is drained immediately before `output_norm`.
- A loop asserts every slot is null after the tail flush. This is **deliberately not** under
  `NDEBUG`: the measurement build is Release, and a dropped contribution would not crash — it would
  silently corrupt the quality number the experiment exists to produce. Cost is a ≤41-iteration
  pointer scan per graph build. (It was initially `#ifndef NDEBUG`, which meant it was compiled out
  of every build anyone would actually run; caught in review of the local verification.)
- Contributions whose target lands on a **non-deferring** block (`>= k`) are still joined at that
  block's top — the boundary at `k` needs no special case, because injection is unconditional at
  every block, and only *production* is gated on `i < k`.
- Multiple contributions targeting the same slot are `ggml_add`-accumulated, not overwritten.

For the production config (`k=16`, `d<=3`, `n_layer=40`) targets span `[d+1, 16+d] ⊆ [2,19]`, so the
tail slot is never used. It is implemented and tested anyway (see RUNBOOK check T2), because it is
needed for any `k` near `n_layer` and because dropping a contribution would silently corrupt the
result rather than crash.

### Row-reduction edge case

`qwen35moe.cpp` contains, in the last block only:

```cpp
if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
```

From that line onward the stream carries `n_outputs` rows, not `n_tokens`. A contribution already
sitting in the tail slot was computed with full `n_tokens` rows, so the tail flush would be a shape
mismatch. The patch row-reduces the tail slot inside that same branch. Only the tail slot can be
non-empty there (every slot `<= il` was drained at the top of its own block), and contributions
produced later in that same block are already reduced because they derive from the reduced `cur`.
Unreachable in the production config; correct anyway.

### MTP head (blk.40)

`res->t_h_nextn` is assigned from `cur` **after** `output_norm`, and the tail flush happens
**before** `output_norm`. Therefore the MTP seed sees a fully-flushed residual stream, as required.
`graph_mtp` itself is untouched: blk.40 is a single block outside `0..k-1`, nothing to defer, and
its own routed experts are GPU-resident.

---

## 3. THE SHARED-EXPERT FINDING

**Claim under test:** "the shared expert stays immediate because it is GPU-resident in production."
**Verdict: CONFIRMED.** `--n-cpu-moe 16` does *not* move the shared expert to CPU.

Evidence, from the v11 tree itself:

`common/common.h:1130`
```cpp
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
```

`common/common.h:1134`
```cpp
inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}
```

`common/common.h:1142`
```cpp
inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex, std::vector<llama_model_tensor_buft_override> & overrides) {
    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}
```

`common/arg.cpp:2788` registers `{"-ncmoe","--n-cpu-moe"}` -> `llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, ...)`.

Matching is `std::regex_search` (unanchored, first-match-wins) at
`src/llama-model-loader.cpp:1227-1253`, against the full tensor name including `.weight`.

Three conclusions follow:

1. **`--n-cpu-moe 16` = blocks 0..15, the FIRST 16.** The loop is `for (int i = 0; i < n; ++i)` with
   no reference to `n_layer`. Help text agrees: *"keep the Mixture of Experts (MoE) weights of the
   first N layers in the CPU"*. Our `LLAMA_DEFER_LAYERS=16` default matches this exactly.
2. **Shared experts stay on GPU.** Their names are `blk.%d.ffn_{gate,up,down}_shexp`
   (`src/llama-arch.cpp:456-458`). The regex demands the literal token `exps` (optionally prefixed
   `ch`) right after `ffn_<up|down|gate|gate_up>_`; it sees `shexp`. `sh` is not `ch` and is not
   empty-then-`exps`. No match, at any offset, because `\.ffn_` occurs only once in the name. Also
   corroborated: `grep -rn "shexp" common/` returns zero hits — nothing in the CLI layer ever
   targets shared experts.
3. **The router `blk.%d.ffn_gate_inp` stays on GPU too** (`src/llama-arch.cpp:436`) — after `gate_`
   the regex needs `exps`, it sees `inp`. This matters: routing decisions are made on-GPU with the
   *current* (non-stale) hidden state, which is what the simulation assumes.

So the tensors that actually move to CPU under `-ncmoe 16` are exactly
`blk.{0..15}.ffn_{gate,up,gate_up,down}_exps` — precisely the set whose output our patch defers.
The correspondence between "what `-ncmoe` puts on the CPU" and "what the patch defers" is exact.

One caveat for the eventual real build: overrides are first-match-wins and `-ot` patterns parsed
earlier win over `-ncmoe`. If the production command line ever grows an `-ot`, re-verify placement.

---

## 4. WHAT THE SIMULATION DOES **NOT** CAPTURE

This is the important section. A GREEN light from this experiment is a green light on **numerics
only**. These risks remain live and must not be assumed away.

### 4.1 Faithfulness of the expert INPUT — handled, but note why it mattered

The failure mode the brief warned about is real and was explicitly avoided: the routed experts for
block `i` are fed `attn_post_norm(i)`, i.e. **block `i`'s own** post-attention-normed hidden state,
computed from a residual stream that does *not* yet contain the deferred contribution. This is
exactly what a real overlapped implementation hands the CPU, because it dispatches the moment that
activation exists. Only the *output* join is late. If the simulation had instead computed the
experts from a later block's hidden state, it would have been measuring a different (and easier)
model. Grep check: `build_layer_ffn(attn_post_norm, il, ...)` — the first argument is unchanged from
stock.

Note the second-order consequence, which IS captured: for a deferring block `i`, `attn_post_norm(i)`
itself is downstream of joins from blocks `i-d..i-1`, so staleness compounds through the stack. The
simulation gets this right for free because the join is in the real stream.

### 4.2 Risks the simulation does not capture

1. **Router/gating drift is captured; expert *weight* staleness is not applicable.** Nothing here.
   But: the router runs on GPU on the *current* stream. In a real build, if the scheduler ever
   dispatched the router early (to give the CPU its expert IDs sooner), routing would become stale
   too — a strictly worse regime that this patch does *not* simulate. **If the real implementation
   needs early routing to hit its latency target, this experiment's result does not cover it.**
   Flag this to the scheduler designer before the build starts.

2. **Numerical precision and accumulation order.** Here the deferred `moe_out` is produced by the
   same backend as everything else in a CPU-only-experts split, then added on the same device in
   fp32. A real overlapped build introduces a genuine CPU->GPU transfer of a `[2048, n_tokens]`
   tensor with its own dtype/copy semantics, and the add happens at a different point in the
   backend-scheduler's graph split. Expect small numerical differences beyond what we measure.
   Should be far below the staleness effect, but it is not zero and it is not measured here.

3. **Batch/ubatch and chunked-prefill interaction.** The simulation holds `d` pending tensors live
   across block boundaries *within one graph*. A real implementation must do the same, and must get
   it right across ubatch boundaries, across the prefill/decode transition, and across speculative
   accept/reject rollbacks. **The rollback case is the scariest and is completely untested here:**
   if a speculative draft is rejected after a deferred contribution has been joined, the pending
   state must be unwound. Our simulation is stateless per graph and never sees this.

4. **KV-cache / recurrent-state interaction.** 30 of 40 blocks are gated-delta-net and carry
   *recurrent state* (`ssm_states_all`, `conv_states_all`) that is mutated in place as the graph
   runs. Deferral changes the input to those blocks' attention (via the residual), so the recurrent
   state trajectory changes — that IS captured. What is *not* captured is any real-implementation
   hazard where the recurrent state update races the deferred join. In a serial graph there is no
   race; in the overlapped build there could be. This is a scheduler-correctness risk, not a
   quality risk, but it is the one most likely to produce a silent wrong answer.

5. **Compute-buffer growth (small, but not nothing).** Holding up to `d` pending
   `[n_embd=2048, n_tokens]` fp32 tensors live extends their lifetime in the ggml graph allocator.
   Upper bound: `d * 2048 * n_tokens * 4` bytes. At `-ub 512, d=3` that is **~12.6 MB**; at decode
   (`n_tokens` small) it is negligible. **Model weight VRAM is completely unchanged** — expert
   placement is untouched by this patch. So the runbook's "VRAM unchanged" expectation is right for
   weights and buffers; a ≤13 MB bump in the reported compute buffer at `d=3` is expected and
   benign. If the measurement agent sees more than that, something is wrong.

6. **Speculative decoding is switched OFF in the measurement** (`--spec-type none`, see RUNBOOK).
   That is deliberate — see the runbook — but it means the *interaction* between deferral and the
   MTP/draft path is unmeasured. A separate follow-up is needed before shipping deferral with spec
   enabled.

7. **Control vectors.** `build_cvec(cur, il)` is applied to block `il`'s output. A deferred
   contribution bypasses block `i`'s cvec application and is instead subject to block `i+d`'s.
   Irrelevant in production (no `--control-vector` in the config) but noted for completeness.

8. **This says nothing about whether the speedup exists.** The patch is deliberately
   scheduling-neutral: everything remains serial, at unchanged cost. A GREEN light means "the
   staleness is not harmful", not "the overlap will pay". The performance case still has to be made
   separately.

9. **The real build can reproduce the off-by-one, and it would be invisible.** §2 documents that
   joining at the top of block `i+d` instead of `i+d+1` is a numerical no-op at `d=1`. The mirror
   image of that mistake in the *scheduler* is worse: if the real implementation places its fence
   before block `i+d`'s `attn_norm`, it gets only `d-1` blocks of overlap while believing it has
   `d`. It would still be numerically correct — just slower than designed, with the shortfall hidden
   inside a speedup number nobody can attribute. **Whoever builds the scheduler should be handed
   this specific statement:** the fence for block `i`'s deferred experts goes immediately before
   block `i+d+1`'s `attn_norm`, and blocks `i+1 .. i+d` must run without the contribution. Ask them
   to assert on it.

---

## 5. Files

| Path | What |
|---|---|
| `deferral_sim.patch` | the patch, `-p1`, against the recovered tree |
| `RUNBOOK.md` | box-side rebuild + measurement commands, decision rule |
| `NOTES.md` | this file |
| `llamacpp-v11/` | local working copy of the recovered v11 source tree (patch applied) |
| `orig/` | pristine copies of the two modified files, for diffing |

Patch touches exactly two files:
- `src/models/qwen35moe.cpp` — the deferral logic
- `src/models/models.h` — one signature change (`build_layer_ffn` gains a defaulted out-param)

Verified: `patch -p1 --dry-run` applies cleanly, no fuzz, against copies of the box's current
`src/models/qwen35moe.cpp` and `src/models/models.h`. Neither file is touched by the two v11 patches.

## 7. RESULTS AND THE DIAGNOSTIC MISSTEP (box run)

### 7.1 Leg A — perplexity, 20 chunks, `-c 2048`, ncm16, deterministic

Corpus `results/deferral_smoke/ppl_corpus.txt`, md5 `fad6232f789eb3c39f1e4d2d9c122969`.
Build: `/data/scratch/deferral_smoke/llama.cpp` (isolated; v11 tree untouched, md5-verified).

| d | PPL | Δ vs d=0 |
|---|---|---|
| 0 | 6.8224 ± 0.13696 | — |
| 1 | 7.8123 ± 0.15842 | **+14.5%** |
| 2 | 9.0465 ± 0.18634 | **+32.6%** |
| 3 | 9.9576 ± 0.20700 | **+45.9%** |

`llama-perplexity` is deterministic on a fixed corpus and seed, so these deltas are signal,
not sampling noise. The curve is monotone and steep — roughly linear in `d` at ~+15%/block.
Per-arm engagement was confirmed independently from the repeating log line
`graph: LLAMA_DEFER_DEPTH env present -> defer_depth=N defer_layers=16` (N = 0,1,2,3 as expected;
`defer_layers` correctly collapses to 0 at `d=0`).

**Read:** this is a strong early indication of RED. Even `d=1` — the minimum deferral that buys
any overlap at all — costs ~14.5% perplexity. HumanEval is the decision instrument (§7.3), but
perplexity of this magnitude is not a rounding error.

### 7.2 The diagnostic misstep — recorded because it nearly produced a false GREEN

Three separate failures on the box, all of the same species: **the experiment silently running the
baseline while reporting it was running the treatment.**

1. `pkill -f "port 18099"` matched the ssh command string that contained it, killing my own session.
2. The "ACTIVE" banner never appeared on the box. I instrumented, saw the caller run and the
   static-initializer lambda apparently not, and concluded the config was silently defaulting to
   OFF. I rewrote it to plain `getenv`.
3. **That conclusion was wrong.** A controlled same-build A/B settled it:
   `--chunks 1` gives PPL **16.3023 at d=0** vs **22.7726 at d=3**. The 22.7726 I had been
   treating as "the baseline" was in fact the *deferred* value. Deferral had been working on the
   box the entire time; only the log line was missing.

The error was inferring *"the code did not run"* from *"the log did not appear"* — while logging
was itself the thing under suspicion. The correct move, which I eventually made, was to stop
reading logs and run a controlled A/B on the actual output.

Leading explanation for the missing line (not yet proven, see §7.4): `LLAMA_LOG_WARN` calls emitted
during the **first graph build** appear to be dropped. That would selectively hide exactly the
one-shot messages (the banner, the schedule) while leaving per-build repeating messages visible —
which is precisely the observed pattern: the unconditional env line appears 87 times in a 20-chunk
run, while the once-guarded banner and 16-line schedule never appear at all.

**Consequence for anyone using this patch: do NOT use the "EXPERT-DEFERRAL SIMULATION ACTIVE"
banner or the schedule dump as your engagement check on Linux.** Use the repeating line:

```
graph: LLAMA_DEFER_DEPTH env present -> defer_depth=N defer_layers=K
```

### 7.3 Leg B — HumanEval-164, `--spec-type none`

Running `d=0` and `d=1` only. Rationale: Leg A already puts `d=2`/`d=3` far outside any plausible
noise band, so the only decision-relevant question left is whether the *minimum* deferral is
survivable. If `d=1` degrades HE-164, the optimisation is dead as specified and `d=2`/`d=3` need
no confirmation. If `d=1` is clean on HE-164 despite +14.5% PPL, that is itself an interesting
result (perplexity overstating agentic harm, consistent with this project's own
`BENCHMARK_LANDSCAPE.md` Spearman +0.24 finding) and would justify measuring `d=2`.

### 7.4 Open item

The first-graph-build log-drop hypothesis is unproven. Definitive test, to run when the GPU is
free: make the schedule log fire on *every* graph build instead of once, rebuild, and re-run. If
it then prints, one-shot messages are being dropped; if it prints only from the second build
onward, that confirms the first-build drop specifically. Not blocking — engagement is already
verifiable via the repeating line — but it should be settled before anyone trusts a one-shot log
in this codebase again.

## 6. Local verification performed (dev half)

- **Compile:** clean CPU-only build on macOS (`-DGGML_CUDA=OFF -DGGML_METAL=OFF`, Release). No
  errors, no warnings from the changed translation unit.
- **Runtime:** `tests/test-llama-archs` can emit a synthetic random model for `qwen35moe`
  (`-o <dir>` -> `qwen35moe-moe.gguf`, 2 layers). A standalone fingerprint harness (fixed token ids,
  one `llama_decode`, FNV-1a-64 over the final-position logit bytes) was run against it.
  - `LLAMA_DEFER_DEPTH` unset == `=0` == the **stock unpatched build**, bit-identical. So `d=0` is a
    true no-op and the four measurement arms can share one binary.
  - Active configurations run to completion with finite, non-NaN logits, no assert failures,
    including the end-of-stack tail-flush path.
  - This is what caught the `i+d` vs `i+d+1` join-point bug (§2). The check paid for itself.
- **Depth curve is live after the fix.** On an 8-layer synthetic `qwen35moe`, with all layers
  deferring:

  | arm | env | FNV-1a-64 of final logits |
  |---|---|---|
  | A | unset | `09fe14918cbb4aa5` |
  | B | `DEPTH=0` | `09fe14918cbb4aa5` |
  | STOCK | unpatched build | `09fe14918cbb4aa5` |
  | C | `DEPTH=1 LAYERS=8` | `b57d91a3176b448c` |
  | D | `DEPTH=2 LAYERS=8` | `e4ca161f6f8e6946` |
  | E | `DEPTH=3 LAYERS=8` | `2b1b0bb6e34327bb` |

  `A == B == STOCK` bit-identical (so `d=0` is a genuine no-op and one binary serves all arms), and
  `d=1,2,3` are all distinct. Before the join-point fix, `C` collided with `A`.

  The schedule log renders correctly at the boundary, including the singular/empty stale-range
  cases: `blk 6 ... -> tail flush (pre output_norm) (blk 7 runs stale)` and
  `blk 7 ... -> tail flush (pre output_norm) (no blk runs stale)`.

- **Conservation guard exercised.** With the assert made unconditional, a Release build runs
  `d∈{0,1,2,3}` across `k∈{1,6,8,16}` — interior joins, tail-flush joins, and mixed — with zero
  assert failures and unchanged fingerprints.

- **Magnitudes from the synthetic model are meaningless.** Its weights are `N(0, 1e-2)` with no
  training signal, so the routed-expert path (SwiGLU through two tiny matmuls) collapses to the
  noise floor. Measured by tensor-level `cb_eval` trace: the deferred `ffn_moe_out_deferred-0` has
  `maxabs = 1.58e-8` against a residual stream at `maxabs ≈ 0.06-0.09` — six orders of magnitude.
  Consequently a run with only *one* deferring block (`DEPTH=1 LAYERS=1`) is bit-identical to
  baseline even though the schedule log and the graph trace both confirm the deferral engaged and
  the nodes executed: the perturbation is below the float32 ulp of the output logits. Empirically,
  two or more contributions must sum into the same slot before the hash moves. This is a property of
  the toy weights, not of the patch. **The local checks establish wiring correctness and the `d=0`
  identity; they say nothing about how much quality moves.** That is exactly what Leg A/B/C are for.

- **Not verified locally:** anything requiring the real 17 GB model or CUDA.
