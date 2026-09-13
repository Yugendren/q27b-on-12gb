# INTERNAL DRAFT — NOT FOR PUBLICATION

Mericanii (Yugendren) · proposed public repo layout and launch checklist for
the Qwen3.8-27B on RTX 3060 12GB artifact. Nothing here is live.

---

## 1. Proposed repo layout

```
qwen3.8-27b-12gb/
├── README.md               # repo front page — flagship config + quickstart
├── PAPER.md                # THE canonical paper; everything links here
├── REPORT.md               # one-line pointer to PAPER.md (kept for old links)
├── LICENSE                 # Apache-2.0, harness + paper only
├── CITATION.cff            # how to cite; url/repository-code fill in at launch
├── harness/
│   ├── profile_hw.py            # generic — hardware snapshot to JSON
│   ├── bench_speed.py           # generic — wraps llama-bench, sha256, nvidia-smi
│   ├── eval_quality.py          # generic — GSM8K + HumanEval vs a live server
│   ├── bestofn.py               # generic — best-of-n verified-selection eval
│   ├── run_matrix.py            # generic — drives the config matrix
│   ├── yaml_lite.py             # generic — stdlib-only minimal YAML parser
│   ├── bootstrap_linux.sh       # MACHINE-SPECIFIC — setup, GPU gating, CUDA build
│   ├── run_{eval,curve,ablations,spec2,speedopt,bestofn}.sh  # MACHINE-SPECIFIC
│   ├── smoke_server.sh          # MACHINE-SPECIFIC — sanity check
│   └── data/                    # cached HumanEval + GSM8K (check licences)
├── configs/
│   ├── matrix.example.yaml      # annotated example, safe to publish as-is
│   └── matrix.smoke.yaml        # minimal smoke-test config
├── results/
│   ├── README.md                # JSON schema description
│   ├── full164/                 # the four full-164 runs + best-of-n + ablations
│   ├── model_hashes.txt         # SHA-256 of every measured GGUF
│   └── summary.md               # derived headline table, cited by README + PAPER
└── docs/
    ├── caveats.md               # noise finding, best-of-n negative, n=20 flags
    └── faq.md                   # contamination, seed=1, chat-template protocol
```

Notes:

- **Generic vs machine-specific.** The Python files are stdlib + `requests` and
  run anywhere with `llama-server`/`llama-bench`/`llama-cli` on PATH. The
  `run_*.sh` drivers and `bootstrap_linux.sh` were written for the author's
  Ubuntu 24.04 box and embed paths, GPU indices and package-manager
  assumptions — scrub or parameterize each before publication (§2b).
- **`results/`** holds the evidence. Every public number must trace to a file
  here; `results/README.md` documents the `bench_speed.py` schema field by
  field (`model_file`, `sha256_16`, `file_size_gb`, `flags`, `pp_tok_s`,
  `tg_tok_s`, `peak_vram_mb`, `llama_cpp_build`, `timestamp`).

### Deliberately excluded

- **Model weights.** Never rehosted; README and PAPER link upstream to
  `unsloth/Qwen3.8-27B-GGUF` and `froggeric/Qwen-Fixed-Chat-Templates`.
- **Multi-GB logs.** Only per-task JSON and short supporting excerpts ship.
- **Absolute paths, hostnames, local usernames** in any config, script or
  committed log — needs an explicit grep pass (§2b).

---

## 2. Pre-launch checklist

### (a) Numbers

- [ ] Every figure in PAPER.md traces to a file in `results/full164/`
- [ ] Every figure in README.md traces to the same source as its PAPER.md
      counterpart (no independent transcription)
- [ ] The four HumanEval scores (49.4%, 62.2%, 72.0%, 82.3%) match
      `results/summary.md` character-for-character
- [ ] No number appears with two different values across README.md, PAPER.md
      and `results/summary.md`
- [ ] `results/model_hashes.txt` — recorded on the measurement machine;
      **commit it to this checkout.** Until it lands, the curve cannot be tied
      to specific files (quantizers re-upload under unchanged filenames).
- [ ] The ~10 points/GiB exchange rate is labelled derived (approximately
      linear over 6.8–10.2 GiB, every adjacent step significant under paired
      McNemar testing, n=4 points — no claim of linearity beyond that), not
      independently measured
- [ ] Flagship numbers (82.3%, 33.8 tok/s, 11642 MiB, 16K ctx, q8_0 KV) match
      their source run

### (b) Reproducibility

- [ ] Harness runs clean from a fresh `git clone` (nothing outside the repo
      except the GGUF downloads and llama.cpp binaries)
- [ ] `bootstrap_linux.sh` tested end to end on a machine that is **not** the
      author's
- [ ] Absolute paths and hostnames scrubbed from scripts, configs, logs
- [ ] `matrix.example.yaml` / `matrix.smoke.yaml` reviewed for machine-specific
      values (GPU index, ports, local paths)
- [ ] Dataset licences for `harness/data/` noted in `docs/` or
      `results/README.md`
- [ ] Each `run_*.sh` generalized, or clearly marked reference-not-turnkey

### (c) Claims hygiene

- [ ] Every claim in PAPER.md and README.md labelled measured vs inferred
- [ ] Warm-cache ngram-mod numbers stay out of headline claims and keep their
      two-sentence warning (PAPER.md §4.1)
- [ ] The Minitron comparison stays flagged n=20-only and matched-protocol
      (0.50 vs 0.80), never mixed with the full-164 table
- [ ] Best-of-5 result stated as a negative result, not spun positive
- [ ] "First independent [X]" stays hedged to "we believe" / "to our knowledge"
- [ ] Eval protocol restated wherever a score appears: full 164-task HumanEval,
      pass@1, temp 0, chat-template prompting via `/v1/chat/completions`
- [ ] Speed figures never silently mix conditions — `llama-bench` tg128 and
      live-serving decode with MTP are different runs and are labelled as such
- [ ] GSM8K's removal stated as a reasoned drop (at ceiling, 0.94–1.00), not
      omitted silently
- [ ] No invented or unverifiable citations anywhere

### (d) Legal / hosting

- [x] `LICENSE` at repo root, Apache-2.0, scoped to harness + paper
      (`pub/LICENSE`, added; scope note in-file covers weights/datasets
      exclusion)
- [x] `CITATION.cff` at repo root (`pub/CITATION.cff`, added; `url` /
      `repository-code` left blank pending the real repo link)
- [ ] README/PAPER state weights are not rehosted and link upstream
- [ ] Chat template attributed and linked, not vendored
- [ ] Third-party benchmark numbers attributed with a source link

### (e) Presentation

- [ ] Tables in README.md and PAPER.md render in an actual GitHub preview
- [ ] The flagship quickstart is copy-pasteable and has been run on a clean
      checkout
- [ ] Contact path for corrections is present and correct in both docs
      (currently `19thkingisreal@gmail.com`; one line per doc, swap both if a
      professional alias replaces it)
- [ ] NOT FOR PUBLICATION banners removed from README.md, PAPER.md and
      REPORT.md — **last step before going public**

---

## 3. Launch sequence

1. **Repo public first**, only after §2 is fully checked off. It is the
   canonical source; everything else links back to it.
2. **Hugging Face community post second** (draft in §5), on the
   `unsloth/Qwen3.8-27B-GGUF` community tab, once the repo link resolves.
3. **Reddit (r/LocalLLaMA) third** (§4), linking the live repo.
4. **Correction window:** 24–48h after the Reddit post before treating numbers
   as final. Watch Issues, HF comments, Reddit replies, and the contact inbox.
5. **Named owner:** Mericanii (Yugendren) watches all channels during the
   window and fixes any claim found wrong.

---

## 4. r/LocalLLaMA post

**Flair:** Discussion. (Not News/Resources — this is a measurement writeup with
a negative result in it, and Discussion is the honest fit for "here's what I
found, tell me where I'm wrong," not a promo tag.)

**Title:**
`Qwen3.8-27B's built-in speculation head is worth +68% decode, free — measured on a 12GB 3060 (plus a full HumanEval quant curve, 6.8-10.2 GiB)`

**Body draft:**

I run Mericanii, an independent lab; this is our work — no product, nothing
for sale, just measurements and the harness that produced them.

Biggest single finding: Qwen3.8-27B ships a **built-in MTP speculation head**
that's free speed — **+68% decode, lossless** (same output distribution,
verified) — and it is silently stripped from GGUF builds under about 8.4 GiB,
so the smaller quants pay for it twice: worse quality *and* no free speedup.
Nothing in the filename or model card says so.

Also ran the full quant curve, because nobody had published it: Qwen3.8-27B
(unsloth GGUF quants) on an RTX 3060 12GB via llama.cpp (build 10566, commit
bb4caa754, CUDA), full 164-task HumanEval, pass@1, temp 0, chat-template
prompting through `/v1/chat/completions`: UD-IQ2_XXS (6.76 GiB) 49.4%,
UD-IQ2_S (7.79 GiB) 62.2%, UD-Q2_K_XL (9.14 GiB) 72.0%, UD-IQ3_XXS (10.17 GiB)
82.3% — roughly 10 points per GiB, every adjacent step significant under
paired McNemar testing (n=4 points, so no claim about the shape between them).
Best config that fits: IQ3_XXS + the built-in MTP head at n=2 + q8_0 KV, 82.3%
at ~34 tok/s in 11.6 GB. Also tried best-of-5 at temp 0.8 to recover quality
without a bigger quant, and it didn't help: pass@1 60.0%, oracle ceiling
75.0%, verified-selection 67.1%, all below the single greedy temp-0 run at
82.3% — pre-registered gates, written before the GPU time was spent, killed
the follow-up run.

Why bother running a 27B locally on a $250 card at all — not to save money
over an API, the accounting doesn't favor local and we're not pretending it
does. It's that the weights, the quant, and the config are yours: no account,
no rate limit, no changing prices, works with the network off. If that's not
what you're optimizing for, an API is probably cheaper and easier — this is
for people who want to own the thing they're running.

Full harness, raw results and paper: `[repo link]`.

**Expected pushback:**

- *"HumanEval is saturated/contaminated at this size."* Agree — known-imperfect
  proxy; the curve is a same-benchmark, same-harness comparison across quant
  sizes on one machine, not a claim about absolute capability.
- *"n=1 seed, no error bars."* Partially — we do report 95% CIs and paired
  McNemar tests on the curve now (see paper §3); what's still true is one seed,
  one run per point, flagged as a limitation.
- *"Chat-template prompting isn't leaderboard-comparable."* Agree — different
  prompting protocol, so not comparable to leaderboard entries. Stated in §7.

**Post timing / flair conventions:** `[TBD — verify]`

---

## 5. Hugging Face community post stub

For the `unsloth/Qwen3.8-27B-GGUF` community tab, **only after the repo is
live**.

> Thanks to the Unsloth team for the UD quants — used four (UD-IQ2_XXS,
> UD-IQ2_S, UD-Q2_K_XL, UD-IQ3_XXS) to measure a HumanEval quality curve on a
> single RTX 3060 12GB with llama.cpp (build 10566, CUDA). Full 164-task pass@1
> at temp 0: 49.4% / 62.2% / 72.0% / 82.3% as GGUF size goes 6.76 → 10.17 GiB,
> about 10 points per GiB, every adjacent step significant under paired
> McNemar testing.
>
> Paper, raw per-run JSON and the harness (Apache-2.0): `[repo link]`. No ask,
> just sharing numbers in case they help anyone picking a quant for their VRAM.
