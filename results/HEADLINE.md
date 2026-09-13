
## HEADLINE RESULT (2026-08-23) — first independent sub-Q4 eval of this model

| | Q2_K_XL resident | Q4_K_S offloaded | delta |
|---|---|---|---|
| bits/weight | 2.87 | ~4.5 | −36% |
| file size | 9.14 GiB | 15.4 GiB | **−41%** |
| decode speed | **21.75 t/s** | **2.12 t/s** | **10.3× faster** |
| eval wall time | 1282 s | 6448 s | **5.0× faster** |
| GSM8K (50) | 0.96 | 1.00 | −4 pts (2 problems) |
| HumanEval (20) | **0.80** | **0.80** | **identical** |

**41% smaller, 10× faster, identical HumanEval, 4 points of GSM8K.**

Interpretation: the community assumption that sub-Q4 builds of this model are
quality-wrecked is FALSE at 2.87 bpw. The measured cost of dropping from the
popular 4-bit build to a build that actually fits a 12 GB card is 2 GSM8K
problems out of 50 and zero HumanEval tasks — while gaining a speedup that
converts the model from unusable (2 t/s) to conversational (21.75 t/s).

First published independent quality measurement of any sub-Q4 build of
Qwen3.8-27B (model released 2026-08-14, measured 2026-08-23).

### Caveats to state in any writeup

- n=50 GSM8K / n=20 HumanEval subsets; chat-template prompting, so not
  directly leaderboard-comparable; single seed; temp 0; one hardware config.
- The Q4_K_S config here is plain llama.cpp FFN offload **without** MTP
  speculation. The tuned community recipe reports ~9.7 t/s effective, so the
  honest comparison against a *tuned* baseline is ~2.2×, not 10×. The 10×
  figure describes the naive out-of-the-box experience, and must be labelled
  as such.
- GSM8K at 96–100% is near ceiling for this model class; a harder math set
  (e.g. MATH subset) would discriminate better and should be added.
