# Decision gates — best-of-n experiment (2026-08-28)

Purpose: kill wasted GPU-hours early. Checked against partial results
(bestofn.py writes after every task).

## GATE 1 — early ceiling check (at ~50 tasks of IQ3_XXS run, ~2h in)
- CONTINUE if pass@5 − pass@1 ≥ 5 points (real headroom exists).
- KILL BOTH RUNS if pass@5 − pass@1 < 3 points: ceiling too low, retries
  don't diversify usefully at temp 0.8 → try temp 1.0 on 20 tasks before
  abandoning the direction entirely.

## GATE 2 — after IQ3_XXS completes (~6h)
- verified-selection ≥ 85%  → thesis CONFIRMED. Proceed: Q2_K_XL run +
  plan v2 (model writes its own tests to lift the 43% visible-test coverage).
- verified < 84% BUT pass@5 ≥ 88% → selection is the bottleneck, not the
  model. Skip Q2 run; build v2 self-tests first (selection fix is cheap,
  reruns are not).
- pass@5 < 86% → direction dead for this model. Stop, publish measurement
  findings as-is, move to franken-quant search.

## GATE 3 — after Q2_K_XL completes
- verified ≥ 80% → "compute substitutes for VRAM" claim holds (72%→80%+
  ≈ the 1-GiB step). Product thesis proven at two points.
- verified < 76% → the effect doesn't transfer down the curve; scope the
  claim to the top build only.

Standing rule: no new experiment starts before the previous gate is
evaluated and its verdict written here.

## GATE 1 VERDICT (2026-08-28, 51/164 tasks): CONTINUE
pass@1=0.678 pass@5=0.882 verified=0.882 headroom=+20.4pts (gate: >=5)
Selection lossless so far (verified==ceiling). Visible tests 48/51.

## GATE 2 VERDICT (2026-08-31, full 164): STOP THIS BRANCH
Final: pass@1=0.600 pass@5=0.750 verified=0.671 vs temp-0 baseline 0.823.
Matched analysis: of 29 temp-0 failures, only 3 recoverable by sampling; 15
temp-0 passes lost by sampling. Max anchored-v2 ceiling ~84.1% (+1.8) — not
worth GPU. MECHANISM: quantization noise + sampling noise compound; at ~3.2
bpw, temp-0.8 best-of-5 ceiling < greedy single shot. Novel negative result;
contradicts standard TTC playbook (measured on full-precision models).
Q2 run auto-killed by gate (saved ~6 GPU-h). NEXT: publish findings;
franken-quant search is the active experimental direction.
