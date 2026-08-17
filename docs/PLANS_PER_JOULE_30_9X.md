# 30.9× Plans/Joule — Evidence Record

This document records the evidence behind the projected **30.9× plans/joule**
headline. It is reproduced from the project's development history
(`docs/t6_spec.md`, `docs/me4_plan.md`, `docs/session_log.md`, and the
`gates/l3_fabric.py` gate) so the number is auditable from this repository.

## The claim, stated precisely

> **Algorithm-class, projected:** our discrete EBM + MPPI planner — projected
> from the certified simulator onto this chip's routed clock and frozen power
> envelope — achieves **30.9× plans/joule** and **2.8× latency** versus the
> best steelmanned run of the LeWM JEPA + CEM baseline on an NVIDIA T4.

Two boundary conditions are load-bearing and non-negotiable:

1. **This is an algorithm-class claim, not a chip-vs-GPU hardware claim.**
   The T4 runs the *opponent's* algorithm (LeWM JEPA + CEM). The chip's role
   is that our planner runs on its fabric. A direct chip-vs-GPU comparison is
   explicitly deferred to sampling-dominated, embedded-envelope scale (the
   `n·f/1e8` thermo crossover law), never at the 4-site toy scale.
2. **Our side is a projection until silicon bring-up (SG0).** The number is
   computed from mcycle measurements on the certified simulator, scaled to the
   routed 8 ns P-profile clock, with frozen power assumptions (5.0 W
   conservative / 2.5 W typical). SG0 converts the projection to measurement.

## The numbers

| Quantity | Value | Source |
|---|---|---|
| LeWM JEPA+CEM, stock anchor (T4) | 6.95 s/plan, 273.9 J net | L3 first read |
| LeWM JEPA+CEM, steelmanned fp32+inference_mode | 2.08 s/plan | L3b |
| LeWM JEPA+CEM, steelmanned fp16 tensor cores (T4) | 1.018 s/plan, 66.3 W → 67.5 J gross / **55.9 J net** | L3b |
| Our planner, certified-sim projection (8 ns clock, 5.0 W) | 362 ms/plan, **1.811 J** | L3 / `l3_fabric.py` |
| **Plans/joule ratio (55.9 / 1.811)** | **30.9×** | L3b close-out |
| Gross-pairing ratio (67.5 / 1.811) | 37.3× | L3b close-out |
| Latency ratio (1.018 / 0.362) | 2.8× | L3b close-out |

Per the frozen protocol, the earlier inflated 151× headline (against the stock
6.95 s anchor) is **retired**; the steelmanned 30.9× replaces it everywhere.
The methodology is "steelman-your-denominator": the opponent's baseline was
improved to its best measured case (fp16 tensor cores, `inference_mode`,
`torch.compile` showed no gain) before comparison, and our own side is stated
at the conservative 5 W assumption.

## How our number is computed

`gates/l3_fabric.py` (protocol frozen before any number was produced):

- One planning decision = R × NROLL × H = **24 × 5 × 8 = 960 model calls**
  plus an analytic proposal line (3 × 6 sweeps × 250 cycles) and a ×1.15
  margin for score arithmetic and refit.
- The planner kernel runs on the certified simulator; the measured cycle span
  is scaled by the routed 8 ns clock to a latency, then by the frozen power
  assumption (5.0 W conservative / 2.5 W typical) to joules.
- Outputs: `joule_per_plan` and `plans_per_joule` at both power points, labeled
  `"PROJECTION until SG0 (clock routed, power assumed)"`.

## The honest counter-read (L3c)

The same sitting measured a custom-CUDA port of **our own** algorithm at the
4-site D1 scale: **0.092 ms / 2.5 mJ gross per decision** (0.024 ms / 1.1 mJ
batched). At that toy scale a GPU beats our today-certified serial fabric
kernel by ~700×. This is disclosed deliberately: the 30.9× advantage is an
algorithm-class result, and the chip earns a hardware-level claim only where
sampling-dominated work saturates the fabric (S3+ scale, per the crossover
law), not at 4 sites.

## Claim boundary

- Supported wording: *projected 30.9× plans/joule (algorithm-class) for our
  EBM + MPPI planner vs the steelmanned LeWM JEPA + CEM baseline on an NVIDIA
  T4, pending SG0 measurement.*
- Do not use this number to imply a measured chip-vs-GPU hardware advantage at
  toy scale, or a same-task (D2) read. The D2 registration states that a
  same-task, full-system factor replaces this algorithm-class pairing as the
  headline when it lands.

## Provenance

Original evidence lives in the private development history: `docs/t6_spec.md`
(L3b/L3c read close-out, 2026-07-19), `docs/me4_plan.md`, `docs/session_log.md`
(L3b+L3c steelman reads), `docs/lewm_card.md`, and `gates/l3_fabric.py`. The
result JSONs (`ci/logs/wm/l3_fabric.json`, `l3c_gpu.json`) were produced by
running that gate and are not committed to this repository.
