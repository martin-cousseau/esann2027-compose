# Compute budget (100 Colab compute units, binding)

The Colab connector does not report the CU balance (`get_subscription` returns only the tier), and Google
publishes no per-GPU rate. Spending below is an **estimate**: planning rate x runtime lifetime (creation to
deletion, idle time included). **The author should check the real balance on the Colab website.**

## Rates (looked up 2026-11-09)

| Accelerator | Reported CU/hour | Date and kind of report | Source |
|---|---|---|---|
| T4 | 1.19 | measured March 2026 | https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025 ; https://gpuperhour.com/blog/google-colab-alternatives |
| L4 | 1.71 | measured March 2026 | same two pages; Fastero (Oct 27, 2026) implies about 1.67 |
| L4 | 1.7-1.8, then 1.94 | user reports, Oct 31 and Nov 2, 2026 (anecdotal) | https://github.com/googlecolab/colabtools/issues/6194 |
| A100 40 GB | 5.40; "close to 7" | March 2026; user report Oct 31, 2026 | Thunder/GPUperhour; colabtools#6194 |
| A100 80 GB | 7.52; 8.88-8.96 | March 2026; user reports Oct 29 - Nov 3, 2026 | Thunder/GPUperhour; colabtools#6194 |
| G4 (RTX PRO 6000) | 8.71; "more than 10" | March 2026; user report Nov 3, 2026 | Thunder; Fastero (https://fastero.com/blog/google-colab-pro); colabtools#6194 |

Since 29 October 2026 users report rates roughly a fifth to a third higher than before for the same GPU.

**Planning rate for L4: 1.94 CU/h (latest report) x 1.15 safety margin = 2.23 CU/h.**
G4 costs about 4.5-5x L4, above the "at most about twice L4" rule, so it was not piloted. A100 costs
3-5x L4 with lower fp32 throughput than L4 when TF32 is off, so it was not piloted either.

## Allocation and spending (updated after every runtime closes)

Since 2026-11-10 the author lets the 100 CU be allocated freely (DECISIONS.md #10); the table is indicative.

| Item | Allocated CU | Spent (est.) | Notes |
|---|---|---|---|
| Phase 0 and timing pilot | 4 | 0 | |
| E1 full grid, core models | 38 | 0 | |
| E2 same-Hessian ablation | 10 | 0 | |
| E3 mechanism runs | 12 | 0 | |
| E4 extra frontier points | 6 | 0 | |
| E5 seeds (4-bit cells) | 12 | 0 | |
| E6 SmolLM2-1.7B | 10 | 0 | |
| Reserve | 8 | 0 | |
| **Total** | **100** | **0** | |

## Runtime log (lifetime -> estimated CU at 2.23 CU/h)

| Runtime | Created (UTC) | Deleted (UTC) | Hours | Est. CU |
|---|---|---|---|---|
| esann-pilot | 2026-11-09 23:38 | reclaimed by Colab, likely ~00:10 (idle-client reclaim seen later) | ~0.6 (upper bound 6.2) | ~1.3 (upper bound 13.8) |
| esann-e1-a | 2026-11-10 05:53 | reclaimed ~06:33 | 0.67 | 1.5 |
| esann-e1-b | 2026-11-10 05:53 | reclaimed ~06:33 | 0.67 | 1.5 |
| esann-e1-a2 | 2026-11-10 06:41 | lost ~07:15 | 0.58 | 1.3 |
| esann-e1-b2 | 2026-11-10 06:41 | lost ~07:35 | 0.9 | 2.0 |
| esann-e1-a3 | 2026-11-10 07:15 | lost ~08:05 | 0.85 | 1.9 |
| esann-a4 | 2026-11-10 11:06 | lost ~15:50 (session suspended) | 4.75 | 10.6 |
| esann-b3 | 2026-11-10 11:06 | lost ~15:50 (session suspended) | 4.75 | 10.6 |
| esann-a5 | 2026-11-10 18:04 | 2026-11-11 07:21 (deleted, queues done) | 13.28 | 29.6 |
| esann-b4 | 2026-11-10 18:04 | 2026-11-11 08:48 (deleted, queues done) | 14.73 | 32.9 |

## Final account (2026-11-11, all runtimes deleted; `list_runtimes` returns none)

Total runtime lifetime (creation to deletion, idle time included): **41.8 L4-hours** (the pilot at its likely
0.6 h; 47.4 h if the pilot lived to its upper bound of 6.2 h).

| Rate (CU per L4-hour) | Basis | Estimated CU |
|---|---|---|
| 2.23 | planning rate (1.94 x 1.15 safety margin) | **93.2** (upper bound 105.7) |
| 1.94 | latest user report (colabtools#6194) | 81.1 (upper bound 91.9) |
| 1.71 | measured in March 2026 | 71.5 (upper bound 81.0) |

The Colab connector does not report the balance: **the author should read the real number on the Colab website.**
The GPU work stopped with all queues done; three cuts kept the plan near the stop line of about 95 CU at the
planning rate (DECISIONS.md #29 E5/E6 scope, #34-#35 E3 Qwen2.5-0.5B reduced to its two 50% cells).
