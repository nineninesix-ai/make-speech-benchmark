# Audio quality — `sim_ref`

A **reporting** metric, not a filter: nothing is rejected here, and no row is removed from any subset.

`would fail prompt QC %` applies the thresholds from `analysis/pilot_qc.py` — the ones prompts had to pass — to this audio stream. For `gt` and `sim_ref` that check was never run during the build (D-09), so the column states how much of the human anchor rests on audio that would not have been accepted as a prompt.

| lang   | audio   |    n |   OVRL |   SIG |   BAK |   OVRL p05 |   OVRL p95 |   clipping % |   narrowband % |   low speech % |   would fail prompt QC % |
|:-------|:--------|-----:|-------:|------:|------:|-----------:|-----------:|-------------:|---------------:|---------------:|-------------------------:|
| en-US  | sim_ref |  909 |  2.752 | 3.277 | 3.455 |      2.028 |      3.281 |            0 |           0.22 |           0    |                     0.22 |
| es-ES  | sim_ref | 1222 |  2.759 | 3.268 | 3.484 |      1.991 |      3.294 |            0 |           1.64 |           0.16 |                     1.8  |
| es-MX  | sim_ref | 1201 |  2.672 | 3.216 | 3.411 |      1.861 |      3.262 |            0 |           1.25 |           0    |                     1.25 |
| nl-NL  | sim_ref | 1319 |  2.804 | 3.287 | 3.562 |      2.082 |      3.302 |            0 |           1.59 |           0    |                     1.59 |
| pt-BR  | sim_ref |  983 |  2.697 | 3.208 | 3.455 |      2.049 |      3.288 |            0 |           0.61 |           0.61 |                     1.22 |
| ky     | sim_ref |  599 |  2.671 | 3.164 | 3.469 |      1.812 |      3.245 |            0 |           0.67 |           0    |                     0.67 |

