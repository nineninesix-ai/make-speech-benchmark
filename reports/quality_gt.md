# Audio quality — `gt`

A **reporting** metric, not a filter: nothing is rejected here, and no row is removed from any subset.

`would fail prompt QC %` applies the thresholds from `analysis/pilot_qc.py` — the ones prompts had to pass — to this audio stream. For `gt` and `sim_ref` that check was never run during the build (D-09), so the column states how much of the human anchor rests on audio that would not have been accepted as a prompt.

| lang   | audio   |    n |   OVRL |   SIG |   BAK |   OVRL p05 |   OVRL p95 |   clipping % |   narrowband % |   low speech % |   would fail prompt QC % |
|:-------|:--------|-----:|-------:|------:|------:|-----------:|-----------:|-------------:|---------------:|---------------:|-------------------------:|
| en-US  | gt      | 1500 |  2.766 | 3.276 | 3.474 |      1.968 |      3.323 |            0 |           1.47 |           0.93 |                     2.4  |
| es-ES  | gt      | 1499 |  2.8   | 3.283 | 3.544 |      2.041 |      3.34  |            0 |           2.33 |           0.4  |                     2.74 |
| es-MX  | gt      | 1500 |  2.662 | 3.181 | 3.413 |      1.834 |      3.279 |            0 |           2.73 |           1.8  |                     4.27 |
| nl-NL  | gt      | 1500 |  2.871 | 3.331 | 3.621 |      2.218 |      3.338 |            0 |           1.13 |           0.13 |                     1.27 |
| pt-BR  | gt      | 1500 |  2.783 | 3.243 | 3.58  |      2.005 |      3.307 |            0 |           2.47 |           1.07 |                     3.53 |
| ky     | gt      |  694 |  2.749 | 3.21  | 3.572 |      1.924 |      3.274 |            0 |           1.87 |           0.58 |                     2.31 |

