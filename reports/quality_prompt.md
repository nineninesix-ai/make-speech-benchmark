# Audio quality — `prompt`

A **reporting** metric, not a filter: nothing is rejected here, and no row is removed from any subset.

`would fail prompt QC %` applies the thresholds from `analysis/pilot_qc.py` — the ones prompts had to pass — to this audio stream. For `gt` and `sim_ref` that check was never run during the build (D-09), so the column states how much of the human anchor rests on audio that would not have been accepted as a prompt.

| lang   | audio   |    n |   OVRL |   SIG |   BAK |   OVRL p05 |   OVRL p95 |   clipping % |   narrowband % |   low speech % |   would fail prompt QC % |   r(OVRL,SIM) |   rho |   n matched |
|:-------|:--------|-----:|-------:|------:|------:|-----------:|-----------:|-------------:|---------------:|---------------:|-------------------------:|--------------:|------:|------------:|
| en-US  | prompt  | 1500 |  2.761 | 3.303 | 3.432 |      1.971 |      3.319 |            0 |           0.27 |           0.27 |                     0.53 |         0.079 | 0.07  |         909 |
| es-ES  | prompt  | 1500 |  2.766 | 3.295 | 3.462 |      2.011 |      3.323 |            0 |           1.33 |           0.2  |                     1.53 |         0.049 | 0.082 |        1222 |
| es-MX  | prompt  | 1500 |  2.68  | 3.228 | 3.391 |      1.887 |      3.289 |            0 |           0.53 |           0    |                     0.53 |         0.124 | 0.123 |        1201 |
| nl-NL  | prompt  | 1500 |  2.808 | 3.316 | 3.523 |      2.152 |      3.317 |            0 |           0.47 |           0    |                     0.47 |         0.064 | 0.087 |        1319 |
| pt-BR  | prompt  | 1500 |  2.726 | 3.232 | 3.487 |      2.004 |      3.302 |            0 |           0.4  |           0    |                     0.4  |         0.114 | 0.03  |         983 |
| ky     | prompt  |  700 |  2.718 | 3.205 | 3.515 |      1.859 |      3.29  |            0 |           1.71 |           0    |                     1.71 |         0.091 | 0.033 |         599 |


## SIM anchor by reference-quality tier

Terciles of DNSMOS OVRL within each language.

| lang   |   high |    low |    mid |
|:-------|-------:|-------:|-------:|
| en-US  | 0.9341 | 0.9282 | 0.9328 |
| es-ES  | 0.941  | 0.9395 | 0.9499 |
| es-MX  | 0.9495 | 0.9397 | 0.9477 |
| ky     | 0.9311 | 0.9141 | 0.919  |
| nl-NL  | 0.9344 | 0.9272 | 0.924  |
| pt-BR  | 0.9399 | 0.9293 | 0.9325 |

