# v1 → v2 attribution ladder

Each column is one deliberate change, measured on identical audio, so the migration table has a cause per delta rather than one lump.

* **drift (env)** — replaying v1's exact decoding parameters on current library and hardware versions. Not a methodological change; reported so it is not mistaken for one.

* **D-02 penalty** — `repetition_penalty=1.1` removed. Positive means the reported anchor was flattered by the penalty.

* **D-10 chunking** — Whisper's long-form path switched off for clips of 2-6 s.

* **D-01 aggregation** — corpus-level instead of the macro mean. This is the largest term and it is a redefinition, not a correction.

| lang   | ASR     |   published v1 (macro) |   legacy (macro) |   n legacy |   nopen (macro) |   n nopen |   v2 (macro) |   n v2 |   v2 (corpus) | v2 corpus 95% CI   |   drift (env) |   D-02 penalty |   agree legacy/nopen |   D-10 chunking |   agree nopen/v2 |   D-01 aggregation |
|:-------|:--------|-----------------------:|-----------------:|-----------:|----------------:|----------:|-------------:|-------:|--------------:|:-------------------|--------------:|---------------:|---------------------:|----------------:|-----------------:|-------------------:|
| en-US  | whisper |                 0.0806 |           0.0805 |       1500 |          0.0807 |      1500 |       0.0807 |   1500 |        0.0774 | [0.0703, 0.0842]   |       -0.0001 |         0.0002 |               0.9893 |               0 |                1 |            -0.0033 |
| es-ES  | whisper |                 0.0485 |           0.0485 |       1499 |          0.0484 |      1499 |       0.0484 |   1499 |        0.0452 | [0.0397, 0.0512]   |        0      |        -0.0001 |               0.9907 |               0 |                1 |            -0.0032 |
| es-MX  | whisper |                 0.0654 |           0.0652 |       1500 |          0.0648 |      1500 |       0.0648 |   1500 |        0.0613 | [0.0541, 0.0691]   |       -0.0002 |        -0.0004 |               0.9867 |               0 |                1 |            -0.0035 |
| nl-NL  | whisper |                 0.0407 |           0.0407 |       1500 |          0.0408 |      1500 |       0.0408 |   1500 |        0.0385 | [0.0331, 0.0453]   |        0      |         0.0001 |               0.992  |               0 |                1 |            -0.0023 |
| pt-BR  | whisper |                 0.0823 |           0.0816 |       1500 |          0.0817 |      1500 |       0.0817 |   1500 |        0.0737 | [0.0608, 0.0884]   |       -0.0007 |         0.0001 |               0.9727 |               0 |                1 |            -0.008  |
| ky     | gigaam  |                 0.1104 |           0.1104 |        694 |          0.1104 |       694 |       0.1104 |    694 |        0.0998 | [0.0773, 0.1311]   |        0      |         0      |               1      |               0 |                1 |            -0.0106 |


## SIM anchor parity with v1

Same encoder, same stored audio, no decoding — this one is expected to reproduce exactly, and is the check that the audio path was not disturbed.

`distinct values` is the number the anchor is really measured on: every example of a speaker shares one prompt clip and one second clip, so the published `n` counts rows, not measurements (D-20).

| lang   |   published v1 |   v2 replay |   delta |   published n |   n rows |   distinct values |   speakers | matches to 4 dp   |
|:-------|---------------:|------------:|--------:|--------------:|---------:|------------------:|-----------:|:------------------|
| en-US  |         0.9317 |      0.9317 |      -0 |           909 |      909 |               629 |        631 | True              |
| es-ES  |         0.9435 |      0.9435 |      -0 |          1222 |     1222 |               587 |        590 | True              |
| es-MX  |         0.9456 |      0.9456 |       0 |          1201 |     1201 |               768 |        771 | True              |
| nl-NL  |         0.9285 |      0.9285 |       0 |          1319 |     1319 |               412 |        412 | True              |
| pt-BR  |         0.9339 |      0.9339 |       0 |           983 |      983 |               161 |        161 | True              |
| ky     |         0.9214 |      0.9214 |      -0 |           599 |      599 |               155 |        155 | True              |

