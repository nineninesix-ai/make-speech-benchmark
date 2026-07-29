# Human anchors — v2

Every figure is a **human anchor**, not a ceiling. A synthesis system can and does exceed it: synthesis is conditioned on the prompt and inherits its recording channel, whereas the anchor compares two *different* recordings of the same person. The comparison is asymmetric by construction (D-11).

Intervals are 95 % cluster bootstraps resampling **speakers**, not rows. Rows are not independent: one prompt serves up to seven examples, and the SIM anchor is constant within a speaker by construction (D-20).

Anchor-normalised comparison is valid **within a language only**. Cross-language differences in the anchor are a property of the recogniser, and for `ky` it is a different recogniser entirely (D-19).


## WER anchor by recogniser

`WER corpus` is the headline: Σ(S+D+I) / Σ N_ref, the seed-tts-eval definition. `WER macro (v1)` is the mean of per-utterance rates, which is what v1 published; on texts with a median of 6-7 words the two differ substantially (D-01). `catastrophic` is the share of utterances above 50 % WER — the indicator that actually catches looping and dropped clauses.

| lang   | ASR     |    n |   speakers |   WER corpus | 95% CI           |   WER macro (v1) |   CER corpus | exact   | catastrophic   |   missing |   empty ref |
|:-------|:--------|-----:|-----------:|-------------:|:-----------------|-----------------:|-------------:|:--------|:---------------|----------:|------------:|
| en-US  | whisper | 1500 |       1012 |       0.0774 | [0.0703, 0.0842] |           0.0807 |       0.0296 | 59.7%   | 1.4%           |         0 |           0 |
| en-US  | mms     | 1500 |       1012 |       0.1784 | [0.1682, 0.1882] |           0.1845 |       0.0605 | 30.6%   | 6.0%           |         0 |           0 |
| es-ES  | whisper | 1499 |        574 |       0.0452 | [0.0397, 0.0512] |           0.0484 |       0.0154 | 73.0%   | 0.9%           |         1 |           0 |
| es-ES  | mms     | 1499 |        574 |       0.1248 | [0.1156, 0.1346] |           0.1307 |       0.0369 | 47.3%   | 3.7%           |         1 |           0 |
| es-MX  | whisper | 1500 |        639 |       0.0613 | [0.0541, 0.0691] |           0.0648 |       0.0214 | 68.3%   | 1.3%           |         0 |           0 |
| es-MX  | mms     | 1500 |        639 |       0.1482 | [0.1358, 0.1610] |           0.1557 |       0.0471 | 40.9%   | 5.1%           |         0 |           0 |
| nl-NL  | whisper | 1500 |        236 |       0.0385 | [0.0331, 0.0453] |           0.0408 |       0.0108 | 76.0%   | 0.5%           |         0 |           0 |
| nl-NL  | mms     | 1500 |        236 |       0.078  | [0.0682, 0.0896] |           0.0842 |       0.0239 | 58.1%   | 1.3%           |         0 |           0 |
| pt-BR  | whisper | 1500 |        229 |       0.0737 | [0.0608, 0.0884] |           0.0817 |       0.0234 | 71.5%   | 3.5%           |         0 |           0 |
| pt-BR  | mms     | 1500 |        229 |       0.2242 | [0.1984, 0.2529] |           0.2314 |       0.0618 | 36.1%   | 12.9%          |         0 |           0 |
| ky     | mms     |  694 |         89 |       0.2425 | [0.2086, 0.2830] |           0.2504 |       0.058  | 25.6%   | 10.7%          |         6 |           0 |
| ky     | gigaam  |  694 |         89 |       0.0998 | [0.0773, 0.1311] |           0.1104 |       0.0293 | 66.4%   | 5.5%           |         6 |           0 |


## Where the recognisers disagree

MMS decodes with CTC and no language model; Whisper's internal LM repairs mispronunciations. A large gap means the intelligible reading depends on the listener's expectations, not on the acoustics (G-03). Deltas are paired bootstraps over the utterances both recognisers scored.

| lang   | A       | B      |   WER A |   WER B |   delta (A−B) | 95% CI             | identical transcripts   |   n shared |
|:-------|:--------|:-------|--------:|--------:|--------------:|:-------------------|:------------------------|-----------:|
| en-US  | whisper | mms    |  0.0774 |  0.1784 |       -0.1011 | [-0.1086, -0.0934] | 32.0%                   |       1500 |
| es-ES  | whisper | mms    |  0.0452 |  0.1248 |       -0.0796 | [-0.0871, -0.0717] | 50.0%                   |       1499 |
| es-MX  | whisper | mms    |  0.0613 |  0.1482 |       -0.0869 | [-0.0952, -0.0788] | 43.0%                   |       1500 |
| nl-NL  | whisper | mms    |  0.0385 |  0.078  |       -0.0395 | [-0.0478, -0.0328] | 57.3%                   |       1500 |
| pt-BR  | whisper | mms    |  0.0737 |  0.2242 |       -0.1505 | [-0.1682, -0.1333] | 35.4%                   |       1500 |
| ky     | mms     | gigaam |  0.2425 |  0.0998 |        0.1427 | [0.1273, 0.1593]   | 24.8%                   |        694 |


## SIM anchor, impostor floor, and the usable range

`floor` is the mean cosine between prompts of **different** speakers — the score an impostor gets for free. `usable range` is anchor minus floor: the entire span in which a cloning system can distinguish itself. Where that span is narrow, an absolute SIM value carries almost no information, which is why v1's anchor-only presentation could not be read (G-02). `distinct values` is below `n rows` because every example of a speaker shares one prompt and one second clip.

| lang   | encoder   |   n rows |   speakers |   distinct values |   anchor | 95% CI           |   floor mean |   floor p95 |   floor pairs |   usable range | sim_norm formula      |
|:-------|:----------|---------:|-----------:|------------------:|---------:|:-----------------|-------------:|------------:|--------------:|---------------:|:----------------------|
| en-US  | wavlm_sv  |      909 |        631 |               629 |   0.9317 | [0.9283, 0.9347] |       0.6119 |      0.8707 |          5000 |         0.3198 | (x − 0.6119) / 0.3198 |
| en-US  | wavlm_ft  |      909 |        631 |               630 |   0.652  | [0.6424, 0.6601] |       0.0725 |      0.2483 |          5000 |         0.5795 | (x − 0.0725) / 0.5795 |
| en-US  | ecapa     |      909 |        631 |               631 |   0.6141 | [0.6042, 0.6227] |       0.0822 |      0.2447 |          5000 |         0.5319 | (x − 0.0822) / 0.5319 |
| es-ES  | wavlm_sv  |     1222 |        590 |               587 |   0.9435 | [0.9392, 0.9469] |       0.7286 |      0.9245 |          5000 |         0.2149 | (x − 0.7286) / 0.2149 |
| es-ES  | wavlm_ft  |     1222 |        590 |               590 |   0.6888 | [0.6791, 0.6980] |       0.1302 |      0.3183 |          5000 |         0.5586 | (x − 0.1302) / 0.5586 |
| es-ES  | ecapa     |     1222 |        590 |               589 |   0.6505 | [0.6405, 0.6599] |       0.132  |      0.3206 |          5000 |         0.5185 | (x − 0.1320) / 0.5185 |
| es-MX  | wavlm_sv  |     1201 |        771 |               768 |   0.9456 | [0.9432, 0.9480] |       0.7315 |      0.932  |          5000 |         0.2141 | (x − 0.7315) / 0.2141 |
| es-MX  | wavlm_ft  |     1201 |        771 |               769 |   0.6882 | [0.6811, 0.6956] |       0.1685 |      0.3576 |          5000 |         0.5197 | (x − 0.1685) / 0.5197 |
| es-MX  | ecapa     |     1201 |        771 |               769 |   0.635  | [0.6268, 0.6438] |       0.122  |      0.299  |          5000 |         0.5131 | (x − 0.1220) / 0.5131 |
| nl-NL  | wavlm_sv  |     1319 |        412 |               412 |   0.9285 | [0.9235, 0.9331] |       0.7446 |      0.9215 |          5000 |         0.184  | (x − 0.7446) / 0.1840 |
| nl-NL  | wavlm_ft  |     1319 |        412 |               412 |   0.6708 | [0.6619, 0.6797] |       0.174  |      0.352  |          5000 |         0.4969 | (x − 0.1740) / 0.4969 |
| nl-NL  | ecapa     |     1319 |        412 |               412 |   0.6232 | [0.6134, 0.6336] |       0.1866 |      0.3666 |          5000 |         0.4366 | (x − 0.1866) / 0.4366 |
| pt-BR  | wavlm_sv  |      983 |        161 |               161 |   0.9339 | [0.9265, 0.9403] |       0.765  |      0.9248 |          5000 |         0.1689 | (x − 0.7650) / 0.1689 |
| pt-BR  | wavlm_ft  |      983 |        161 |               161 |   0.6346 | [0.6166, 0.6502] |       0.1864 |      0.3754 |          5000 |         0.4482 | (x − 0.1864) / 0.4482 |
| pt-BR  | ecapa     |      983 |        161 |               161 |   0.5763 | [0.5578, 0.5930] |       0.1257 |      0.2993 |          5000 |         0.4506 | (x − 0.1257) / 0.4506 |
| ky     | wavlm_sv  |      599 |        155 |               155 |   0.9214 | [0.9121, 0.9302] |       0.7173 |      0.9279 |          5000 |         0.204  | (x − 0.7173) / 0.2040 |
| ky     | wavlm_ft  |      599 |        155 |               155 |   0.6069 | [0.5849, 0.6267] |       0.184  |      0.3837 |          5000 |         0.4229 | (x − 0.1840) / 0.4229 |
| ky     | ecapa     |      599 |        155 |               155 |   0.5593 | [0.5371, 0.5811] |       0.1484 |      0.3428 |          5000 |         0.411  | (x − 0.1484) / 0.4110 |


## SIM breakdowns

Every cell carries `n` and the number of distinct speakers behind it. `small cell` marks fewer than 30 speakers, where the mean should not be read as a trend. `second-clip duration` is the control v1 lacks: the card bins by prompt duration only, leaving the other clip in the pair — down to ~2 s — uncontrolled (D-21).

| lang   | encoder   | cut                  | value   |    SIM | 95% CI           |    n |   speakers | small cell   |
|:-------|:----------|:---------------------|:--------|-------:|:-----------------|-----:|-----------:|:-------------|
| en-US  | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9284 | [0.9225, 0.9338] |  301 |        204 | False        |
| en-US  | wavlm_sv  | prompt duration      | <3.7    | 0.9228 | [0.9137, 0.9308] |  149 |        112 | False        |
| en-US  | wavlm_sv  | prompt duration      | >=4.5   | 0.9367 | [0.9322, 0.9406] |  459 |        315 | False        |
| en-US  | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.948  | [0.9435, 0.9526] |  166 |        133 | False        |
| en-US  | wavlm_sv  | second-clip duration | <3.5    | 0.9268 | [0.9227, 0.9306] |  686 |        449 | False        |
| en-US  | wavlm_sv  | second-clip duration | >=4.5   | 0.9432 | [0.9316, 0.9533] |   57 |         49 | False        |
| en-US  | wavlm_sv  | gender               | female  | 0.9301 | [0.9257, 0.9341] |  556 |        278 | False        |
| en-US  | wavlm_sv  | gender               | male    | 0.9317 | [0.9255, 0.9378] |  197 |        197 | False        |
| en-US  | wavlm_sv  | gender               | unknown | 0.9372 | [0.9301, 0.9436] |  156 |        156 | False        |
| en-US  | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6414 | [0.6263, 0.6556] |  301 |        204 | False        |
| en-US  | wavlm_ft  | prompt duration      | <3.7    | 0.6265 | [0.6069, 0.6458] |  149 |        112 | False        |
| en-US  | wavlm_ft  | prompt duration      | >=4.5   | 0.6672 | [0.6532, 0.6797] |  459 |        315 | False        |
| en-US  | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.698  | [0.6805, 0.7141] |  166 |        133 | False        |
| en-US  | wavlm_ft  | second-clip duration | <3.5    | 0.6368 | [0.6267, 0.6471] |  686 |        449 | False        |
| en-US  | wavlm_ft  | second-clip duration | >=4.5   | 0.701  | [0.6640, 0.7347] |   57 |         49 | False        |
| en-US  | wavlm_ft  | gender               | female  | 0.6352 | [0.6221, 0.6478] |  556 |        278 | False        |
| en-US  | wavlm_ft  | gender               | male    | 0.6623 | [0.6463, 0.6771] |  197 |        197 | False        |
| en-US  | wavlm_ft  | gender               | unknown | 0.699  | [0.6822, 0.7148] |  156 |        156 | False        |
| en-US  | ecapa     | prompt duration      | 3.7-4.5 | 0.6003 | [0.5841, 0.6158] |  301 |        204 | False        |
| en-US  | ecapa     | prompt duration      | <3.7    | 0.5754 | [0.5551, 0.5958] |  149 |        112 | False        |
| en-US  | ecapa     | prompt duration      | >=4.5   | 0.6358 | [0.6218, 0.6487] |  459 |        315 | False        |
| en-US  | ecapa     | second-clip duration | 3.5-4.5 | 0.6632 | [0.6457, 0.6804] |  166 |        133 | False        |
| en-US  | ecapa     | second-clip duration | <3.5    | 0.5985 | [0.5879, 0.6089] |  686 |        449 | False        |
| en-US  | ecapa     | second-clip duration | >=4.5   | 0.66   | [0.6189, 0.6975] |   57 |         49 | False        |
| en-US  | ecapa     | gender               | female  | 0.6026 | [0.5886, 0.6155] |  556 |        278 | False        |
| en-US  | ecapa     | gender               | male    | 0.6145 | [0.5978, 0.6306] |  197 |        197 | False        |
| en-US  | ecapa     | gender               | unknown | 0.655  | [0.6366, 0.6726] |  156 |        156 | False        |
| es-ES  | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9476 | [0.9438, 0.9511] |  492 |        240 | False        |
| es-ES  | wavlm_sv  | prompt duration      | <3.7    | 0.9321 | [0.9169, 0.9435] |  263 |        130 | False        |
| es-ES  | wavlm_sv  | prompt duration      | >=4.5   | 0.9456 | [0.9409, 0.9501] |  467 |        220 | False        |
| es-ES  | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.9528 | [0.9466, 0.9588] |  175 |         78 | False        |
| es-ES  | wavlm_sv  | second-clip duration | <3.5    | 0.9413 | [0.9363, 0.9454] | 1005 |        494 | False        |
| es-ES  | wavlm_sv  | second-clip duration | >=4.5   | 0.9572 | [0.9518, 0.9628] |   42 |         18 | True         |
| es-ES  | wavlm_sv  | gender               | female  | 0.9424 | [0.9347, 0.9488] |  507 |        169 | False        |
| es-ES  | wavlm_sv  | gender               | male    | 0.9444 | [0.9405, 0.9478] |  652 |        384 | False        |
| es-ES  | wavlm_sv  | gender               | unknown | 0.9426 | [0.9332, 0.9519] |   63 |         37 | False        |
| es-ES  | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6981 | [0.6860, 0.7096] |  492 |        240 | False        |
| es-ES  | wavlm_ft  | prompt duration      | <3.7    | 0.6411 | [0.6159, 0.6637] |  263 |        130 | False        |
| es-ES  | wavlm_ft  | prompt duration      | >=4.5   | 0.7059 | [0.6911, 0.7200] |  467 |        220 | False        |
| es-ES  | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.7383 | [0.7186, 0.7568] |  175 |         78 | False        |
| es-ES  | wavlm_ft  | second-clip duration | <3.5    | 0.6777 | [0.6667, 0.6882] | 1005 |        494 | False        |
| es-ES  | wavlm_ft  | second-clip duration | >=4.5   | 0.747  | [0.7133, 0.7790] |   42 |         18 | True         |
| es-ES  | wavlm_ft  | gender               | female  | 0.6924 | [0.6749, 0.7083] |  507 |        169 | False        |
| es-ES  | wavlm_ft  | gender               | male    | 0.6865 | [0.6756, 0.6967] |  652 |        384 | False        |
| es-ES  | wavlm_ft  | gender               | unknown | 0.6833 | [0.6562, 0.7075] |   63 |         37 | False        |
| es-ES  | ecapa     | prompt duration      | 3.7-4.5 | 0.6615 | [0.6488, 0.6733] |  492 |        240 | False        |
| es-ES  | ecapa     | prompt duration      | <3.7    | 0.6014 | [0.5746, 0.6270] |  263 |        130 | False        |
| es-ES  | ecapa     | prompt duration      | >=4.5   | 0.6667 | [0.6510, 0.6818] |  467 |        220 | False        |
| es-ES  | ecapa     | second-clip duration | 3.5-4.5 | 0.7017 | [0.6805, 0.7219] |  175 |         78 | False        |
| es-ES  | ecapa     | second-clip duration | <3.5    | 0.6393 | [0.6278, 0.6501] | 1005 |        494 | False        |
| es-ES  | ecapa     | second-clip duration | >=4.5   | 0.7062 | [0.6677, 0.7408] |   42 |         18 | True         |
| es-ES  | ecapa     | gender               | female  | 0.6518 | [0.6329, 0.6697] |  507 |        169 | False        |
| es-ES  | ecapa     | gender               | male    | 0.6503 | [0.6388, 0.6610] |  652 |        384 | False        |
| es-ES  | ecapa     | gender               | unknown | 0.6433 | [0.6132, 0.6722] |   63 |         37 | False        |
| es-MX  | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9455 | [0.9418, 0.9490] |  425 |        279 | False        |
| es-MX  | wavlm_sv  | prompt duration      | <3.7    | 0.9377 | [0.9304, 0.9447] |  234 |        160 | False        |
| es-MX  | wavlm_sv  | prompt duration      | >=4.5   | 0.9492 | [0.9460, 0.9524] |  542 |        332 | False        |
| es-MX  | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.9519 | [0.9468, 0.9568] |  248 |        149 | False        |
| es-MX  | wavlm_sv  | second-clip duration | <3.5    | 0.9432 | [0.9404, 0.9459] |  897 |        590 | False        |
| es-MX  | wavlm_sv  | second-clip duration | >=4.5   | 0.9574 | [0.9474, 0.9654] |   56 |         32 | False        |
| es-MX  | wavlm_sv  | gender               | female  | 0.9411 | [0.9374, 0.9446] |  492 |        246 | False        |
| es-MX  | wavlm_sv  | gender               | male    | 0.9478 | [0.9445, 0.9507] |  648 |        488 | False        |
| es-MX  | wavlm_sv  | gender               | unknown | 0.9592 | [0.9519, 0.9658] |   61 |         37 | False        |
| es-MX  | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6835 | [0.6715, 0.6959] |  425 |        279 | False        |
| es-MX  | wavlm_ft  | prompt duration      | <3.7    | 0.6569 | [0.6381, 0.6740] |  234 |        160 | False        |
| es-MX  | wavlm_ft  | prompt duration      | >=4.5   | 0.7054 | [0.6958, 0.7156] |  542 |        332 | False        |
| es-MX  | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.7173 | [0.7001, 0.7331] |  248 |        149 | False        |
| es-MX  | wavlm_ft  | second-clip duration | <3.5    | 0.6774 | [0.6699, 0.6856] |  897 |        590 | False        |
| es-MX  | wavlm_ft  | second-clip duration | >=4.5   | 0.7319 | [0.7032, 0.7599] |   56 |         32 | False        |
| es-MX  | wavlm_ft  | gender               | female  | 0.6759 | [0.6640, 0.6878] |  492 |        246 | False        |
| es-MX  | wavlm_ft  | gender               | male    | 0.6944 | [0.6857, 0.7030] |  648 |        488 | False        |
| es-MX  | wavlm_ft  | gender               | unknown | 0.7218 | [0.6966, 0.7449] |   61 |         37 | False        |
| es-MX  | ecapa     | prompt duration      | 3.7-4.5 | 0.632  | [0.6183, 0.6458] |  425 |        279 | False        |
| es-MX  | ecapa     | prompt duration      | <3.7    | 0.5996 | [0.5812, 0.6170] |  234 |        160 | False        |
| es-MX  | ecapa     | prompt duration      | >=4.5   | 0.6528 | [0.6404, 0.6649] |  542 |        332 | False        |
| es-MX  | ecapa     | second-clip duration | 3.5-4.5 | 0.6665 | [0.6475, 0.6848] |  248 |        149 | False        |
| es-MX  | ecapa     | second-clip duration | <3.5    | 0.6218 | [0.6125, 0.6314] |  897 |        590 | False        |
| es-MX  | ecapa     | second-clip duration | >=4.5   | 0.7084 | [0.6751, 0.7353] |   56 |         32 | False        |
| es-MX  | ecapa     | gender               | female  | 0.6237 | [0.6096, 0.6375] |  492 |        246 | False        |
| es-MX  | ecapa     | gender               | male    | 0.6401 | [0.6309, 0.6497] |  648 |        488 | False        |
| es-MX  | ecapa     | gender               | unknown | 0.6723 | [0.6417, 0.7005] |   61 |         37 | False        |
| nl-NL  | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9309 | [0.9232, 0.9376] |  384 |        119 | False        |
| nl-NL  | wavlm_sv  | prompt duration      | <3.7    | 0.925  | [0.9171, 0.9316] |  727 |        227 | False        |
| nl-NL  | wavlm_sv  | prompt duration      | >=4.5   | 0.9367 | [0.9274, 0.9449] |  208 |         66 | False        |
| nl-NL  | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.9431 | [0.9263, 0.9563] |   70 |         22 | True         |
| nl-NL  | wavlm_sv  | second-clip duration | <3.5    | 0.9279 | [0.9228, 0.9325] | 1242 |        388 | False        |
| nl-NL  | wavlm_sv  | second-clip duration | >=4.5   | 0.8895 | [0.8412, 0.9538] |    7 |          2 | True         |
| nl-NL  | wavlm_sv  | gender               | female  | 0.9393 | [0.9302, 0.9477] |  236 |         59 | False        |
| nl-NL  | wavlm_sv  | gender               | male    | 0.9271 | [0.9222, 0.9319] |  888 |        289 | False        |
| nl-NL  | wavlm_sv  | gender               | unknown | 0.9219 | [0.8989, 0.9385] |  195 |         64 | False        |
| nl-NL  | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6878 | [0.6714, 0.7045] |  384 |        119 | False        |
| nl-NL  | wavlm_ft  | prompt duration      | <3.7    | 0.6513 | [0.6392, 0.6624] |  727 |        227 | False        |
| nl-NL  | wavlm_ft  | prompt duration      | >=4.5   | 0.7078 | [0.6877, 0.7274] |  208 |         66 | False        |
| nl-NL  | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.718  | [0.6925, 0.7418] |   70 |         22 | True         |
| nl-NL  | wavlm_ft  | second-clip duration | <3.5    | 0.6681 | [0.6593, 0.6770] | 1242 |        388 | False        |
| nl-NL  | wavlm_ft  | second-clip duration | >=4.5   | 0.6734 | [0.6291, 0.7324] |    7 |          2 | True         |
| nl-NL  | wavlm_ft  | gender               | female  | 0.6722 | [0.6556, 0.6880] |  236 |         59 | False        |
| nl-NL  | wavlm_ft  | gender               | male    | 0.6709 | [0.6594, 0.6820] |  888 |        289 | False        |
| nl-NL  | wavlm_ft  | gender               | unknown | 0.669  | [0.6367, 0.6953] |  195 |         64 | False        |
| nl-NL  | ecapa     | prompt duration      | 3.7-4.5 | 0.6375 | [0.6193, 0.6554] |  384 |        119 | False        |
| nl-NL  | ecapa     | prompt duration      | <3.7    | 0.6067 | [0.5931, 0.6197] |  727 |        227 | False        |
| nl-NL  | ecapa     | prompt duration      | >=4.5   | 0.6545 | [0.6288, 0.6792] |  208 |         66 | False        |
| nl-NL  | ecapa     | second-clip duration | 3.5-4.5 | 0.6803 | [0.6468, 0.7129] |   70 |         22 | True         |
| nl-NL  | ecapa     | second-clip duration | <3.5    | 0.6203 | [0.6099, 0.6301] | 1242 |        388 | False        |
| nl-NL  | ecapa     | second-clip duration | >=4.5   | 0.5781 | [0.4919, 0.6930] |    7 |          2 | True         |
| nl-NL  | ecapa     | gender               | female  | 0.6319 | [0.6074, 0.6552] |  236 |         59 | False        |
| nl-NL  | ecapa     | gender               | male    | 0.6202 | [0.6086, 0.6318] |  888 |        289 | False        |
| nl-NL  | ecapa     | gender               | unknown | 0.6266 | [0.5940, 0.6540] |  195 |         64 | False        |
| pt-BR  | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9491 | [0.9411, 0.9566] |  220 |         36 | False        |
| pt-BR  | wavlm_sv  | prompt duration      | <3.7    | 0.9262 | [0.9143, 0.9360] |  579 |         95 | False        |
| pt-BR  | wavlm_sv  | prompt duration      | >=4.5   | 0.9399 | [0.9300, 0.9493] |  184 |         30 | False        |
| pt-BR  | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.9451 | [0.9278, 0.9581] |   80 |         13 | True         |
| pt-BR  | wavlm_sv  | second-clip duration | <3.5    | 0.9327 | [0.9244, 0.9395] |  861 |        141 | False        |
| pt-BR  | wavlm_sv  | second-clip duration | >=4.5   | 0.9376 | [0.9166, 0.9589] |   42 |          7 | True         |
| pt-BR  | wavlm_sv  | gender               | female  | 0.9447 | [0.9311, 0.9561] |  125 |         18 | True         |
| pt-BR  | wavlm_sv  | gender               | male    | 0.9339 | [0.9252, 0.9415] |  582 |         97 | False        |
| pt-BR  | wavlm_sv  | gender               | unknown | 0.929  | [0.9111, 0.9420] |  276 |         46 | False        |
| pt-BR  | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6904 | [0.6666, 0.7151] |  220 |         36 | False        |
| pt-BR  | wavlm_ft  | prompt duration      | <3.7    | 0.6055 | [0.5829, 0.6270] |  579 |         95 | False        |
| pt-BR  | wavlm_ft  | prompt duration      | >=4.5   | 0.6595 | [0.6220, 0.6972] |  184 |         30 | False        |
| pt-BR  | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.6277 | [0.5625, 0.6882] |   80 |         13 | True         |
| pt-BR  | wavlm_ft  | second-clip duration | <3.5    | 0.6301 | [0.6133, 0.6471] |  861 |        141 | False        |
| pt-BR  | wavlm_ft  | second-clip duration | >=4.5   | 0.7393 | [0.6872, 0.7857] |   42 |          7 | True         |
| pt-BR  | wavlm_ft  | gender               | female  | 0.6378 | [0.5943, 0.6800] |  125 |         18 | True         |
| pt-BR  | wavlm_ft  | gender               | male    | 0.6294 | [0.6073, 0.6516] |  582 |         97 | False        |
| pt-BR  | wavlm_ft  | gender               | unknown | 0.6442 | [0.6135, 0.6737] |  276 |         46 | False        |
| pt-BR  | ecapa     | prompt duration      | 3.7-4.5 | 0.6325 | [0.6062, 0.6562] |  220 |         36 | False        |
| pt-BR  | ecapa     | prompt duration      | <3.7    | 0.5509 | [0.5272, 0.5721] |  579 |         95 | False        |
| pt-BR  | ecapa     | prompt duration      | >=4.5   | 0.5893 | [0.5441, 0.6362] |  184 |         30 | False        |
| pt-BR  | ecapa     | second-clip duration | 3.5-4.5 | 0.6004 | [0.5357, 0.6629] |   80 |         13 | True         |
| pt-BR  | ecapa     | second-clip duration | <3.5    | 0.5701 | [0.5515, 0.5889] |  861 |        141 | False        |
| pt-BR  | ecapa     | second-clip duration | >=4.5   | 0.6576 | [0.5710, 0.7411] |   42 |          7 | True         |
| pt-BR  | ecapa     | gender               | female  | 0.5912 | [0.5372, 0.6387] |  125 |         18 | True         |
| pt-BR  | ecapa     | gender               | male    | 0.5815 | [0.5578, 0.6042] |  582 |         97 | False        |
| pt-BR  | ecapa     | gender               | unknown | 0.5587 | [0.5265, 0.5922] |  276 |         46 | False        |
| ky     | wavlm_sv  | prompt duration      | 3.7-4.5 | 0.9321 | [0.9237, 0.9401] |  200 |         52 | False        |
| ky     | wavlm_sv  | prompt duration      | <3.7    | 0.9167 | [0.9010, 0.9296] |  296 |         76 | False        |
| ky     | wavlm_sv  | prompt duration      | >=4.5   | 0.9138 | [0.8778, 0.9402] |  103 |         27 | True         |
| ky     | wavlm_sv  | second-clip duration | 3.5-4.5 | 0.911  | [0.8429, 0.9509] |   43 |         11 | True         |
| ky     | wavlm_sv  | second-clip duration | <3.5    | 0.9223 | [0.9131, 0.9302] |  540 |        140 | False        |
| ky     | wavlm_sv  | second-clip duration | >=4.5   | 0.9174 | [0.8106, 0.9779] |   16 |          4 | True         |
| ky     | wavlm_sv  | gender               | female  | 0.9124 | [0.8776, 0.9377] |  108 |         27 | True         |
| ky     | wavlm_sv  | gender               | male    | 0.9213 | [0.9080, 0.9340] |  230 |         62 | False        |
| ky     | wavlm_sv  | gender               | unknown | 0.9251 | [0.9115, 0.9361] |  261 |         66 | False        |
| ky     | wavlm_ft  | prompt duration      | 3.7-4.5 | 0.6381 | [0.6138, 0.6620] |  200 |         52 | False        |
| ky     | wavlm_ft  | prompt duration      | <3.7    | 0.5887 | [0.5557, 0.6165] |  296 |         76 | False        |
| ky     | wavlm_ft  | prompt duration      | >=4.5   | 0.5985 | [0.5272, 0.6620] |  103 |         27 | True         |
| ky     | wavlm_ft  | second-clip duration | 3.5-4.5 | 0.5928 | [0.4915, 0.6679] |   43 |         11 | True         |
| ky     | wavlm_ft  | second-clip duration | <3.5    | 0.6039 | [0.5833, 0.6244] |  540 |        140 | False        |
| ky     | wavlm_ft  | second-clip duration | >=4.5   | 0.745  | [0.7124, 0.7707] |   16 |          4 | True         |
| ky     | wavlm_ft  | gender               | female  | 0.6212 | [0.5592, 0.6716] |  108 |         27 | True         |
| ky     | wavlm_ft  | gender               | male    | 0.5986 | [0.5713, 0.6244] |  230 |         62 | False        |
| ky     | wavlm_ft  | gender               | unknown | 0.6083 | [0.5727, 0.6408] |  261 |         66 | False        |
| ky     | ecapa     | prompt duration      | 3.7-4.5 | 0.5979 | [0.5674, 0.6271] |  200 |         52 | False        |
| ky     | ecapa     | prompt duration      | <3.7    | 0.5416 | [0.5106, 0.5701] |  296 |         76 | False        |
| ky     | ecapa     | prompt duration      | >=4.5   | 0.5354 | [0.4691, 0.5938] |  103 |         27 | True         |
| ky     | ecapa     | second-clip duration | 3.5-4.5 | 0.5691 | [0.4773, 0.6375] |   43 |         11 | True         |
| ky     | ecapa     | second-clip duration | <3.5    | 0.5564 | [0.5344, 0.5784] |  540 |        140 | False        |
| ky     | ecapa     | second-clip duration | >=4.5   | 0.6337 | [0.5814, 0.6743] |   16 |          4 | True         |
| ky     | ecapa     | gender               | female  | 0.5622 | [0.5055, 0.6127] |  108 |         27 | True         |
| ky     | ecapa     | gender               | male    | 0.5574 | [0.5311, 0.5836] |  230 |         62 | False        |
| ky     | ecapa     | gender               | unknown | 0.5598 | [0.5234, 0.5955] |  261 |         66 | False        |


## WER breakdowns

Same rules: `n`, speakers, and a flag on thin cells.

| lang   | ASR     | cut           | value   |   WER corpus | 95% CI           |   n |   speakers | small cell   |
|:-------|:--------|:--------------|:--------|-------------:|:-----------------|----:|-----------:|:-------------|
| en-US  | whisper | target length | 10-12   |       0.076  | [0.0655, 0.0868] | 450 |        377 | False        |
| en-US  | whisper | target length | 13-16   |       0.0697 | [0.0590, 0.0814] | 345 |        306 | False        |
| en-US  | whisper | target length | 17+     |       0.0466 | [0.0261, 0.0704] |  30 |         29 | False        |
| en-US  | whisper | target length | 3-5     |       0.0933 | [0.0690, 0.1185] | 225 |        220 | False        |
| en-US  | whisper | target length | 6-9     |       0.0903 | [0.0755, 0.1054] | 450 |        387 | False        |
| en-US  | whisper | gender        | female  |       0.0803 | [0.0714, 0.0901] | 676 |        565 | False        |
| en-US  | whisper | gender        | male    |       0.0712 | [0.0596, 0.0832] | 442 |        386 | False        |
| en-US  | whisper | gender        | unknown |       0.0794 | [0.0670, 0.0927] | 382 |        335 | False        |
| en-US  | mms     | target length | 10-12   |       0.1803 | [0.1635, 0.1970] | 450 |        377 | False        |
| en-US  | mms     | target length | 13-16   |       0.1656 | [0.1513, 0.1802] | 345 |        306 | False        |
| en-US  | mms     | target length | 17+     |       0.1121 | [0.0769, 0.1518] |  30 |         29 | False        |
| en-US  | mms     | target length | 3-5     |       0.1897 | [0.1552, 0.2255] | 225 |        220 | False        |
| en-US  | mms     | target length | 6-9     |       0.201  | [0.1805, 0.2210] | 450 |        387 | False        |
| en-US  | mms     | gender        | female  |       0.1902 | [0.1746, 0.2052] | 676 |        565 | False        |
| en-US  | mms     | gender        | male    |       0.1586 | [0.1439, 0.1749] | 442 |        386 | False        |
| en-US  | mms     | gender        | unknown |       0.1809 | [0.1627, 0.2009] | 382 |        335 | False        |
| es-ES  | whisper | target length | 10-12   |       0.0371 | [0.0301, 0.0449] | 450 |        286 | False        |
| es-ES  | whisper | target length | 13-16   |       0.0443 | [0.0359, 0.0537] | 375 |        244 | False        |
| es-ES  | whisper | target length | 3-5     |       0.0858 | [0.0615, 0.1105] | 225 |        172 | False        |
| es-ES  | whisper | target length | 6-9     |       0.0472 | [0.0372, 0.0579] | 449 |        295 | False        |
| es-ES  | whisper | gender        | female  |       0.0509 | [0.0416, 0.0602] | 569 |        362 | False        |
| es-ES  | whisper | gender        | male    |       0.0419 | [0.0360, 0.0482] | 815 |        433 | False        |
| es-ES  | whisper | gender        | unknown |       0.0403 | [0.0253, 0.0576] | 115 |        102 | False        |
| es-ES  | mms     | target length | 10-12   |       0.1219 | [0.1069, 0.1369] | 450 |        286 | False        |
| es-ES  | mms     | target length | 13-16   |       0.1196 | [0.1052, 0.1341] | 375 |        244 | False        |
| es-ES  | mms     | target length | 3-5     |       0.1864 | [0.1473, 0.2237] | 225 |        172 | False        |
| es-ES  | mms     | target length | 6-9     |       0.1198 | [0.1034, 0.1381] | 449 |        295 | False        |
| es-ES  | mms     | gender        | female  |       0.1373 | [0.1220, 0.1513] | 569 |        362 | False        |
| es-ES  | mms     | gender        | male    |       0.1154 | [0.1040, 0.1272] | 815 |        433 | False        |
| es-ES  | mms     | gender        | unknown |       0.1282 | [0.0998, 0.1570] | 115 |        102 | False        |
| es-MX  | whisper | target length | 10-12   |       0.0609 | [0.0496, 0.0726] | 450 |        301 | False        |
| es-MX  | whisper | target length | 13-16   |       0.0533 | [0.0425, 0.0652] | 375 |        276 | False        |
| es-MX  | whisper | target length | 3-5     |       0.0889 | [0.0688, 0.1123] | 225 |        174 | False        |
| es-MX  | whisper | target length | 6-9     |       0.0663 | [0.0537, 0.0801] | 450 |        304 | False        |
| es-MX  | whisper | gender        | female  |       0.0637 | [0.0530, 0.0763] | 562 |        376 | False        |
| es-MX  | whisper | gender        | male    |       0.06   | [0.0509, 0.0699] | 835 |        492 | False        |
| es-MX  | whisper | gender        | unknown |       0.059  | [0.0414, 0.0773] | 103 |         95 | False        |
| es-MX  | mms     | target length | 10-12   |       0.1443 | [0.1289, 0.1607] | 450 |        301 | False        |
| es-MX  | mms     | target length | 13-16   |       0.1393 | [0.1212, 0.1591] | 375 |        276 | False        |
| es-MX  | mms     | target length | 3-5     |       0.2186 | [0.1827, 0.2539] | 225 |        174 | False        |
| es-MX  | mms     | target length | 6-9     |       0.1476 | [0.1269, 0.1695] | 450 |        304 | False        |
| es-MX  | mms     | gender        | female  |       0.1526 | [0.1350, 0.1726] | 562 |        376 | False        |
| es-MX  | mms     | gender        | male    |       0.145  | [0.1305, 0.1592] | 835 |        492 | False        |
| es-MX  | mms     | gender        | unknown |       0.1501 | [0.1176, 0.1840] | 103 |         95 | False        |
| nl-NL  | whisper | target length | 10-12   |       0.0378 | [0.0288, 0.0485] | 450 |        104 | False        |
| nl-NL  | whisper | target length | 13-16   |       0.0331 | [0.0259, 0.0409] | 345 |         82 | False        |
| nl-NL  | whisper | target length | 17+     |       0.034  | [0.0148, 0.0558] |  30 |         28 | False        |
| nl-NL  | whisper | target length | 3-5     |       0.0492 | [0.0332, 0.0665] | 225 |         83 | False        |
| nl-NL  | whisper | target length | 6-9     |       0.0443 | [0.0335, 0.0577] | 450 |        139 | False        |
| nl-NL  | whisper | gender        | female  |       0.0323 | [0.0238, 0.0420] | 272 |         92 | False        |
| nl-NL  | whisper | gender        | male    |       0.0396 | [0.0333, 0.0466] | 994 |        196 | False        |
| nl-NL  | whisper | gender        | unknown |       0.0409 | [0.0296, 0.0543] | 234 |         91 | False        |
| nl-NL  | mms     | target length | 10-12   |       0.0706 | [0.0568, 0.0867] | 450 |        104 | False        |
| nl-NL  | mms     | target length | 13-16   |       0.0639 | [0.0538, 0.0760] | 345 |         82 | False        |
| nl-NL  | mms     | target length | 17+     |       0.1002 | [0.0601, 0.1424] |  30 |         28 | False        |
| nl-NL  | mms     | target length | 3-5     |       0.1079 | [0.0820, 0.1370] | 225 |         83 | False        |
| nl-NL  | mms     | target length | 6-9     |       0.0952 | [0.0788, 0.1164] | 450 |        139 | False        |
| nl-NL  | mms     | gender        | female  |       0.0736 | [0.0591, 0.0904] | 272 |         92 | False        |
| nl-NL  | mms     | gender        | male    |       0.0777 | [0.0666, 0.0908] | 994 |        196 | False        |
| nl-NL  | mms     | gender        | unknown |       0.0844 | [0.0685, 0.1040] | 234 |         91 | False        |
| pt-BR  | whisper | target length | 10-12   |       0.0654 | [0.0486, 0.0844] | 300 |        124 | False        |
| pt-BR  | whisper | target length | 13-16   |       0.0483 | [0.0340, 0.0656] | 150 |         82 | False        |
| pt-BR  | whisper | target length | 3-5     |       0.0941 | [0.0709, 0.1229] | 525 |        143 | False        |
| pt-BR  | whisper | target length | 6-9     |       0.0829 | [0.0635, 0.1042] | 525 |        159 | False        |
| pt-BR  | whisper | gender        | female  |       0.0525 | [0.0350, 0.0718] | 132 |         78 | False        |
| pt-BR  | whisper | gender        | male    |       0.0782 | [0.0623, 0.0955] | 774 |        184 | False        |
| pt-BR  | whisper | gender        | unknown |       0.0728 | [0.0588, 0.0896] | 594 |        161 | False        |
| pt-BR  | mms     | target length | 10-12   |       0.2129 | [0.1812, 0.2503] | 300 |        124 | False        |
| pt-BR  | mms     | target length | 13-16   |       0.1802 | [0.1511, 0.2135] | 150 |         82 | False        |
| pt-BR  | mms     | target length | 3-5     |       0.23   | [0.1932, 0.2728] | 525 |        143 | False        |
| pt-BR  | mms     | target length | 6-9     |       0.2537 | [0.2165, 0.2935] | 525 |        159 | False        |
| pt-BR  | mms     | gender        | female  |       0.1962 | [0.1556, 0.2424] | 132 |         78 | False        |
| pt-BR  | mms     | gender        | male    |       0.2329 | [0.2023, 0.2650] | 774 |        184 | False        |
| pt-BR  | mms     | gender        | unknown |       0.2195 | [0.1906, 0.2540] | 594 |        161 | False        |
| ky     | mms     | target length | 10-12   |       0.2241 | [0.1740, 0.2888] |  70 |         28 | False        |
| ky     | mms     | target length | 3-5     |       0.279  | [0.2374, 0.3289] | 240 |         41 | False        |
| ky     | mms     | target length | 6-9     |       0.2326 | [0.1980, 0.2772] | 384 |         73 | False        |
| ky     | mms     | gender        | female  |       0.2675 | [0.2221, 0.3284] | 119 |         32 | False        |
| ky     | mms     | gender        | male    |       0.2383 | [0.1971, 0.2882] | 274 |         60 | False        |
| ky     | mms     | gender        | unknown |       0.2364 | [0.2022, 0.2766] | 301 |         52 | False        |
| ky     | gigaam  | target length | 10-12   |       0.0823 | [0.0475, 0.1294] |  70 |         28 | False        |
| ky     | gigaam  | target length | 3-5     |       0.1647 | [0.1255, 0.2175] | 240 |         41 | False        |
| ky     | gigaam  | target length | 6-9     |       0.0784 | [0.0563, 0.1075] | 384 |         73 | False        |
| ky     | gigaam  | gender        | female  |       0.1261 | [0.0803, 0.1876] | 119 |         32 | False        |
| ky     | gigaam  | gender        | male    |       0.0977 | [0.0732, 0.1278] | 274 |         60 | False        |
| ky     | gigaam  | gender        | unknown |       0.0913 | [0.0634, 0.1277] | 301 |         52 | False        |


## Audio quality of the three streams

`would fail prompt QC %` applies the prompt-acceptance thresholds to each stream. Prompts pass by construction. `gt` and `sim_ref` never faced those filters during the build (D-09), so this column measures how much of the human anchor rests on audio that would have been rejected as a prompt. Reported, not filtered: removing rows would break `utt` stability with v1.

| lang   | audio   |    n |   DNSMOS OVRL |   OVRL p05 |   clipping % |   narrowband % |   low speech % |   would fail prompt QC % |
|:-------|:--------|-----:|--------------:|-----------:|-------------:|---------------:|---------------:|-------------------------:|
| en-US  | prompt  | 1500 |         2.761 |      1.971 |            0 |           0.27 |           0.27 |                     0.53 |
| en-US  | gt      | 1500 |         2.766 |      1.968 |            0 |           1.47 |           0.93 |                     2.4  |
| en-US  | sim_ref |  909 |         2.752 |      2.028 |            0 |           0.22 |           0    |                     0.22 |
| es-ES  | prompt  | 1500 |         2.766 |      2.011 |            0 |           1.33 |           0.2  |                     1.53 |
| es-ES  | gt      | 1499 |         2.8   |      2.041 |            0 |           2.33 |           0.4  |                     2.74 |
| es-ES  | sim_ref | 1222 |         2.759 |      1.991 |            0 |           1.64 |           0.16 |                     1.8  |
| es-MX  | prompt  | 1500 |         2.68  |      1.887 |            0 |           0.53 |           0    |                     0.53 |
| es-MX  | gt      | 1500 |         2.662 |      1.834 |            0 |           2.73 |           1.8  |                     4.27 |
| es-MX  | sim_ref | 1201 |         2.672 |      1.861 |            0 |           1.25 |           0    |                     1.25 |
| nl-NL  | prompt  | 1500 |         2.808 |      2.152 |            0 |           0.47 |           0    |                     0.47 |
| nl-NL  | gt      | 1500 |         2.871 |      2.218 |            0 |           1.13 |           0.13 |                     1.27 |
| nl-NL  | sim_ref | 1319 |         2.804 |      2.082 |            0 |           1.59 |           0    |                     1.59 |
| pt-BR  | prompt  | 1500 |         2.726 |      2.004 |            0 |           0.4  |           0    |                     0.4  |
| pt-BR  | gt      | 1500 |         2.783 |      2.005 |            0 |           2.47 |           1.07 |                     3.53 |
| pt-BR  | sim_ref |  983 |         2.697 |      2.049 |            0 |           0.61 |           0.61 |                     1.22 |
| ky     | prompt  |  700 |         2.718 |      1.859 |            0 |           1.71 |           0    |                     1.71 |
| ky     | gt      |  694 |         2.749 |      1.924 |            0 |           1.87 |           0.58 |                     2.31 |
| ky     | sim_ref |  599 |         2.671 |      1.812 |            0 |           0.67 |           0    |                     0.67 |

