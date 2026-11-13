## dolly (A) vs alpaca (B)

A: 15011 texts, 221434 tokens. B: 52002 texts, 855114 tokens.
100 subsamples per side and budget, seed 0.

Budget: 20000 tokens

| metric | better | A | B | A - B | 95 % interval | whole data A - B | more diverse |
|---|---|---:|---:|---:|---|---:|---|
| distinct-1 | higher | 0.2253 | 0.1739 | 0.0514 | [0.0424, 0.0602] | 0.0632 | A |
| distinct-2 | higher | 0.6592 | 0.6196 | 0.0396 | [-0.0021, 0.0548] | 0.1920 | unclear |
| distinct-3 | higher | 0.8416 | 0.8703 | -0.0286 | [-0.0895, -0.0109] | 0.1458 | B |
| entropy-1 | higher | 9.1324 | 8.9020 | 0.2304 | [0.1544, 0.3127] | 0.3476 | A |
| entropy-2 | higher | 12.7973 | 12.7746 | 0.0226 | [-0.2454, 0.1173] | -0.2373 | unclear |
| cr-gzip | lower | 2.4802 | 2.6493 | -0.1691 | [-0.2131, -0.0980] | -0.2220 | A |
| cr-bz2 | lower | 2.8834 | 3.1797 | -0.2963 | [-0.3545, -0.2267] | -0.5479 | A |
| heaps-beta | higher | 0.7951 | 0.7597 | 0.0355 | [0.0060, 0.0642] | 0.0957 | A |

Budget: 500 texts

| metric | better | A | B | A - B | 95 % interval | whole data A - B | more diverse |
|---|---|---:|---:|---:|---|---:|---|
| self-bleu | lower | 0.2138 | 0.1776 | 0.0363 | [0.0173, 0.0600] | n/a | B |
| rouge-l | lower | 0.1126 | 0.1214 | -0.0087 | [-0.0228, 0.0037] | n/a | unclear |
| vendi | higher | 232.4960 | 246.9926 | -14.4966 | [-27.6811, -3.5682] | n/a | B |
| near-dup | lower | 0.0024 | 0.0000 | 0.0024 | [0.0000, 0.0080] | n/a | unclear |
| opening-top10 | lower | 0.2639 | 0.1133 | 0.1506 | [0.1029, 0.1850] | 0.1566 | B |
| opening-hhi | lower | 0.0118 | 0.0037 | 0.0081 | [0.0050, 0.0116] | 0.0082 | B |
| template-top10 | lower | 0.0821 | 0.0414 | 0.0407 | [0.0129, 0.0711] | n/a | B |
