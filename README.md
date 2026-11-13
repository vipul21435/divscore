# divscore

Lexical and structural diversity of a text dataset, with every metric taken at a fixed token budget so that
datasets of different sizes can be compared.

Status: early. The `measure` command and the metrics listed under "How it works" run and are tested; the rest of
the planned metric set and the `compare`, `openings` and `curve` commands are not written yet (see "Not yet
implemented").

## Why this is hard

Almost every diversity number moves with the amount of text it is computed on. Distinct-n falls as a corpus grows,
because common n-grams keep repeating while new ones arrive more slowly; a compressor finds more repeats in more
text, so the compression ratio rises. A dataset of 50,000 records will look less diverse than one of 5,000 records
drawn from the same source, and a dataset of long answers less diverse than one of short ones, for no reason other
than size. A comparison is only fair when both sides are measured on the same amount of text.

## Install

```
pip install git+https://github.com/vipul21435/divscore
```

Python 3.10 or newer. The only runtime dependency is numpy.

## Quickstart

`examples/data/` holds two small files written for this repo: 30 varied questions and 30 prompts built from three
templates.

```
divscore measure examples/data/templated.jsonl --budget 250 --resamples 200
```

```
## templated.jsonl

30 texts, 294 tokens (9.8 per text).
Budget: 250 tokens, 200 subsamples, seed 0.

| metric | better | whole dataset | at budget | 95 % interval | sd |
|---|---|---:|---:|---|---:|
| distinct-1 | higher | 0.1497 | 0.1736 | [0.1640, 0.1760] | 0.0042 |
| distinct-2 | higher | 0.2538 | 0.2857 | [0.2723, 0.2946] | 0.0067 |
| distinct-3 | higher | 0.3376 | 0.3723 | [0.3535, 0.3850] | 0.0086 |
| entropy-1 | higher | 4.4062 | 4.3771 | [4.3360, 4.4216] | 0.0212 |
| entropy-2 | higher | 5.2183 | 5.1636 | [5.1234, 5.2017] | 0.0198 |
| cr-gzip | lower | 4.9568 | 4.3266 | [4.1762, 4.4852] | 0.0763 |
| cr-bz2 | lower | 4.0058 | 3.6529 | [3.5857, 3.7653] | 0.0452 |
```

At the same budget the varied file gives distinct-2 0.9705 [0.9524, 0.9870] and gzip ratio 1.6957
[1.6588, 1.7321] (`python examples/measure_examples.py` prints both tables). `--format json` writes the same
numbers as JSON; `--field`, `--role` and `--input-format` select the text in JSONL, JSON, CSV/TSV or plain text
files, including the turns of one role in chat records.

## How it works

**Fixed token budget.** Texts are tokenised into words and punctuation (lowercased unless `--keep-case`). One
subsample draws texts in random order without replacement until exactly `--budget` tokens are taken; the last text
is cut at the token where the budget is reached, and its raw text is cut at the same point. Every metric is computed
on the same subsample, the draw is repeated `--resamples` times, and the report gives the mean, the standard
deviation and the equal-tailed percentile interval across draws. That interval says how much the metric moves
between random samples of that size; it is not a confidence interval for the whole-dataset value. The
whole-dataset ("naive") value is printed beside it.

**Metrics so far.** N-grams never cross a text boundary.

- distinct-n: distinct n-grams over total n-grams, for n = 1, 2, 3. Li et al. (2016) state distinct-1 and distinct-2
  as distinct unigrams and bigrams "divided by total number of generated words"; that form is available in the API
  as `distinct_n(..., per="token")`. [A Diversity-Promoting Objective Function for Neural Conversation
  Models](https://arxiv.org/abs/1510.03055), NAACL 2016.
- entropy-1, entropy-2: Shannon entropy in bits of the unigram and bigram distribution.
- cr-gzip, cr-bz2: uncompressed bytes over compressed bytes of the texts joined by newlines (lzma is available in the
  API). Compression ratio as a diversity score is among the measures compared by Shaib et al.,
  [Standardizing the Measurement of Text Diversity](https://arxiv.org/abs/2403.00553) (2024).

## Not yet implemented

Self-BLEU and ROUGE-L homogenisation, the n-gram Vendi score, MinHash near-duplicate rate, opening concentration,
template counts, the type-token curve with a Heaps' law fit, length distribution, `divscore compare` with intervals on
the differences, `divscore openings`, `divscore curve`, and results on real public datasets. There are no measured
numbers on real data in this README yet.

## Limitations

- Tokenisation is a regular expression for words and punctuation. It suits languages written with spaces; it is not
  a subword tokeniser, so budgets are not comparable to model token counts.
- Cutting the last text of a subsample keeps the token count exact but leaves one partial text per draw.
- A budget larger than a dataset is refused rather than padded; the largest usable common budget is the size of the
  smaller dataset.

## Related tools

- [datasieve](https://github.com/vipul21435/datasieve) removes duplicates from training data; divscore only
  measures.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
