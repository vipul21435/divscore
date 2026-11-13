# Changelog

## 0.1.0

First release.

- Loaders for JSONL, JSON, CSV/TSV and plain text, with dotted and joined fields and chat-role selection
  (OpenAI-style and ShareGPT-style records).
- Token-budget metrics: distinct-n (per n-gram or per token), unigram and bigram entropy, gzip, bz2 and lzma
  compression ratios, Heaps exponent of the type-token curve.
- Text-budget metrics: self-BLEU (checked against nltk), ROUGE-L homogenisation (checked against rouge-score),
  n-gram Vendi score with a hashed cosine kernel, MinHash near-duplicate rate, top-10 opening share and Herfindahl
  index of three-word openings, share of texts holding one of the ten most frequent masked templates.
- Repeated subsampling at a fixed number of tokens or texts, with percentile intervals; optional truncation of every
  text to a fixed length.
- Commands `measure`, `compare`, `openings` and `curve` (plot with the `plot` extra), with Markdown and JSON output.
- Scripts that fetch four public instruction datasets and reproduce the README numbers.
