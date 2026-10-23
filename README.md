# dlmbench

Code and samples for **[Hacking Generative Perplexity: Why Unconditional Text
Evaluation Needs New Metrics](https://arxiv.org/abs/2606.08417)**
by Antonio Franca and Alexander Tong.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/antoniofrancaib/dlmbench.git
cd dlmbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[paper]'
```

For neural metrics, install `.[neural]`; for sentence-encoder metrics, add
`.[encoders]`. A locked environment is available through `uv.lock`.

## Samples

All 80 generated sample banks are included: [67 OWT banks](samples/owt) and
[13 LM1B banks](samples/lm1b), with 1,024 samples each (318 MiB in total).
Each `samples/<dataset>/<bank>/samples.jsonl` contains one JSON object per line:

```json
{"id": 0, "text": "Generated text."}
```

From the repository directory:

```bash
dlmbench samples list --suite paper-main
dlmbench samples show lm1b_topk_iid_k32 --index 0
dlmbench samples verify lm1b_topk_iid_k32
```

The [catalog](src/dlmbench/resources/banks.json) records sample counts, checksums,
and generation metadata. Outside a clone, `dlmbench samples fetch <bank-id>`
downloads a bank from GitHub into `samples/`. Use `DLMBENCH_DATA_DIR` to choose
another parent directory. Samples are kept out of the pip package.

## Evaluate

Pass a sample-bank ID or a JSONL file with a `text` field:

```bash
dlmbench evaluate --list
dlmbench evaluate generated.jsonl --reference reference.jsonl \
  --metrics self_bleu ed mmd --out scores.json
```

These three metrics need no model downloads. Metric definitions and parameters
are in [protocols.py](src/dlmbench/metrics/protocols.py).

## Paper results

```bash
dlmbench reproduce --list
dlmbench reproduce all --out results/paper
```

This rebuilds tables and plots from the recorded scores. The two original
reference banks and raw human/LLM judgments are unavailable in this repository,
so it does not support complete recomputation of every paper result.

## Generate

```bash
dlmbench generate list
dlmbench generate show lm1b_langflow_1024_nfe
dlmbench generate run --help
dlmbench generate prepare-corpus --help
```

Naïve samplers use corpus counts or phrases and a local tokenizer. Learned
generators need an upstream checkout, checkpoint, and tokenizer. Use a separate
Python 3.11 environment with the matching [dependency lock](environments/);
upstream revisions and requirements are in the
[generator catalog](src/dlmbench/resources/generators/catalog.json).
Most learned adapters still need validation with real checkpoints. Historical
locks have known dependency advisories; load only trusted checkpoints, since
some formats use Python pickle.

## Tests

```bash
python -m pip install -e '.[paper,dev]'
pytest
ruff check src tests
```

## Citation

```bibtex
@misc{franca2026hackinggenerativeperplexity,
  title = {Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs New Metrics},
  author = {Antonio Franca and Alexander Tong},
  year = {2026},
  eprint = {2606.08417},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2606.08417}
}
```

[MIT licensed](LICENSE).
