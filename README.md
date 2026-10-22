# Senova-4B

Text decision model based on Qwen3.5-4B. The service accepts `state/questions`
and returns typed answers with probabilities through `POST /v1/systemone`.
The same endpoint serves public and private evaluation inputs; it does not
load any benchmark files, gold labels or scoring metadata.

Training: supervised fine-tuning (SFT) only, using more than 5 million
training examples. The training mixture combines traditional information
retrieval (IR) data with hard examples constructed for decision-making datasets.
Base model: [Qwen/Qwen3.5-4B @ 851bf6e](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
The reference revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
the latest upstream main checked on 2026-10-22; it is not a verified record
of the training-time checkout. Model weights are Apache-2.0;
this inference repository retains its MIT license.

| Submission artifact | Link / revision |
|---|---|
| Model weights and tokenizer | [WangSenQWQ/Senova-4B](https://huggingface.co/WangSenQWQ/Senova-4B/tree/58bcb8a28cff3453b1fc401ae79beca3742601c8); revision `58bcb8a28cff3453b1fc401ae79beca3742601c8` |
| Inference code | [wswsddd/inference-code](https://github.com/wswsddd/inference-code); use the exact commit linked in the submission entry |
| Complete public results | [scores.json](https://huggingface.co/datasets/WangSenQWQ/Senova-4B-decision-index-results/blob/e9974586ec8b122ce5129bf714f5f8ba3a547bc7/runs/qwen3.5_4b_pointwise_v53_full_ck1047/scores.json) |

## 1. Install and download

Use the submitted code commit, Linux x86_64, Python 3.12 and an NVIDIA driver
compatible with the CUDA 13.0 Torch wheel. Target hardware is one RTX PRO 6000
(96 GB). Backend, frontend and evaluation client use one environment:

```bash
bash scripts/install_env.sh .venv
source .venv/bin/activate
```

Core versions are pinned in `requirements.txt`: SGLang 0.5.15, Torch 2.11.0
(observed build `2.11.0+cu130`), Transformers 5.12.1, sglang-kernel 0.4.4 and
FlashInfer 0.6.12. These are observed working versions, not a complete
transitive lockfile. A fresh installation and target-GPU execution still
require validation.

The model and results repositories are public. Download the pinned model
revision (no authentication required):

```bash
hf download WangSenQWQ/Senova-4B \
  --revision 58bcb8a28cff3453b1fc401ae79beca3742601c8 \
  --local-dir /path/to/model
```

Use the same downloaded directory for both commands below.

## 2. Start inference

Terminal 1, from the code repository:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_v53_submission.sh /path/to/model
```

Wait until the backend is ready:

```bash
curl --noproxy '*' -f http://127.0.0.1:8000/health
```

Terminal 2, from the same repository:

```bash
source .venv/bin/activate
python -m decision_index.systemone_server \
  --tokenizer /path/to/model \
  --backend-url http://127.0.0.1:8000/v1 \
  --host 127.0.0.1 --port 8008
```

Terminal 3, using the same environment:

```bash
source .venv/bin/activate
python scripts/smoke_test.py --url http://127.0.0.1:8008
```

The smoke test checks a synthetic `choice` and `noul` request and their returned
probabilities. It requires no evaluation data. Both services bind to localhost;
run the evaluator on this host. The frontend has no authentication.

## 3. Run private evaluation

Keep the private evaluator and its scoring unchanged. Configure its System One
endpoint as `http://127.0.0.1:8008` and model name as `Senova-4B`. Send each
private case's `state` and `questions` directly; do not send gold answers or
`_evaluation` metadata. No public suite directory is required.

The official kit's HTTP adapter can call this endpoint with:

```text
--engine http --option base_url=http://127.0.0.1:8008 --model Senova-4B
```

For an evaluator with its own request loop, this synthetic example demonstrates
the complete wire contract:

```python
import httpx

payload = {
    "model": "Senova-4B",
    "state": "The kettle is red.",
    "questions": {
        "color": {"type": "choice", "instructions": "What color is the kettle?",
                  "criteria": {"red": "Red", "blue": "Blue"}},
        "is_red": {"type": "noul", "instructions": "The kettle is red."},
    },
}
with httpx.Client(timeout=600, trust_env=False) as client:
    r = client.post("http://127.0.0.1:8008/v1/systemone", json=payload)
    r.raise_for_status()
    answers = r.json()["answers"]
    print(answers["color"]["choice"], answers["color"]["probabilities"])
    print(answers["is_red"]["noul"])  # P(true), in [0, 1]
```

Replace only the synthetic `state/questions` with private inputs. `choice`
returns an original option key and a probability for every original option;
`noul` returns P(true). Instructions may be a string or JSON object.

## 4. Exact inference settings and limits

- Single GPU, TP=1, DP=1; one running backend request; one frontend decision at
  a time. Weights use `dtype=auto` (checkpoint dtype: bfloat16), no quantization.
- Context: at most 262144 tokens per rendered question, including the requested
  output token; memory fraction 0.8; CUDA graphs use SGLang's default.
- Thinking is fixed off. Each question uses one next-token distribution from
  native SGLang `/generate`, temperature=0, max_new_tokens=1. Candidate logprobs
  are softmax-normalized over all candidates and mapped back to original keys.
- Supported types: `choice` and `noul`; text/JSON state only. Candidate capacity
  is 588 distinct verified single-token uppercase labels for this tokenizer.
  Each choice needs at least two candidates. HTTP request bodies are limited
  to 16 MiB. No state truncation or option dropping is performed.
- HTTP 413 denotes declared capacity failure; 422 invalid/unsupported input;
  502 backend failure; `GET /health` returns 503 until the backend is ready.
- Prompt version: `di-unified-letter-options-v4-ragtruth-train-aligned`;
  scoring version: `sglang-next-token-v1`. The complete prompt transformations
  are implemented in `decision_index/engines/sglang_engine.py`.

For official latency measurement, use one client, one request at a time on the
single GPU. Maintainers measure mean, median and P80; each must be at most
1000 ms. The public-run parallel timings are not this measurement.

## 5. Public evaluation record

Complete public suite 0.3: index 61.60; all 140178 scoreable requests succeeded
(140620 predictions before 442 exclusions). This is a public index, not the
official private Full score. The historical result uses its original model
identifier and run path; the release name is Senova-4B.

Historical execution: NVIDIA H100 80GB GPUs, eight endpoints, 64 client workers, TP=2 / DP=4 per
endpoint, context 262144, memory fraction 0.8, at most 32 running requests per
endpoint, disabled CUDA graphs, thinking off, unified letters, native logprobs.
Historical server environment: SGLang 0.5.15, Torch 2.11.0, Transformers 5.12.1,
sglang-kernel 0.4.4 and FlashInfer 0.6.12, using Python 3.12. These versions
were reconstructed from the environment selected by the original deployment
script and referenced by its startup logs; installed package metadata predates
the public run. A contemporaneous full package lockfile was not captured.
The current requirements describe the deployment environment separately.

For public reproduction, first rebuild/import the official suite as described
in `UPSTREAM_README.md`, then use the running frontend:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8008 --model Senova-4B \
  --suite-dir suite-0.3 --edition 0.3 --compact \
  --out runs/senova_4b_reproduction
```

## Validation and references

The shared environment passed 33 HTTP-contract and engine/runner tests. The
documented backend, frontend and smoke-test flow also passed with this package
and model on one H100 80GB (test ports 8016/8018). Synthetic tokenizer
fingerprints match the previous client stack. Full-suite single-GPU parity,
fresh installation and RTX PRO 6000 memory/latency remain unverified.

[Official submission requirements](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard)
and [engine interface](https://github.com/apolinario/decision-index/blob/main/docs/engines.md).
