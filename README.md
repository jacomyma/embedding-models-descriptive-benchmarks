# Embedding Situational Benchmarks

A lightweight, MTEB-inspired framework for evaluating how well embedding models capture
graded, situational meaning — currently the WVS Likert-continuum benchmark.

## Structure

```
embedding_benchmark/
│
├── data/
│   └── WVS Statements.csv — World Values Survey statements + Likert codes
│
├── benchmark/
│   ├── __init__.py    — public API
│   ├── models.py      — adapters (HuggingFace/sentence-transformers)
│   ├── tasks.py       — benchmark tasks
│   ├── cache.py       — disk caching of embeddings
│   └── runner.py      — orchestrator + result helpers
│
├── experiments/       — exploratory notebooks
├── run-locally.ipynb  — run benchmark locally
└── run-in-colab.ipynb — run benchmark in Google Colab
```

## Quickstart

```bash
pip install sentence-transformers datasets scipy scikit-learn pandas tqdm
```

```python
from benchmark import BenchmarkRunner
from pathlib import Path

runner = BenchmarkRunner(
    model_configs=[
        {"type": "sentence_transformer", "model": "BAAI/bge-small-en-v1.5"},
        {"type": "sentence_transformer", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    ],
    task_names=["likert-wvs"],
    output_dir=Path("results"),
)

results = runner.run()
```

## Built-in Tasks

| Key           | Task class               | Metric              | Dataset                |
|---------------|---------------------------|---------------------|-------------------------|
| `likert-wvs`  | `LikertContinuumWVSTask` | mean \|Spearman ρ\| | WVS Statements (`data/`) |


## Adding a Model

Add a dict to `model_configs`:

```python
{"type": "sentence_transformer", "model": "intfloat/e5-large-v2", "device": "cuda"}
```


## Adding a Task

```python
from benchmark.tasks import Task, TASK_REGISTRY
from benchmark.cache import encode_with_cache
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MyTask(Task):
    name: str = "My-Task"

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        texts, labels = my_data_loader()
        embs = encode_with_cache(model, texts, self.name, cache_dir, **kwargs)
        score = my_eval_logic(embs, labels)
        return {"my_metric": score, "main_score": score}

TASK_REGISTRY["my-task"] = MyTask()
```


## Key Design Decisions

- **Cache first**: embeddings are cached to `.cache/embeddings/` keyed by model + dataset + text hash — re-running after a crash doesn't re-encode.
- **Write on completion**: each `(model, task)` result is written to `results/` immediately, so partial sweeps are recoverable.
- **No framework lock-in**: the `EmbeddingModel.encode()` interface is two lines; wrapping a new backend takes ~20 lines.
- **Config-driven**: models are plain dicts, easy to serialize, log, or generate programmatically.
