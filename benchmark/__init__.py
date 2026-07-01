"""
embedding_benchmark — A lightweight MTEB-inspired benchmark scaffold.
"""

from .models import EmbeddingModel, SentenceTransformerModel, load_model
from .tasks import Task, LikertContinuumWVSTask, get_tasks, TASK_REGISTRY
from .runner import BenchmarkRunner, TaskResult, results_to_dataframe, pivot_main_scores
from .cache import encode_with_cache

__all__ = [
    "EmbeddingModel",
    "SentenceTransformerModel",
    "load_model",
    "Task",
    "LikertContinuumWVSTask",
    "get_tasks",
    "TASK_REGISTRY",
    "BenchmarkRunner",
    "TaskResult",
    "results_to_dataframe",
    "pivot_main_scores",
    "encode_with_cache",
]
