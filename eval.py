#!/usr/bin/env python3
"""Production-ready ML model benchmarking with statistical rigor."""

import argparse
import json
import time
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime

# XTYLearner imports
from xtylearner.models import get_model, get_model_names
from xtylearner.data import get_dataset
from xtylearner.training import Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class BenchmarkDataBundle:
    """Reusable data structures for benchmark iterations."""

    train_loader: DataLoader
    val_loader: DataLoader
    x_dim: int
    y_dim: int


@dataclass
class BenchmarkResult:
    """Structured benchmark result with metadata."""
    name: str
    value: float
    unit: str
    range: List[float]  # min, max for confidence intervals
    samples: int
    timestamp: str
    commit: str
    environment: Dict[str, Any]


class ModelBenchmarker:
    """Production benchmarking with statistical methods for XTYLearner models."""
    
    def __init__(self, config_path: str = "benchmark_config.json"):
        self.config = self._load_config(config_path)
        self.results = []
        torch.set_num_threads(self.config.get("torch_num_threads", 1))
        self._data_cache: Dict[Tuple[str, int], BenchmarkDataBundle] = {}
        self._timing_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
        cache_root = os.environ.get("XTYLEARNER_CACHE_DIR", ".cache/xtylearner")
        self._cache_dir = Path(cache_root)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration with defaults tailored for XTYLearner."""
        defaults = {
            "iterations": 5,
            "warmup_iterations": 2,
            "confidence_level": 0.95,
            "metrics": ["val_outcome_rmse", "val_treatment_accuracy", "train_time_seconds"],
            "models": ["cycle_dual", "mean_teacher", "prob_circuit", "ganite", "flow_ssc"],
            "datasets": ["synthetic", "synthetic_mixed"],
            "statistical_method": "bootstrap",
            "training_epochs": 10,
            "sample_size": 100,
            "random_seed": 42
        }
        
        if Path(path).exists():
            with open(path) as f:
                config = json.load(f)
                return {**defaults, **config}
        return defaults

    def _prepare_data(self, dataset_name: str) -> BenchmarkDataBundle:
        """Create (or reuse) dataset splits and data loaders."""

        cache_key = (dataset_name, self.config["sample_size"])
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        dataset_kwargs: Dict[str, Any] = {"n_samples": self.config["sample_size"]}

        # Add dataset-specific parameters
        if dataset_name in ["synthetic", "synthetic_mixed"]:
            dataset_kwargs["d_x"] = 2
        if dataset_name == "synthetic_mixed":
            dataset_kwargs.update({"label_ratio": 0.5})
        elif dataset_name == "criteo_uplift":
            dataset_kwargs.update({"prefer_real": False, "seed": 42})
        elif dataset_name == "nhefs":
            dataset_kwargs.update({"seed": 42})

        cache_params = {
            "dataset": dataset_name,
            "n_samples": dataset_kwargs["n_samples"],
        }
        if "d_x" in dataset_kwargs:
            cache_params["d_x"] = dataset_kwargs["d_x"]
        if dataset_name == "synthetic_mixed":
            cache_params["label_ratio"] = dataset_kwargs["label_ratio"]
        elif dataset_name == "criteo_uplift":
            cache_params["prefer_real"] = dataset_kwargs["prefer_real"]
            cache_params["seed"] = dataset_kwargs["seed"]
        elif dataset_name == "nhefs":
            cache_params["seed"] = dataset_kwargs["seed"]

        cache_hash = hashlib.sha1(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:12]
        dataset_cache_dir = self._cache_dir / "datasets"
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_cache_dir / f"{cache_hash}.pt"

        if cache_file.exists():
            payload = torch.load(cache_file)
            train_tensors = tuple(payload["train"])
            val_tensors = tuple(payload["val"])
        else:
            dataset_seed = np.random.randint(0, 10000)
            dataset_kwargs["seed"] = dataset_seed
            full_ds = get_dataset(dataset_name, **dataset_kwargs)

            half = len(full_ds) // 2
            train_tensors = tuple(t[:half].clone() for t in full_ds.tensors)
            val_tensors = tuple(t[half:].clone() for t in full_ds.tensors)

            tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
            torch.save(
                {
                    "train": train_tensors,
                    "val": val_tensors,
                    "metadata": {"seed": dataset_seed, "params": cache_params},
                },
                tmp_file,
            )
            tmp_file.replace(cache_file)

        train_ds = TensorDataset(*train_tensors)
        val_ds = TensorDataset(*val_tensors)

        train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=10)

        bundle = BenchmarkDataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            x_dim=train_ds.tensors[0].size(1),
            y_dim=train_ds.tensors[1].size(1),
        )
        self._data_cache[cache_key] = bundle
        return bundle
    
    def benchmark_model(self, model_name: str, dataset_name: str) -> List[BenchmarkResult]:
        """Run benchmarks with multiple passes for statistical accuracy."""
        results = []
        np.random.seed(self.config["random_seed"])
        torch.manual_seed(self.config["random_seed"])

        print(f"Benchmarking {model_name} on {dataset_name}...")

        timing_record: Dict[str, Any] = {
            "model": model_name,
            "dataset": dataset_name,
        }

        prep_start = time.perf_counter()
        data_bundle = self._prepare_data(dataset_name)
        timing_record["data_prep_seconds"] = time.perf_counter() - prep_start

        # Warmup runs (not counted)
        warmup_iterations = self.config["warmup_iterations"]
        if warmup_iterations:
            print(f"Running {warmup_iterations} warmup iterations...")
            warmup_start = time.perf_counter()
            for i in range(warmup_iterations):
                print(f"  Warmup {i+1}/{warmup_iterations}")
                self._run_warmup_pass(model_name, data_bundle)
            timing_record["warmup_seconds"] = time.perf_counter() - warmup_start
        else:
            print("Skipping warmup iterations (configured as 0).")
            timing_record["warmup_seconds"] = 0.0

        # Actual measurements
        print(f"Running {self.config['iterations']} measurement iterations...")
        metric_samples = {metric: [] for metric in self.config["metrics"]}

        iteration_start = time.perf_counter()
        cumulative_train_time = 0.0
        for i in range(self.config["iterations"]):
            print(f"  Iteration {i+1}/{self.config['iterations']}")
            iteration_results = self._run_single_benchmark(model_name, data_bundle)
            
            for metric in self.config["metrics"]:
                if metric in iteration_results:
                    metric_samples[metric].append(iteration_results[metric])
            train_time_value = iteration_results.get("train_time_seconds")
            if isinstance(train_time_value, (int, float)) and not np.isnan(train_time_value):
                cumulative_train_time += float(train_time_value)

        timing_record["benchmark_seconds"] = time.perf_counter() - iteration_start
        timing_record["train_time_seconds"] = cumulative_train_time
        timing_record["total_seconds"] = (
            timing_record["data_prep_seconds"]
            + timing_record["warmup_seconds"]
            + timing_record["benchmark_seconds"]
        )

        # Calculate statistics for each metric
        for metric_name, samples in metric_samples.items():
            if not samples:
                continue
                
            mean_val = np.mean(samples)
            confidence_interval = self._calculate_confidence_interval(samples)
            
            result = BenchmarkResult(
                name=f"{model_name}_{dataset_name}_{metric_name}",
                value=mean_val,
                unit=self._get_unit(metric_name),
                range=[confidence_interval[0], confidence_interval[1]],
                samples=len(samples),
                timestamp=datetime.utcnow().isoformat(),
                commit=os.environ.get("GITHUB_SHA", "local"),
                environment={
                    "runner_os": os.environ.get("RUNNER_OS", "unknown"),
                    "python_version": sys.version.split()[0],
                    "torch_version": torch.__version__,
                    "model_name": model_name,
                    "dataset_name": dataset_name
                }
            )
            results.append(result)
            print(f"    {metric_name}: {mean_val:.4f} ± {(confidence_interval[1] - confidence_interval[0])/2:.4f}")
        
        self._timing_info[(model_name, dataset_name)] = timing_record
        return results
    
    def _build_model_components(
        self, model_name: str, data_bundle: BenchmarkDataBundle
    ) -> Tuple[Any, Any]:
        """Instantiate a model and its optimiser based on ``model_name``."""

        model_kwargs = {"d_x": data_bundle.x_dim, "d_y": data_bundle.y_dim, "k": 2}

        if model_name == "lp_knn":
            model_kwargs["n_neighbors"] = 3
        elif model_name == "ctm_t":
            model_kwargs = {"d_in": data_bundle.x_dim + data_bundle.y_dim + 1}

        model = get_model(model_name, **model_kwargs)
        lr = getattr(model, "default_lr", 0.001)

        if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
            gen_params: List[torch.nn.Parameter] = []
            disc_params: List[torch.nn.Parameter] = []

            if hasattr(model, "generator_parameters"):
                gen_params = [
                    p for p in model.generator_parameters() if p.requires_grad
                ]
            if hasattr(model, "discriminator_parameters"):
                disc_params = [
                    p for p in model.discriminator_parameters() if p.requires_grad
                ]

            if not gen_params or not disc_params:
                named_params = list(model.named_parameters())
                if not gen_params:
                    gen_params = [
                        p
                        for name, p in named_params
                        if p.requires_grad and not name.startswith("D_")
                    ]
                if not disc_params:
                    disc_params = [
                        p
                        for name, p in named_params
                        if p.requires_grad and name.startswith("D_")
                    ]

            if not gen_params:
                gen_params = [
                    p
                    for p in getattr(model, "parameters", lambda: [])()
                    if p.requires_grad
                ]
            if not disc_params:
                disc_params = [
                    p
                    for p in getattr(model, "parameters", lambda: [])()
                    if p.requires_grad
                ]

            optimizer = (
                torch.optim.Adam(gen_params, lr=lr),
                torch.optim.Adam(disc_params, lr=lr),
            )
        else:
            params = [
                p
                for p in getattr(model, "parameters", lambda: [])()
                if p.requires_grad
            ]
            if not params:
                params = [torch.zeros(1, requires_grad=True)]
            optimizer = torch.optim.Adam(params, lr=lr)

        return model, optimizer

    def _run_warmup_pass(
        self, model_name: str, data_bundle: BenchmarkDataBundle
    ) -> None:
        """Perform a lightweight warmup to stabilise kernels and data pipelines."""

        try:
            model, optimizer = self._build_model_components(model_name, data_bundle)
            trainer = Trainer(
                model,
                optimizer,
                data_bundle.train_loader,
                val_loader=data_bundle.val_loader,
                logger=None,
            )
            trainer.fit(1)
            trainer.evaluate(data_bundle.val_loader)
        except Exception as exc:
            print(f"    Warmup skipped due to error: {exc}")

    def _run_single_benchmark(
        self, model_name: str, data_bundle: BenchmarkDataBundle
    ) -> Dict[str, float]:
        """Run a single benchmark iteration and return metrics."""
        try:
            model, opt = self._build_model_components(model_name, data_bundle)

            # Train and evaluate
            start_time = time.perf_counter()
            trainer = Trainer(
                model,
                opt,
                data_bundle.train_loader,
                val_loader=data_bundle.val_loader,
                logger=None,
            )
            trainer.fit(self.config["training_epochs"])
            train_time = time.perf_counter() - start_time

            breakdown = getattr(model, "loss_breakdown", None)
            if breakdown:
                formatted = ", ".join(
                    f"{name}={value.item():.4f}" if hasattr(value, "item") else f"{name}={value}"
                    for name, value in breakdown.items()
                )
                print(f"    Loss breakdown (last batch): {formatted}")

            # Get metrics
            val_metrics = trainer.evaluate(data_bundle.val_loader)
            
            # Return standardized metrics
            return {
                "val_outcome_rmse": val_metrics.get("outcome rmse", float("nan")),
                "val_treatment_accuracy": val_metrics.get("treatment accuracy", float("nan")),
                "train_time_seconds": train_time
            }
            
        except Exception as e:
            print(f"    Error in benchmark: {e}")
            return {
                "val_outcome_rmse": float("nan"),
                "val_treatment_accuracy": float("nan"),
                "train_time_seconds": float("nan")
            }
    
    def _calculate_confidence_interval(self, samples: List[float]) -> Tuple[float, float]:
        """Bootstrap confidence interval calculation."""
        if len(samples) < 2:
            return (samples[0], samples[0]) if samples else (0, 0)
        
        # Remove NaN values
        clean_samples = [s for s in samples if not np.isnan(s)]
        if not clean_samples:
            return (0, 0)

        if self.config["statistical_method"] == "bootstrap":
            n_bootstrap = 1000
            sample_array = np.asarray(clean_samples, dtype=float)
            resamples = np.random.choice(
                sample_array,
                size=(n_bootstrap, sample_array.size),
                replace=True,
            )
            bootstrap_means = resamples.mean(axis=1)

            alpha = 1 - self.config["confidence_level"]
            lower = np.percentile(bootstrap_means, alpha/2 * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
            return (lower, upper)
        else:
            # Simple std-based interval
            mean_val = np.mean(clean_samples)
            std_val = np.std(clean_samples)
            margin = 1.96 * std_val / np.sqrt(len(clean_samples))  # 95% CI
            return (mean_val - margin, mean_val + margin)
    
    def _get_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric."""
        if "rmse" in metric_name:
            return "rmse"
        elif "accuracy" in metric_name:
            return "accuracy"
        elif "time" in metric_name or "seconds" in metric_name:
            return "seconds"
        else:
            return "value"
    
    def run_benchmarks(self, models: Optional[List[str]] = None, 
                      datasets: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run benchmarks for specified models and datasets."""
        models = models or self.config["models"]
        datasets = datasets or self.config["datasets"]
        
        # Filter models to only available ones
        available_models = set(get_model_names())
        models = [m for m in models if m in available_models]
        
        if not models:
            raise ValueError(f"No valid models found. Available: {sorted(available_models)}")
        
        print(f"Running benchmarks for {len(models)} models on {len(datasets)} datasets")
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        
        all_results = []
        for model in models:
            for dataset in datasets:
                results = self.benchmark_model(model, dataset)
                all_results.extend(results)
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        output_data = {
            "results": [asdict(r) for r in self.results],
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "commit": os.environ.get("GITHUB_SHA", "local"),
                "config": self.config,
                "environment": {
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "platform": sys.platform
                },
                "timings": list(self._timing_info.values()),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark XTYLearner models")
    parser.add_argument("--model", type=str, help="Single model to benchmark")
    parser.add_argument("--dataset", type=str, help="Single dataset to use")
    parser.add_argument("--models", nargs="+", help="List of models to benchmark")
    parser.add_argument("--datasets", nargs="+", help="List of datasets to use")
    parser.add_argument("--output", type=str, default="benchmark_results.json", 
                       help="Output file path")
    parser.add_argument("--config", type=str, default="benchmark_config.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Determine models and datasets
    if args.model and args.dataset:
        models = [args.model]
        datasets = [args.dataset]
    else:
        models = args.models
        datasets = args.datasets
    
    # Create benchmarker and run
    benchmarker = ModelBenchmarker(args.config)
    
    try:
        results = benchmarker.run_benchmarks(models, datasets)
        benchmarker.save_results(args.output)
        
        print(f"\nCompleted {len(results)} benchmark measurements")
        
        # Print summary
        for result in results:
            print(f"{result.name}: {result.value:.4f} {result.unit}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
