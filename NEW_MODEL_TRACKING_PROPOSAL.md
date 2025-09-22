# Production-Ready ML Model Benchmarking System for GitHub Actions

## Architectural overview with progressive enhancement

The system builds on proven patterns from ASV, github-action-benchmark, and major ML libraries, adapted for lightweight GitHub-native operation. The architecture supports starting simple and progressively adding sophistication without infrastructure changes.

**Core Design Principles:**
- **Zero Infrastructure**: Everything runs in GitHub (Actions, Pages, artifacts)
- **Statistical Rigor**: Multiple measurement passes with confidence intervals
- **ML-Aware**: First-class support for model quality metrics beyond timing
- **Fork-Safe**: Secure handling of external PRs without compromising data
- **Progressive Enhancement**: Start with basics, add features incrementally

## Complete implementation with production patterns

### Core Evaluation Script with Statistical Robustness

```python
# eval.py - Production-ready evaluation with statistical rigor
import json
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

@dataclass
class BenchmarkResult:
    """Structured benchmark result with metadata"""
    name: str
    value: float
    unit: str
    range: List[float]  # min, max for confidence intervals
    samples: int
    timestamp: str
    commit: str
    environment: Dict[str, Any]
    
class ModelBenchmarker:
    """Production benchmarking with statistical methods"""
    
    def __init__(self, config_path: str = "benchmark_config.json"):
        self.config = self._load_config(config_path)
        self.results = []
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration with defaults"""
        defaults = {
            "iterations": 5,
            "warmup_iterations": 2,
            "confidence_level": 0.95,
            "metrics": ["accuracy", "f1_score", "latency_ms"],
            "models": ["model_v1", "model_v2"],
            "datasets": ["validation_set"],
            "statistical_method": "bootstrap"
        }
        
        if Path(path).exists():
            with open(path) as f:
                config = json.load(f)
                return {**defaults, **config}
        return defaults
    
    def benchmark_model(self, model_name: str, dataset: str) -> List[BenchmarkResult]:
        """Run benchmarks with multiple passes for statistical accuracy"""
        results = []
        
        # Load model and data once
        model = self._load_model(model_name)
        X, y = self._load_dataset(dataset)
        
        # Warmup runs (not counted)
        for _ in range(self.config["warmup_iterations"]):
            self._run_inference(model, X)
        
        # Actual measurements
        for metric_name in self.config["metrics"]:
            samples = []
            
            for _ in range(self.config["iterations"]):
                if metric_name == "latency_ms":
                    # Measure inference latency
                    start = time.perf_counter()
                    predictions = self._run_inference(model, X)
                    elapsed = (time.perf_counter() - start) * 1000
                    samples.append(elapsed)
                else:
                    # Measure model quality metrics
                    predictions = self._run_inference(model, X)
                    score = self._calculate_metric(metric_name, y, predictions)
                    samples.append(score)
            
            # Calculate statistics
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            confidence_interval = self._calculate_confidence_interval(samples)
            
            result = BenchmarkResult(
                name=f"{model_name}_{dataset}_{metric_name}",
                value=mean_val,
                unit=self._get_unit(metric_name),
                range=[confidence_interval[0], confidence_interval[1]],
                samples=len(samples),
                timestamp=datetime.utcnow().isoformat(),
                commit=os.environ.get("GITHUB_SHA", "local"),
                environment={
                    "runner_os": os.environ.get("RUNNER_OS", "unknown"),
                    "python_version": sys.version.split()[0],
                    "model_hash": self._get_model_hash(model)
                }
            )
            results.append(result)
        
        return results
    
    def _calculate_confidence_interval(self, samples: List[float]) -> tuple:
        """Bootstrap confidence interval calculation"""
        if self.config["statistical_method"] == "bootstrap":
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                resample = np.random.choice(samples, size=len(samples), replace=True)
                bootstrap_means.append(np.mean(resample))
            
            alpha = 1 - self.config["confidence_level"]
            lower = np.percentile(bootstrap_means, alpha/2 * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
            return (lower, upper)
        else:
            # T-distribution based interval
            from scipy import stats
            confidence = self.config["confidence_level"]
            n = len(samples)
            m, se = np.mean(samples), stats.sem(samples)
            h = se * stats.t.ppf((1 + confidence) / 2., n-1)
            return (m-h, m+h)
    
    def compare_with_baseline(self, baseline_path: str, threshold: float = 0.05):
        """Statistical comparison with baseline"""
        if not Path(baseline_path).exists():
            return {"status": "no_baseline", "regressions": []}
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        regressions = []
        improvements = []
        
        for result in self.results:
            baseline_result = self._find_baseline_match(result, baseline)
            
            if baseline_result:
                # Check for statistically significant difference
                if self._is_significant_change(result, baseline_result):
                    change_pct = ((result.value - baseline_result["value"]) / 
                                 baseline_result["value"]) * 100
                    
                    if abs(change_pct) > threshold * 100:
                        change_data = {
                            "metric": result.name,
                            "baseline": baseline_result["value"],
                            "current": result.value,
                            "change_percent": change_pct,
                            "confidence_interval": result.range
                        }
                        
                        if self._is_regression(result.name, change_pct):
                            regressions.append(change_data)
                        else:
                            improvements.append(change_data)
        
        return {
            "status": "fail" if regressions else "pass",
            "regressions": regressions,
            "improvements": improvements
        }
```

### GitHub Actions Workflow with Matrix Parallelization

```yaml
# .github/workflows/benchmark.yml
name: Model Quality Benchmarking

on:
  push:
    branches: [main]
  pull_request:
    types: [labeled, synchronize]
  workflow_dispatch:
    inputs:
      full_benchmark:
        description: 'Run full benchmark suite'
        required: false
        default: 'false'

# Prevent concurrent updates to benchmark data
concurrency:
  group: benchmark-${{ github.ref }}
  cancel-in-progress: false

jobs:
  # Dynamic matrix generation based on available models
  setup-matrix:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
      cache-key: ${{ steps.cache-key.outputs.key }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate benchmark matrix
        id: set-matrix
        run: |
          # Detect available models and datasets
          MODELS=$(ls models/*.json | xargs -n1 basename | jq -R -s -c 'split("\n")[:-1]')
          DATASETS=$(ls data/*.json | xargs -n1 basename | jq -R -s -c 'split("\n")[:-1]')
          
          # Create matrix based on PR vs main branch
          if [[ "${{ github.event_name }}" == "pull_request" ]]; then
            # Limited matrix for PRs
            matrix='{"model":["model_small"],"dataset":["test_dataset"]}'
          else
            # Full matrix for main branch
            matrix=$(jq -n --argjson models "$MODELS" --argjson datasets "$DATASETS" \
              '{model: $models, dataset: $datasets}')
          fi
          
          echo "matrix=$matrix" >> $GITHUB_OUTPUT
      
      - name: Generate cache key
        id: cache-key
        run: |
          key="ml-deps-${{ runner.os }}-$(date +'%Y%W')-${{ hashFiles('requirements.txt', 'models/*.json') }}"
          echo "key=$key" >> $GITHUB_OUTPUT

  # Parallel benchmark execution
  benchmark:
    needs: setup-matrix
    runs-on: ${{ matrix.model == 'model_large' && 'ubuntu-latest-8-cores' || 'ubuntu-latest' }}
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix: ${{ fromJSON(needs.setup-matrix.outputs.matrix) }}
      max-parallel: 4  # Prevent runner exhaustion
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need history for comparisons
      
      # Multi-layer caching strategy
      - name: Setup Python with cache
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Cache ML models and datasets
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/huggingface
            ./models/pretrained
            ./data/processed
          key: ${{ needs.setup-matrix.outputs.cache-key }}
          restore-keys: |
            ml-deps-${{ runner.os }}-
      
      # Run benchmarks with retry logic
      - name: Run benchmarks
        uses: nick-fields/retry@v3
        with:
          timeout_minutes: 15
          max_attempts: 3
          retry_on: error
          command: |
            python eval.py \
              --model ${{ matrix.model }} \
              --dataset ${{ matrix.dataset }} \
              --output results-${{ matrix.model }}-${{ matrix.dataset }}.json
      
      # Upload individual results
      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: results-${{ matrix.model }}-${{ matrix.dataset }}
          path: results-*.json
          retention-days: 7
          compression-level: 6

  # Aggregate and analyze results
  analyze:
    needs: benchmark
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      pages: write
      id-token: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download all results
        uses: actions/download-artifact@v4
        with:
          pattern: results-*
          merge-multiple: true
          path: ./results
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install visualization dependencies
        run: |
          pip install matplotlib pandas numpy scipy
          # Set headless backend
          export MPLBACKEND=Agg
      
      - name: Aggregate results and update history
        id: aggregate
        run: |
          python scripts/aggregate_results.py \
            --input-dir ./results \
            --history-file history.json \
            --output-file current-benchmarks.json
      
      # Generate visualizations
      - name: Generate benchmark charts
        run: |
          python scripts/generate_charts.py \
            --history history.json \
            --output-dir ./charts
      
      # Statistical comparison with baseline
      - name: Compare with baseline
        id: compare
        if: github.event_name == 'pull_request'
        run: |
          # Fetch baseline from main branch
          git fetch origin main
          git checkout origin/main -- history.json || echo "{}" > baseline.json
          
          python scripts/compare_benchmarks.py \
            --baseline baseline.json \
            --current current-benchmarks.json \
            --threshold 0.05 \
            --output comparison.md
          
          # Set outputs for PR comment
          echo "has_regressions=$([[ -s comparison.md ]] && echo 'true' || echo 'false')" >> $GITHUB_OUTPUT
      
      # Update PR with results
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: peter-evans/create-or-update-comment@v4
        with:
          issue-number: ${{ github.event.pull_request.number }}
          body-path: comparison.md
          
      # Deploy to GitHub Pages (main branch only)
      - name: Deploy dashboard
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./charts
          destination_dir: benchmarks
```

### Advanced Storage Pattern with History Management

```python
# scripts/storage_manager.py
import json
import gzip
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import hashlib

class BenchmarkStorageManager:
    """Production storage with compression and retention policies"""
    
    def __init__(self, storage_dir: Path = Path("benchmark-data")):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        
        # Configuration for different storage tiers
        self.tiers = {
            "hot": {"days": 7, "compression": None},
            "warm": {"days": 30, "compression": "gzip"},
            "cold": {"days": 365, "compression": "gzip", "sampling": 10}
        }
    
    def store_benchmark(self, data: Dict, tier: str = "hot") -> str:
        """Store benchmark with appropriate compression"""
        timestamp = datetime.utcnow()
        data_hash = self._hash_data(data)
        
        # Prevent duplicate storage
        if self._exists(data_hash):
            return data_hash
        
        # Apply tier-specific storage
        if tier == "hot":
            path = self.storage_dir / f"{timestamp.isoformat()}_{data_hash}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            path = self.storage_dir / f"{timestamp.isoformat()}_{data_hash}.json.gz"
            with gzip.open(path, 'wt') as f:
                json.dump(data, f)
        
        return data_hash
    
    def migrate_storage_tiers(self):
        """Migrate data between storage tiers based on age"""
        now = datetime.utcnow()
        
        for file in self.storage_dir.glob("*.json"):
            file_time = datetime.fromisoformat(file.stem.split('_')[0])
            age_days = (now - file_time).days
            
            if age_days > self.tiers["hot"]["days"]:
                # Compress and move to warm tier
                self._compress_file(file)
                file.unlink()
            
            # Sample old data for cold tier
            if age_days > self.tiers["warm"]["days"]:
                if hash(file.name) % self.tiers["cold"]["sampling"] != 0:
                    file.unlink()  # Keep only 1/10th of cold data
    
    def query_benchmarks(self, 
                        start_date: datetime = None,
                        end_date: datetime = None,
                        metrics: List[str] = None) -> List[Dict]:
        """Query benchmarks with filters"""
        results = []
        
        for file in sorted(self.storage_dir.glob("*.json*")):
            # Extract timestamp from filename
            timestamp_str = file.stem.split('_')[0].replace('.json', '')
            file_time = datetime.fromisoformat(timestamp_str)
            
            # Apply date filters
            if start_date and file_time < start_date:
                continue
            if end_date and file_time > end_date:
                continue
            
            # Load data
            data = self._load_file(file)
            
            # Apply metric filters
            if metrics:
                data = self._filter_metrics(data, metrics)
            
            results.append(data)
        
        return results
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary stats for dashboard"""
        recent_data = self.query_benchmarks(
            start_date=datetime.utcnow() - timedelta(days=30)
        )
        
        stats = {
            "total_benchmarks": len(recent_data),
            "models": set(),
            "metrics": {},
            "trends": {}
        }
        
        for benchmark in recent_data:
            for result in benchmark.get("results", []):
                metric_name = result["name"]
                if metric_name not in stats["metrics"]:
                    stats["metrics"][metric_name] = []
                stats["metrics"][metric_name].append(result["value"])
        
        # Calculate trends
        for metric, values in stats["metrics"].items():
            if len(values) > 1:
                # Simple linear regression for trend
                x = list(range(len(values)))
                slope = np.polyfit(x, values, 1)[0]
                stats["trends"][metric] = "improving" if slope > 0 else "degrading"
        
        return stats
```

### Visualization and Dashboard Generation

```python
# scripts/generate_charts.py
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import seaborn as sns

class BenchmarkVisualizer:
    """Production visualization with multiple chart types"""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
        
    def generate_dashboard(self, history_file: str, output_dir: str):
        """Generate complete dashboard with multiple visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load benchmark history
        with open(history_file) as f:
            history = json.load(f)
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Performance trends over time
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_series(ax1, history, metric_type="latency")
        
        # Panel 2: Model quality metrics
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_quality_radar(ax2, history)
        
        # Panel 3: Comparison matrix
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_comparison_matrix(ax3, history)
        
        # Panel 4: Resource usage
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_resource_usage(ax4, history)
        
        # Panel 5: Statistical distribution
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_distribution(ax5, history)
        
        # Panel 6: Regression detection
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_regression_detection(ax6, history)
        
        # Panel 7: Confidence intervals
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_confidence_intervals(ax7, history)
        
        # Save dashboard
        plt.suptitle('ML Model Quality Benchmark Dashboard', fontsize=16, y=0.995)
        plt.savefig(output_path / 'dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate individual charts for embedding
        self._generate_individual_charts(history, output_path)
        
        # Generate interactive HTML dashboard
        self._generate_html_dashboard(history, output_path)
    
    def _plot_time_series(self, ax, history, metric_type="latency"):
        """Plot time series with trend lines and annotations"""
        # Extract time series data
        timestamps = []
        values = {}
        
        for entry in history:
            timestamp = pd.to_datetime(entry["timestamp"])
            timestamps.append(timestamp)
            
            for result in entry["results"]:
                if metric_type in result["name"]:
                    model = result["name"].split("_")[0]
                    if model not in values:
                        values[model] = []
                    values[model].append(result["value"])
        
        # Plot each model
        for i, (model, vals) in enumerate(values.items()):
            ax.plot(timestamps[:len(vals)], vals, 
                   marker='o', label=model, color=self.colors[i],
                   linewidth=2, markersize=6)
            
            # Add trend line
            if len(vals) > 3:
                z = np.polyfit(range(len(vals)), vals, 1)
                p = np.poly1d(z)
                ax.plot(timestamps[:len(vals)], p(range(len(vals))),
                       "--", alpha=0.5, color=self.colors[i])
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Performance Trends Over Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_confidence_intervals(self, ax, history):
        """Plot metrics with confidence intervals"""
        latest = history[-1] if history else {"results": []}
        
        metrics = []
        values = []
        errors = []
        
        for result in latest["results"]:
            if "accuracy" in result["name"]:
                metrics.append(result["name"].split("_")[0])
                values.append(result["value"])
                if "range" in result:
                    error = [(result["value"] - result["range"][0]),
                            (result["range"][1] - result["value"])]
                    errors.append(error)
                else:
                    errors.append([0, 0])
        
        # Create bar chart with error bars
        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, values, yerr=np.array(errors).T,
                     capsize=5, color=self.colors[:len(metrics)],
                     alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_html_dashboard(self, history, output_path):
        """Generate interactive HTML dashboard"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Benchmark Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { 
                    display: inline-block; 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 10px;
                    border-radius: 5px;
                }
                .metric-value { font-size: 2em; font-weight: bold; }
                .metric-trend { color: #666; }
                .improvement { color: green; }
                .regression { color: red; }
            </style>
        </head>
        <body>
            <h1>ML Model Quality Benchmarks</h1>
            <div id="summary-cards"></div>
            <div id="time-series-plot"></div>
            <div id="comparison-plot"></div>
            
            <script>
                const benchmarkData = {data};
                
                // Generate summary cards
                const latestData = benchmarkData[benchmarkData.length - 1];
                const summaryDiv = document.getElementById('summary-cards');
                
                latestData.results.forEach(result => {{
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    
                    const trend = result.trend || 'stable';
                    const trendClass = trend === 'up' ? 'improvement' : 
                                      trend === 'down' ? 'regression' : '';
                    
                    card.innerHTML = `
                        <div class="metric-name">${{result.name}}</div>
                        <div class="metric-value">${{result.value.toFixed(3)}}</div>
                        <div class="metric-trend ${{trendClass}}">${{trend}}</div>
                    `;
                    summaryDiv.appendChild(card);
                }});
                
                // Generate time series plot
                // ... Plotly.js code for interactive charts
            </script>
        </body>
        </html>
        """.format(data=json.dumps(history))
        
        with open(output_path / 'index.html', 'w') as f:
            f.write(html_template)
```

### PR Comment Integration with Rich Formatting

```python
# scripts/generate_pr_comment.py
import json
from pathlib import Path
from typing import Dict, List
import os

class PRCommentGenerator:
    """Generate rich PR comments with benchmark results"""
    
    def generate_comment(self, comparison_data: Dict) -> str:
        """Generate formatted markdown comment for PR"""
        
        # Header with summary
        comment = "## 📊 Benchmark Results\n\n"
        
        # Quick status
        if comparison_data["status"] == "pass":
            comment += "✅ **All benchmarks passed!**\n\n"
        else:
            comment += "⚠️ **Performance regressions detected**\n\n"
        
        # Performance comparison table
        comment += "### Performance Comparison\n\n"
        comment += "| Metric | Baseline | Current | Change | Status |\n"
        comment += "|--------|----------|---------|--------|--------|\n"
        
        for metric in comparison_data["metrics"]:
            change = metric["change_percent"]
            status = self._get_status_emoji(metric["name"], change)
            
            comment += f"| {metric['name']} | {metric['baseline']:.3f} | "
            comment += f"{metric['current']:.3f} | {change:+.1f}% | {status} |\n"
        
        # Regressions detail
        if comparison_data["regressions"]:
            comment += "\n### ⚠️ Regressions Detected\n\n"
            for reg in comparison_data["regressions"]:
                comment += f"- **{reg['metric']}**: {reg['change_percent']:.1f}% slower "
                comment += f"({reg['baseline']:.3f} → {reg['current']:.3f})\n"
        
        # Improvements
        if comparison_data["improvements"]:
            comment += "\n### ✨ Improvements\n\n"
            for imp in comparison_data["improvements"]:
                comment += f"- **{imp['metric']}**: {abs(imp['change_percent']):.1f}% faster\n"
        
        # Visualization links
        comment += "\n### 📈 Visualizations\n\n"
        comment += f"[View Dashboard](https://github.com/{os.environ['GITHUB_REPOSITORY']}/pages/benchmarks)\n\n"
        
        # Collapsible detailed results
        comment += "<details>\n<summary>📋 Detailed Results</summary>\n\n"
        comment += "```json\n"
        comment += json.dumps(comparison_data, indent=2)
        comment += "\n```\n</details>\n"
        
        # Footer with metadata
        comment += f"\n---\n"
        comment += f"*Benchmarked on {comparison_data['timestamp']} "
        comment += f"| Commit: {comparison_data['commit'][:7]}*"
        
        return comment
    
    def _get_status_emoji(self, metric_name: str, change_percent: float) -> str:
        """Get appropriate emoji based on metric type and change"""
        # For quality metrics, higher is better
        if any(x in metric_name for x in ["accuracy", "f1", "precision", "recall"]):
            if change_percent > 1:
                return "✅ Improved"
            elif change_percent < -1:
                return "❌ Degraded"
            else:
                return "➡️ Stable"
        
        # For performance metrics, lower is better
        elif any(x in metric_name for x in ["latency", "time", "memory"]):
            if change_percent < -1:
                return "✅ Improved"
            elif change_percent > 1:
                return "❌ Degraded"
            else:
                return "➡️ Stable"
        
        return "➡️"
```

### Production Error Handling and Retry Patterns

```yaml
# .github/workflows/benchmark-with-recovery.yml
name: Resilient Benchmarking

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Benchmark with comprehensive error handling
        run: |
          set -euo pipefail
          
          # Function for retrying commands
          retry() {
            local max_attempts=$1
            shift
            local count=0
            
            until "$@"; do
              exit_code=$?
              count=$((count + 1))
              
              if [ $count -lt $max_attempts ]; then
                echo "Attempt $count failed with exit code $exit_code. Retrying..."
                sleep $((2 ** count))  # Exponential backoff
              else
                echo "Failed after $count attempts"
                return $exit_code
              fi
            done
            
            return 0
          }
          
          # Run benchmark with timeout and retry
          retry 3 timeout 600 python eval.py \
            --model ${{ matrix.model }} \
            --fallback-on-error \
            --output results.json || {
              echo "Full benchmark failed, running minimal suite"
              python eval_minimal.py --output results.json
            }
      
      - name: Validate results
        run: |
          python -c "
          import json
          import sys
          
          with open('results.json') as f:
              data = json.load(f)
          
          # Validate structure
          required_fields = ['results', 'metadata', 'timestamp']
          missing = [f for f in required_fields if f not in data]
          
          if missing:
              print(f'Missing fields: {missing}')
              sys.exit(1)
          
          # Validate data quality
          for result in data['results']:
              if result['samples'] < 3:
                  print(f'Warning: Low sample count for {result[\"name\"]}}')
              
              if result['range'][1] - result['range'][0] > result['value'] * 0.5:
                  print(f'Warning: High variance for {result[\"name\"]}')
          "
```

## Fork PR security pattern implementation

```yaml
# .github/workflows/benchmark-fork-safe.yml
name: Fork-Safe Benchmarking

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  # Job 1: Run benchmarks (no secrets)
  benchmark-fork:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: |
          python eval.py --output results.json
      
      - name: Upload results as artifact
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results.json

---
# .github/workflows/benchmark-comment.yml
name: Process Benchmark Results

on:
  workflow_run:
    workflows: ["Fork-Safe Benchmarking"]
    types: [completed]

jobs:
  # Job 2: Process results (has secrets)
  comment-results:
    runs-on: ubuntu-latest
    if: github.event.workflow_run.conclusion == 'success'
    permissions:
      pull-requests: write
    
    steps:
      - name: Download artifacts
        uses: actions/github-script@v7
        with:
          script: |
            const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
              owner: context.repo.owner,
              repo: context.repo.repo,
              run_id: ${{ github.event.workflow_run.id }}
            });
            
            const matchArtifact = artifacts.data.artifacts.find(a => 
              a.name === 'benchmark-results'
            );
            
            const download = await github.rest.actions.downloadArtifact({
              owner: context.repo.owner,
              repo: context.repo.repo,
              artifact_id: matchArtifact.id,
              archive_format: 'zip'
            });
            
            require('fs').writeFileSync('results.zip', Buffer.from(download.data));
      
      - name: Process and comment
        run: |
          unzip results.zip
          python scripts/generate_pr_comment.py --input results.json
```

## Key production considerations and optimizations

### Caching Strategy Hierarchy
1. **System dependencies**: Weekly cache with hash-based keys
2. **Model weights**: Content-based caching with Git LFS for large files
3. **Compiled code**: SHA-based caching for reproducibility
4. **Results**: Time-based retention with automatic cleanup

### Performance Optimizations
- **Matrix parallelization**: Max 4 concurrent jobs to prevent runner exhaustion
- **Warmup iterations**: 2 warmup runs before actual measurements
- **Statistical methods**: Bootstrap confidence intervals for robust comparisons
- **Progressive benchmarking**: Quick smoke tests for PRs, full suite for main

### Reliability Patterns
- **Retry mechanisms**: Exponential backoff with 3 attempts
- **Fallback strategies**: Minimal benchmark suite on failures
- **Validation**: Schema validation and sanity checks on results
- **Timeout management**: Job-level and step-level timeouts

### Security Best Practices
- **Fork PR isolation**: Two-workflow pattern for untrusted code
- **Permission minimization**: Least privilege for each job
- **Secret management**: Pass secrets only where needed
- **Branch protection**: Never run on pull_request with write permissions

This implementation provides a production-ready foundation that handles edge cases, scales efficiently, and maintains statistical rigor while remaining lightweight and GitHub-native. The progressive enhancement approach allows starting simple and adding sophistication as needed without infrastructure changes.