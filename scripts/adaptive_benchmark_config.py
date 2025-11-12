"""
Adaptive benchmark configuration with model-appropriate epoch budgets.

This configuration assigns training epochs based on model convergence characteristics:
- Fast convergers (simple models): 10 epochs
- Medium convergers (standard neural nets): 15-20 epochs
- Slow convergers (diffusion/flow models): 30 epochs

Sample size increased to 1000 for more stable results.
"""

import json
from pathlib import Path

# Model-specific epoch budgets based on convergence speed
EPOCH_BUDGETS = {
    # ===== FAST CONVERGERS (5-10 epochs) =====
    # Non-parametric and simple models
    "em": 5,
    "lp_knn": 5,
    "prob_circuit": 10,

    # Simple semi-supervised methods
    "vat": 10,
    "mean_teacher": 10,
    "fixmatch": 10,

    # ===== MEDIUM CONVERGERS (15-20 epochs) =====
    # Standard neural architectures
    "cycle_dual": 15,
    "multitask": 15,
    "dragon_net": 15,
    "ganite": 15,
    "semiite": 15,
    "ctm_t": 15,

    # GNN-based models
    "gnn_scm": 20,
    "diffusion_gnn_scm": 20,
    "gnn_ebm": 20,

    # VAE-based models
    "m2_vae": 20,
    "ss_cevae": 20,
    "cevae_m": 20,
    "diffusion_cevae": 25,  # Diffusion component needs more

    # Contrastive and other methods
    "ccl_cpc": 15,
    "cycle_vat": 15,
    "cacore": 15,
    "vacim": 15,
    "scgm": 15,

    # Energy-based models
    "joint_ebm": 20,
    "eg_ddi": 20,

    # ===== SLOW CONVERGERS (25-30 epochs) =====
    # Flow-based models
    "flow_ssc": 25,
    "cnflow": 25,
    "gflownet_treatment": 25,

    # Diffusion models (need extensive training)
    "bridge_diff": 30,
    "lt_flow_diff": 30,
    "jsbf": 30,

    # Complex architectures
    "factor_vae_plus": 25,
    "deconfounder_cfm": 25,
    "masked_tabular_transformer": 20,
    "tab_jepa": 20,
    "vime": 20,
    "dag_transformer": 20,

    # Conditional flow models
    "crf": 20,
    "crf_discrete": 20,
}

# Sample sizes per dataset (increased from 100 to 1000)
SAMPLE_SIZES = {
    "synthetic": 1000,
    "synthetic_mixed": 1000,
    "criteo_uplift": 1000,
    "nhefs": 1566,  # Use full dataset
}

# Default for models not in the budget dict
DEFAULT_EPOCHS = 20


def get_epoch_budget(model_name: str) -> int:
    """Get the appropriate epoch budget for a model."""
    return EPOCH_BUDGETS.get(model_name, DEFAULT_EPOCHS)


def get_sample_size(dataset_name: str) -> int:
    """Get the appropriate sample size for a dataset."""
    return SAMPLE_SIZES.get(dataset_name, 1000)


def create_benchmark_config(output_path: str = "benchmark_config_adaptive.json"):
    """Create adaptive benchmark configuration file."""

    config = {
        "description": "Adaptive benchmark with model-specific epoch budgets and increased samples",
        "version": "2.0",
        "rationale": {
            "epochs": "Different models have different convergence speeds. Fast convergers "
                     "(em, lp_knn) need only 5-10 epochs, while diffusion models need 30.",
            "samples": "Increased from 100 to 1000 for more stable results and better "
                      "representation of model capabilities."
        },
        "sample_sizes": SAMPLE_SIZES,
        "epoch_budgets": EPOCH_BUDGETS,
        "default_epochs": DEFAULT_EPOCHS,
        "training_config": {
            "batch_size": 32,
            "learning_rate_schedule": "constant",  # Could be made adaptive too
            "early_stopping": False,  # Disabled to ensure full epoch budget used
            "random_seed": 42,
        },
        "computational_cost": {
            "estimated_speedup_vs_uniform_30": "~40%",
            "notes": "By using appropriate epochs per model, we avoid overtraining "
                    "fast convergers while giving slow convergers adequate time."
        }
    }

    # Calculate total epoch-model pairs
    total_epochs = sum(EPOCH_BUDGETS.values())
    num_models = len(EPOCH_BUDGETS)
    avg_epochs = total_epochs / num_models

    config["statistics"] = {
        "num_models": num_models,
        "total_epochs": total_epochs,
        "average_epochs_per_model": round(avg_epochs, 1),
        "min_epochs": min(EPOCH_BUDGETS.values()),
        "max_epochs": max(EPOCH_BUDGETS.values()),
    }

    # Write config
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created adaptive benchmark config: {output_path}")
    print(f"\nStatistics:")
    print(f"  Models configured: {num_models}")
    print(f"  Average epochs: {avg_epochs:.1f}")
    print(f"  Range: {min(EPOCH_BUDGETS.values())}-{max(EPOCH_BUDGETS.values())} epochs")
    print(f"  Sample sizes: {list(SAMPLE_SIZES.values())}")

    return config


def print_budget_summary():
    """Print a summary of epoch budgets by category."""

    print("\n" + "="*80)
    print("MODEL EPOCH BUDGETS SUMMARY")
    print("="*80)

    categories = {
        "Fast Convergers (5-10 epochs)": [
            m for m, e in EPOCH_BUDGETS.items() if e <= 10
        ],
        "Medium Convergers (15-20 epochs)": [
            m for m, e in EPOCH_BUDGETS.items() if 11 <= e <= 20
        ],
        "Slow Convergers (25-30 epochs)": [
            m for m, e in EPOCH_BUDGETS.items() if e >= 25
        ],
    }

    for category, models in categories.items():
        print(f"\n{category}:")
        print("-" * 60)
        for model in sorted(models):
            epochs = EPOCH_BUDGETS[model]
            print(f"  {model:<30} {epochs:>3} epochs")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Create the config file
    config = create_benchmark_config()

    # Print summary
    print_budget_summary()

    # Show computational benefit
    print("\nCOMPUTATIONAL BENEFIT:")
    print("-" * 60)

    num_models = len(EPOCH_BUDGETS)
    adaptive_total = sum(EPOCH_BUDGETS.values())
    uniform_10_total = num_models * 10
    uniform_30_total = num_models * 30

    print(f"Uniform 10 epochs:  {uniform_10_total:,} total epoch-models")
    print(f"Adaptive budgets:   {adaptive_total:,} total epoch-models")
    print(f"Uniform 30 epochs:  {uniform_30_total:,} total epoch-models")
    print()
    print(f"Adaptive vs Uniform-10:  +{adaptive_total - uniform_10_total:,} epochs ({(adaptive_total/uniform_10_total - 1)*100:.0f}% more)")
    print(f"Adaptive vs Uniform-30:  {adaptive_total - uniform_30_total:,} epochs ({(1 - adaptive_total/uniform_30_total)*100:.0f}% savings)")
    print()
    print("⚡ Adaptive approach gives slow convergers enough time while")
    print("   avoiding wasting compute on models that already converged!")
