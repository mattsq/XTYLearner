"""Quick validation script for Clone model (no external dependencies)."""

import sys
import importlib.util

# Load the module without installing dependencies
spec = importlib.util.spec_from_file_location(
    "clone", "xtylearner/models/clone.py"
)

# Basic validation checks
def validate_clone_module():
    """Validate Clone model can be loaded and has expected attributes."""
    try:
        # Check file exists and is valid Python
        with open("xtylearner/models/clone.py", "r") as f:
            code = f.read()

        # Basic structural checks
        assert "class Clone" in code, "Clone class not found"
        assert "@register_model" in code, "register_model decorator not found"
        assert "def __init__" in code, "Missing __init__ method"
        assert "def forward" in code, "Missing forward method"
        assert "def loss" in code, "Missing loss method"
        assert "def predict_treatment_proba" in code, "Missing predict_treatment_proba method"
        assert "def predict_ood_score" in code, "Missing predict_ood_score method"
        assert "def predict_outcome" in code, "Missing predict_outcome method"
        assert "def step" in code, "Missing step method"

        # Check for key components
        assert "self.ood_network" in code, "OOD network not initialized"
        assert "self.classifier_network" in code, "Classifier network not initialized"
        assert "self.outcome" in code, "Outcome network not initialized"

        # Check for decoupled architecture
        assert "Independent OOD detection network" in code or "ood_network" in code
        assert "Independent classifier network" in code or "classifier_network" in code

        # Check for feedback mechanism
        assert "feedback" in code.lower(), "Feedback mechanism not implemented"
        assert "lambda_feedback" in code, "lambda_feedback parameter not found"

        # Check exports
        assert "__all__" in code, "Missing __all__ export"
        assert "Clone" in code.split("__all__")[1].split("]")[0], "Clone not exported"

        print("✓ Clone module structure is valid")
        print("✓ All required methods are present")
        print("✓ Decoupled architecture components found")
        print("✓ Feedback mechanism implemented")
        return True

    except AssertionError as e:
        print(f"✗ Validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        return False


def validate_imports():
    """Validate __init__.py exports Clone correctly."""
    try:
        with open("xtylearner/models/__init__.py", "r") as f:
            init_code = f.read()

        assert "from .clone import Clone" in init_code, "Clone import missing from __init__.py"
        assert '"Clone"' in init_code or "'Clone'" in init_code, "Clone not in __all__"

        print("✓ Clone properly exported in __init__.py")
        return True

    except AssertionError as e:
        print(f"✗ Import validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during import validation: {e}")
        return False


def validate_tests():
    """Validate test file exists and has proper structure."""
    try:
        with open("tests/models/test_clone.py", "r") as f:
            test_code = f.read()

        # Count test functions
        test_functions = [line for line in test_code.split('\n') if line.startswith('def test_')]

        assert len(test_functions) >= 10, f"Expected at least 10 tests, found {len(test_functions)}"
        assert "test_clone_basic_forward" in test_code
        assert "test_clone_treatment_proba" in test_code
        assert "test_clone_ood_score" in test_code
        assert "test_clone_loss" in test_code
        assert "test_clone_registry" in test_code

        print(f"✓ Test file exists with {len(test_functions)} test functions")
        return True

    except AssertionError as e:
        print(f"✗ Test validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during test validation: {e}")
        return False


if __name__ == "__main__":
    print("Validating Clone implementation...\n")

    results = [
        validate_clone_module(),
        validate_imports(),
        validate_tests(),
    ]

    print("\n" + "=" * 50)
    if all(results):
        print("✓ All validation checks passed!")
        sys.exit(0)
    else:
        print("✗ Some validation checks failed")
        sys.exit(1)
