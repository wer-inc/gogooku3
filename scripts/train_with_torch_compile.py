#!/usr/bin/env python
"""
Enhanced training script with PyTorch 2.x compilation support.

This wrapper adds torch.compile optimization to the existing training pipeline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def setup_torch_compile_env():
    """Set up environment for torch.compile optimization."""

    # PyTorch 2.x compilation environment
    env_vars = {
        # Compilation settings
        "TORCH_COMPILE_DEBUG": "0",  # Set to 1 for debugging
        "TORCH_LOGS": "+dynamo",  # Enable dynamo logs
        "TORCH_COMPILE_CACHE_SIZE_LIMIT": "64",  # Cache size in GB
        # Backend optimizations
        "TORCHDYNAMO_VERBOSE": "0",  # Set to 1 for verbose output
        "TORCHDYNAMO_SUPPRESS_ERRORS": "0",  # Don't suppress compilation errors
        # CUDA optimizations
        "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # cuDNN optimizations
        "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
        "CUDNN_BENCHMARK": "1",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    logger.info("✅ PyTorch 2.x compilation environment configured")
    logger.info(f"   PyTorch version: {torch.__version__}")

    # Check if torch.compile is available
    if hasattr(torch, "compile"):
        logger.info("   torch.compile is available")
        return True
    else:
        logger.warning("   torch.compile not available - PyTorch 2.x required")
        return False


def compile_model_with_options(model, compile_config: dict[str, Any] | None = None):
    """
    Compile model with PyTorch 2.x torch.compile.

    Args:
        model: PyTorch model to compile
        compile_config: Compilation configuration

    Returns:
        Compiled model
    """

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available - returning original model")
        return model

    default_config = {
        "mode": "max-autotune",  # Options: "default", "reduce-overhead", "max-autotune"
        "fullgraph": False,  # Don't require full graph compilation
        "dynamic": False,  # Static shapes for better optimization
        "backend": "inductor",  # Default backend
        "options": {
            "triton.cudagraphs": True,  # Enable CUDA graphs
            "max_autotune": True,  # Maximum autotuning
            "coordinate_descent_tuning": True,  # Better tuning
            "epilogue_fusion": True,  # Fuse epilogue operations
            "shape_padding": True,  # Pad shapes for better performance
        },
    }

    if compile_config:
        default_config.update(compile_config)

    try:
        logger.info(
            f"🔧 Compiling model with torch.compile (mode={default_config['mode']})"
        )

        # Extract options for the backend
        backend_options = default_config.pop("options", {})

        # Compile the model
        compiled_model = torch.compile(
            model,
            mode=default_config["mode"],
            fullgraph=default_config["fullgraph"],
            dynamic=default_config["dynamic"],
            backend=default_config["backend"],
            options=backend_options,
        )

        logger.info("✅ Model compiled successfully")
        return compiled_model

    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        logger.warning("Falling back to eager mode")
        return model


def patch_train_atft_for_compile():
    """
    Monkey-patch the train_atft module to add torch.compile support.
    """

    try:
        # Import the training script
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import train_atft

        # Store original model creation function
        if hasattr(train_atft, "create_model"):
            original_create_model = train_atft.create_model
        else:
            # Find the model creation function in the module
            for attr_name in dir(train_atft):
                if "create" in attr_name.lower() and "model" in attr_name.lower():
                    original_create_model = getattr(train_atft, attr_name)
                    break
            else:
                logger.warning("Could not find model creation function to patch")
                return False

        def create_model_with_compile(*args, **kwargs):
            """Wrapper that compiles the model after creation."""

            # Create the original model
            model = original_create_model(*args, **kwargs)

            # Compile if enabled via environment
            if os.getenv("ENABLE_TORCH_COMPILE", "1") == "1":
                compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
                compile_config = {
                    "mode": compile_mode,
                    "fullgraph": os.getenv("TORCH_COMPILE_FULLGRAPH", "0") == "1",
                    "dynamic": os.getenv("TORCH_COMPILE_DYNAMIC", "0") == "1",
                }

                model = compile_model_with_options(model, compile_config)

            return model

        # Replace the function
        if hasattr(train_atft, "create_model"):
            train_atft.create_model = create_model_with_compile
        else:
            setattr(
                train_atft, original_create_model.__name__, create_model_with_compile
            )

        logger.info("✅ train_atft patched for torch.compile support")
        return True

    except Exception as e:
        logger.error(f"Failed to patch train_atft: {e}")
        return False


def main():
    """
    Main entry point that wraps train_atft with torch.compile support.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Train ATFT-GAT-FAN with torch.compile"
    )

    # Compilation options
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Require full graph compilation",
    )
    parser.add_argument(
        "--compile-dynamic", action="store_true", help="Use dynamic shapes"
    )

    # Pass remaining arguments to train_atft
    args, remaining_args = parser.parse_known_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 60)
    print("🚀 TORCH.COMPILE ENHANCED TRAINING")
    print("=" * 60)

    # Set up torch.compile environment
    has_compile = setup_torch_compile_env()

    if not has_compile:
        print("\n⚠️ PyTorch 2.x not detected. Running without torch.compile")
    else:
        # Set compilation options via environment
        os.environ["ENABLE_TORCH_COMPILE"] = "0" if args.no_compile else "1"
        os.environ["TORCH_COMPILE_MODE"] = args.compile_mode
        os.environ["TORCH_COMPILE_FULLGRAPH"] = "1" if args.compile_fullgraph else "0"
        os.environ["TORCH_COMPILE_DYNAMIC"] = "1" if args.compile_dynamic else "0"

        print("\n📝 Compilation settings:")
        print(f"  Mode: {args.compile_mode}")
        print(f"  Full graph: {args.compile_fullgraph}")
        print(f"  Dynamic shapes: {args.compile_dynamic}")

    # Patch the training script
    if has_compile and not args.no_compile:
        patch_success = patch_train_atft_for_compile()
        if not patch_success:
            print("⚠️ Could not patch training script for torch.compile")

    # Run the original training script with remaining arguments
    sys.argv = ["train_atft.py"] + remaining_args

    # Import and run train_atft
    import train_atft

    # Check if it has a main function
    if hasattr(train_atft, "main"):
        return train_atft.main()
    elif hasattr(train_atft, "train"):
        return train_atft.train()
    else:
        # Execute the module
        exec(open(PROJECT_ROOT / "scripts" / "train_atft.py").read())
        return 0


if __name__ == "__main__":
    sys.exit(main())
