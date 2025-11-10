"""
Validators for gogooku5-dataset CLI configuration.

Performs comprehensive validation of:
- Environment dependencies (GPU, Python packages)
- JQuants API credentials
- File system permissions
- Feature compatibility (e.g., Futures API Premium requirement)
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

from .config import Config

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class Validator:
    """Configuration and environment validator."""

    def __init__(self, config: Config, strict: bool = False):
        """
        Initialize validator.

        Args:
            config: Configuration object to validate
            strict: If True, treat warnings as errors
        """
        self.config = config
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validations.

        Returns:
            (success, errors, warnings) tuple
        """
        self.errors = []
        self.warnings = []

        # Core validations
        self._validate_python_version()
        self._validate_directories()
        self._validate_dependencies()

        if self.config.command == "build":
            self._validate_build_config()
            self._validate_gpu()
            self._validate_jquants_credentials()
            self._validate_features()

        # In strict mode, warnings become errors
        if self.strict and self.warnings:
            self.errors.extend(self.warnings)
            self.warnings = []

        success = len(self.errors) == 0
        return success, self.errors, self.warnings

    def _validate_python_version(self) -> None:
        """Validate Python version (3.10+)."""
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 10):
            self.errors.append(f"Python 3.10+ required, found {major}.{minor}")

    def _validate_directories(self) -> None:
        """Validate output directories exist and are writable."""
        if hasattr(self.config, "output_dir"):
            output_dir = Path(self.config.output_dir)
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.errors.append(f"Cannot create output directory {output_dir}: {e}")
            elif not os.access(output_dir, os.W_OK):
                self.errors.append(f"Output directory {output_dir} is not writable")

    def _validate_dependencies(self) -> None:
        """Validate required Python packages."""
        # Map package names to import names
        required_packages = {
            "polars": "polars",
            "pyarrow": "pyarrow",
            "numpy": "numpy",
            "pandas": "pandas",
            "requests": "requests",
            "python-dotenv": "dotenv",
        }

        missing = []
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(package_name)

        if missing:
            self.errors.append(f"Missing required packages: {', '.join(missing)}")

    def _validate_build_config(self) -> None:
        """Validate build command configuration."""
        # Validate date range
        from datetime import datetime

        try:
            start = datetime.fromisoformat(self.config.start_date).date()
            end = datetime.fromisoformat(self.config.end_date).date()

            if start >= end:
                self.errors.append(f"Start date ({start}) must be before end date ({end})")

            # Warn if range is very large (>10 years)
            days = (end - start).days
            if days > 3650:
                self.warnings.append(
                    f"Date range is {days} days (~{days//365} years). " "Consider using chunked execution."
                )

        except ValueError as e:
            self.errors.append(f"Invalid date format: {e}")

        # Validate chunk mode
        if self.config.resume and self.config.chunk_mode == "full":
            self.errors.append("--resume requires chunked execution (--chunk-mode=chunks or auto)")

    def _validate_gpu(self) -> None:
        """Validate GPU availability if GPU mode enabled."""
        if not self.config.gpu_enabled:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                msg = "GPU-ETL enabled but CUDA not available"
                if self.strict:
                    self.errors.append(msg)
                else:
                    self.warnings.append(msg + " (will fall back to CPU)")
        except ImportError:
            self.warnings.append("GPU-ETL enabled but PyTorch not installed (cannot detect CUDA)")

        # Check cuDF/cuGraph for graph features
        if self.config.enable_graph and self.config.gpu_enabled:
            import importlib.util

            if importlib.util.find_spec("cudf") is None or importlib.util.find_spec("cugraph") is None:
                self.warnings.append(
                    "Graph features enabled but cuDF/cuGraph not installed. " "Graph features will be skipped."
                )

    def _validate_jquants_credentials(self) -> None:
        """Validate JQuants API credentials."""
        if not self.config.jquants_enabled:
            return

        email = self.config.jquants_email if hasattr(self.config, "jquants_email") else None
        password = self.config.jquants_password if hasattr(self.config, "jquants_password") else None

        if not email or not password:
            self.errors.append(
                "JQuants credentials missing. Set JQUANTS_AUTH_EMAIL and " "JQUANTS_AUTH_PASSWORD in .env"
            )

    def _validate_features(self) -> None:
        """Validate feature configuration."""
        # Futures requires Premium plan
        if self.config.futures_continuous:
            plan_tier = getattr(self.config, "jquants_plan_tier", "standard").lower()
            if plan_tier != "premium":
                self.warnings.append(
                    "Futures continuous features require Premium plan. "
                    f"Current plan: {plan_tier}. Features will be skipped."
                )

        # Warn if force_refresh and cache disabled
        if self.config.force_refresh and not getattr(self.config, "cache_enabled", True):
            self.warnings.append("--force-refresh has no effect when cache is disabled")


def validate_config(config: Config, strict: bool = False) -> None:
    """
    Validate configuration and raise exception if invalid.

    Args:
        config: Configuration to validate
        strict: Treat warnings as errors

    Raises:
        ValidationError: If validation fails
    """
    validator = Validator(config, strict=strict)
    success, errors, warnings = validator.validate_all()

    # Log warnings
    for warning in warnings:
        logger.warning(f"⚠️  {warning}")

    # Fail on errors
    if not success:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  ❌ {err}" for err in errors)
        raise ValidationError(error_msg)

    if warnings:
        logger.info(f"✅ Validation passed with {len(warnings)} warning(s)")
    else:
        logger.info("✅ Validation passed")


def check_environment(config: Config, strict: bool = False) -> Tuple[bool, dict]:
    """
    Comprehensive environment check (for --check command).

    Args:
        config: Configuration object
        strict: Strict mode (fail on any warning)

    Returns:
        (success, report_dict) tuple
    """
    report = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "dependencies": {},
        "gpu": {},
        "credentials": {},
        "file_system": {},
    }

    # Check dependencies
    packages = ["polars", "pyarrow", "numpy", "pandas", "torch", "cudf", "cugraph"]
    for package in packages:
        try:
            mod = __import__(package.replace("-", "_"))
            version = getattr(mod, "__version__", "unknown")
            report["dependencies"][package] = {
                "installed": True,
                "version": version,
            }
        except ImportError:
            report["dependencies"][package] = {
                "installed": False,
                "version": None,
            }

    # Check GPU
    try:
        import torch

        report["gpu"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"]["cuda_version"] = torch.version.cuda
            report["gpu"]["device_count"] = torch.cuda.device_count()
            report["gpu"]["device_name"] = torch.cuda.get_device_name(0)
        else:
            report["gpu"]["cuda_version"] = None
            report["gpu"]["device_count"] = 0
    except ImportError:
        report["gpu"]["cuda_available"] = False
        report["gpu"]["error"] = "PyTorch not installed"

    # Check credentials
    if hasattr(config, "jquants_email"):
        report["credentials"]["jquants_configured"] = bool(config.jquants_email and config.jquants_password)
        report["credentials"]["jquants_email"] = config.jquants_email[:5] + "***" if config.jquants_email else None
        report["credentials"]["plan_tier"] = getattr(config, "jquants_plan_tier", "unknown")

    # Check file system
    if hasattr(config, "output_dir"):
        output_dir = Path(config.output_dir)
        report["file_system"]["output_dir"] = str(output_dir)
        report["file_system"]["output_dir_exists"] = output_dir.exists()
        report["file_system"]["output_dir_writable"] = os.access(output_dir, os.W_OK) if output_dir.exists() else None

    # Determine success
    success = True
    if strict:
        # In strict mode, require GPU and all dependencies
        if not report["gpu"].get("cuda_available"):
            success = False
        required = ["polars", "torch", "cudf", "cugraph"]
        if not all(report["dependencies"].get(pkg, {}).get("installed") for pkg in required):
            success = False

    return success, report


def print_check_report(report: dict) -> None:
    """Pretty-print environment check report."""
    print("\n" + "=" * 70)
    print("🩺 Environment Check Report")
    print("=" * 70)

    print(f"\n📦 Python Version: {report['python_version']}")

    print("\n📚 Dependencies:")
    for pkg, info in sorted(report["dependencies"].items()):
        if info["installed"]:
            print(f"  ✅ {pkg:15s} {info['version']}")
        else:
            print(f"  ❌ {pkg:15s} NOT INSTALLED")

    print("\n🖥️  GPU:")
    gpu = report["gpu"]
    if gpu.get("cuda_available"):
        print(f"  ✅ CUDA Available:  {gpu['cuda_version']}")
        print(f"  ✅ Device Count:    {gpu['device_count']}")
        print(f"  ✅ Device Name:     {gpu['device_name']}")
    else:
        error = gpu.get("error", "CUDA not available")
        print(f"  ❌ {error}")

    print("\n🔑 Credentials:")
    cred = report["credentials"]
    if cred.get("jquants_configured"):
        print(f"  ✅ JQuants:         {cred['jquants_email']}")
        print(f"  ✅ Plan Tier:       {cred['plan_tier']}")
    else:
        print("  ❌ JQuants credentials not configured")

    print("\n💾 File System:")
    fs = report["file_system"]
    if fs.get("output_dir"):
        if fs.get("output_dir_exists"):
            writable = "writable" if fs["output_dir_writable"] else "NOT WRITABLE"
            print(f"  ✅ Output Dir:      {fs['output_dir']} ({writable})")
        else:
            print(f"  ⚠️  Output Dir:      {fs['output_dir']} (does not exist)")
    else:
        print("  ℹ️  Output Dir:      Not specified (use --output-dir or run build command)")

    print("\n" + "=" * 70)
