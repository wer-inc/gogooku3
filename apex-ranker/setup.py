"""Setup configuration for APEX-Ranker package."""

from setuptools import find_packages, setup

setup(
    name="apex-ranker",
    version="0.1.0",
    description="APEX-Ranker: PatchTST-based ranking system for Japanese stock market",
    author="Wer Inc.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "polars>=0.19.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
