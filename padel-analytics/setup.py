"""
setup.py
========
Minimal setup script so the project can be installed in editable
mode with:  pip install -e .

This adds the src/ directory to the Python path permanently,
meaning you never need to set PYTHONPATH manually.

Install once:
    pip install -e .

Then run from anywhere:
    python main.py --input data/raw/match.mp4
"""

from setuptools import setup, find_packages

setup(
    name="padel-analytics",
    version="1.0.0",
    description="Padel Game Analytics — Shot Classification System",
    author="Your Name",
    python_requires=">=3.10",

    # Tell setuptools that source code lives in src/
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        "opencv-python>=4.9",
        "ultralytics>=8.2",
        "mediapipe>=0.10",
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.13",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "tqdm>=4.66",
        "loguru>=0.7",
        "pyyaml>=6.0",
    ],

    extras_require={
        "torch": ["torch>=2.3", "torchvision>=0.18"],
        "dev":   ["pytest>=8.2", "pytest-cov>=5.0", "jupyterlab>=4.2"],
    },

    entry_points={
        "console_scripts": [
            # Lets you run:  padel-analytics --input video.mp4
            "padel-analytics=main:main",
        ],
    },
)