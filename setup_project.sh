#!/bin/bash

echo "Creating project structure..."

mkdir -p padel-analytics/data/raw
mkdir -p padel-analytics/data/processed
mkdir -p padel-analytics/data/outputs
mkdir -p padel-analytics/data/annotations

mkdir -p padel-analytics/models/detection
mkdir -p padel-analytics/models/classification
mkdir -p padel-analytics/models/pretrained

mkdir -p padel-analytics/src
mkdir -p padel-analytics/notebooks
mkdir -p padel-analytics/tests

touch padel-analytics/src/config.py
touch padel-analytics/src/detection.py
touch padel-analytics/src/tracking.py
touch padel-analytics/src/pose_estimator.py
touch padel-analytics/src/shot_classifier.py
touch padel-analytics/src/analytics.py
touch padel-analytics/src/visualizer.py
touch padel-analytics/src/exporter.py
touch padel-analytics/src/utils.py

touch padel-analytics/notebooks/01_exploration.ipynb
touch padel-analytics/notebooks/02_model_training.ipynb
touch padel-analytics/notebooks/03_analytics_dashboard.ipynb

touch padel-analytics/tests/test_detection.py
touch padel-analytics/tests/test_classifier.py

touch padel-analytics/main.py
touch padel-analytics/requirements.txt
touch padel-analytics/README.md

echo "✅ Project structure created!"