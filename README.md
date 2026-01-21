# exoplanets-ml

This project applies multiple machine learning models to classify Kepler Objects of Interest (KOIs) into three categories:
	•	False Positive
	•	Candidate
	•	Confirmed Exoplanet

The goal is to evaluate and compare different modeling approaches while following reproducible, production-style ML practices.

⸻

Project Overview

The pipeline performs the following steps:
	1.	Securely downloads large astronomical datasets
	2.	Merges stellar and KOI information
	3.	Cleans and preprocesses the data
	4.	Engineers domain-specific features
	5.	Trains and evaluates multiple classification models
	6.	Generates performance metrics and visualizations

All steps are automated and reproducible via GitHub Actions.

⸻

Models Implemented

The following models are trained and evaluated:
	•	K-Nearest Neighbors (KNN)
	•	Support Vector Machine (SVM, RBF kernel)
	•	Random Forest (tuned)
	•	XGBoost
	•	Multilayer Perceptron (Neural Network)

Performance is assessed using:
	•	Confusion matrices
	•	Precision, recall, and F1-score
	•	Confidence vs. correctness analysis

### Repo Structure
exoplanets-ml/
│
├── src/
│   ├── main.py                # Full ML pipeline
│   └── download_drive_file.py # Secure dataset download
│
├── data/                      # Dataset (created at runtime)
├── results/                   # Generated plots and outputs
│
├── .github/
│   └── workflows/
│       └── run-exoplanets.yml # GitHub Actions workflow
│
├── requirements.txt           # Python dependencies
└── README.md
