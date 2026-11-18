
# PathSnap
### Distributed Map-Matching and Route Reconstruction Using Machine Learning

## Overview

PathSnap is a machine learning-based GPS map-matching and route reconstruction system. It addresses the problem of identifying the actual roads travelled by a vehicle from noisy GPS observations and an available road network.

GPS trajectories are not always accurate representations of the roads travelled. Signal interference, inaccurate location measurements, urban infrastructure, irregular sampling and GPS dropouts can cause recorded positions to deviate from the actual vehicle path.

PathSnap addresses this problem by generating candidate road segments for GPS observations, extracting relevant features, applying supervised machine learning models and reconstructing a coherent route using road-network connectivity and sequential decoding.

The project brings together data preprocessing, feature engineering, model experimentation, cross-validation, hyperparameter optimisation, route reconstruction, geographical visualisation and an evaluator-facing application.

## Project Objectives

- Process raw GPS trajectories and road-network data.
- Generate candidate road segments for individual GPS observations.
- Extract spatial, geometric and trajectory-related features.
- Train and compare multiple supervised machine learning models.
- Evaluate model performance using trip-wise cross-validation.
- Optimise model hyperparameters using Bayesian optimisation.
- Reconstruct realistic routes using Hidden Markov Models (HMM) and Viterbi decoding.
- Visualise GPS observations, road networks, reference routes and predicted routes.
- Support model inference and evaluation through an API and user interface.

## System Architecture

The PathSnap pipeline consists of the following stages:

1. **Data Processing:** Read and inspect the raw GPS trajectories and road-network files.
2. **Candidate Generation:** Identify possible road segments corresponding to each GPS observation.
3. **Feature Engineering:** Extract numerical features describing the relationship between GPS observations and candidate road segments.
4. **Model Training:** Train multiple classifiers to distinguish the reference road segment from incorrect candidates.
5. **Model Evaluation:** Compare models using grouped cross-validation and observation-level and route-level metrics.
6. **Hyperparameter Optimisation:** Use Bayesian optimisation to identify a suitable configuration for the selected model.
7. **Route Reconstruction:** Apply sequential decoding using road-network connectivity and HMM/Viterbi-based methods.
8. **Visualisation:** Generate geographical maps to compare observed GPS trajectories and reconstructed routes.
9. **Deployment:** Integrate the prediction pipeline with the backend API and evaluator-facing interface.

## Dataset

The project uses GPS trajectory data and a corresponding road network.

Each trip contains the following files:

| File | Description |
|---|---|
| `.track` | GPS observations containing longitude, latitude and timestamps |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed connections between road-network nodes |
| `.route` | Reference sequence of road-segment IDs |

The supplied dataset is organised into three partitions:

| Partition | Number of trips | Purpose |
|---|---:|---|
| Training | 80 | Model training, cross-validation and hyperparameter optimisation |
| Student testing | 10 | Development-stage evaluation and model comparison |
| Evaluator testing | 10 | Final evaluation |

The evaluator-testing partition is kept separate from model training, cross-validation and hyperparameter selection.

## Data Processing and Feature Engineering

The raw GPS observations and road-network information are processed to generate candidate road segments.

For each observation, candidate road segments are identified using spatial proximity and the configured candidate-generation constraints.

The feature-engineering stage extracts numerical information describing the relationship between the GPS observation and each candidate. These features are used as inputs to the supervised learning models.

Each candidate is associated with a binary target:

- `1`: The candidate corresponds to the reference road segment.
- `0`: The candidate does not correspond to the reference road segment.

The processed candidate-level dataset is used by the model training and evaluation pipelines.

## Machine Learning Models

Five supervised machine learning approaches are included in the model experimentation pipeline.

| Model | Algorithm family |
|---|---|
| Logistic Regression | Linear classification |
| Random Forest | Ensemble learning |
| HistGradientBoosting | Gradient boosting |
| Support Vector Machine (RBF) | Kernel-based classification |
| Multi-Layer Perceptron (MLP) | Neural network |

The models are configured and trained using the project's existing model definitions. Additional sampled experiments are implemented for MLP and SVM in the remaining-model training script.

Model selection considers generalisation, cross-validation consistency, classification performance, probability quality and route-level behaviour rather than relying solely on training accuracy.

## Model Evaluation and Cross-Validation

Model evaluation is carried out using five-fold GroupKFold cross-validation, with trip ID as the grouping variable.

Grouping by trip ensures that candidate rows belonging to the same GPS trajectory do not appear in both the training and validation portions of a fold.

For each fold, the model is trained on the training trips and evaluated on the held-out validation trips. The process is repeated across five folds, and the mean and standard deviation of the evaluation metrics are calculated.

The evaluation includes:

- Accuracy
- Precision
- Recall
- Macro-F1 score
- Log Loss, where applicable
- Observation-level Top-1 Accuracy
- Candidate Miss Rate
- Route Continuity
- Training Time
- Inference Time

The ten student-testing trips are used for development-stage evaluation after model selection. They are not used as the objective for cross-validation or Bayesian optimisation.

## Hyperparameter Optimisation

Bayesian hyperparameter optimisation is implemented using Optuna with the Tree-structured Parzen Estimator (TPE) sampler.

The search process explores selected model hyperparameters and evaluates configurations using grouped cross-validation on the training trips.

The optimisation objective is to maximise mean candidate-level Macro-F1 across the GroupKFold splits.

The optimisation pipeline retains trial configurations, objective values, fold-level results and the best-performing configuration. The tuned model is subsequently compared with its untuned counterpart using the same evaluation methodology.

The evaluator-testing partition is not used during model selection or hyperparameter optimisation.

## HMM and Viterbi Route Reconstruction

Independent classification of GPS observations can produce road-segment predictions that are individually plausible but disconnected from one another.

To address this, PathSnap incorporates sequential route reconstruction using Hidden Markov Models and Viterbi decoding.

The route reconstruction stage uses candidate predictions together with road-network connectivity and transition information to identify a more coherent sequence of road segments.

The implementation considers emission and transition information, missing candidates, GPS gaps and disconnected candidate transitions.

Classifier-only predictions can be compared with predictions obtained after sequential decoding to examine the effect of route reconstruction on accuracy and continuity.

## Geographical Visualisation

Folium is used for interactive geographical visualisation.

The visualisation components support the examination of:

- Original GPS observations.
- Relevant road-network geometry.
- Reference routes.
- Predicted road-segment sequences.

These visualisations help examine how candidate selection and sequential route reconstruction affect the final matched trajectory.

## Application and Deployment

PathSnap follows a two-laptop deployment architecture.

**Laptop 1 – Backend and Model**

- Hosts the trained classifier and required model configurations.
- Maintains road-network information.
- Runs the prediction and route-reconstruction pipeline.
- Exposes evaluation functionality through the backend API.

**Laptop 2 – Evaluator Interface**

- Displays the available evaluation trips.
- Allows the evaluator to select a trip.
- Sends the selected trip ID to the backend.
- Displays the returned prediction and evaluation results.

Ground-truth information used for evaluation is maintained on the server side and is not exposed through the evaluator interface.

## Repository Structure

```text
path-snap/
│
├── data/
│   ├── train/
│   ├── student_test/
│   └── evaluator_test/
│
├── src/
│   ├── data/
│   │   ├── eda.py
│   │   └── labels.py
│   │
│   ├── eval/
│   │   ├── cv.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── candidates.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── train_remaining.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── models/
│   ├── viz/
│   └── common.py
│
├── member_c_eval/
│   ├── pathsnap_eval/
│   │   ├── cross_validation.py
│   │   ├── evaluator.py
│   │   ├── models.py
│   │   ├── tuning.py
│   │   ├── data.py
│   │   ├── route_metrics.py
│   │   └── plots.py
│   │
│   ├── prepare_evaluation_data.py
│   ├── run_model_comparison.py
│   ├── run_bayesian_tuning.py
│   ├── compare_hmm_predictions.py
│   ├── config.py
│   └── requirements-evaluation.txt
│
├── tests/
│
├── pathsnap_final_metrics.py
├── results/
├── README.md
└── .gitignore
```

Raw datasets and generated experiment outputs are maintained separately from the source-code modules.

## Installation

The project can be executed using Python in an Ubuntu environment or Ubuntu through WSL2 on Windows.

Install the required system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Navigate to the repository:

```bash
cd /path/to/path-snap
```

Create and activate a virtual environment:

```bash
python3 -m venv ~/.venvs/pathsnap
source ~/.venvs/pathsnap/bin/activate
```

Install the evaluation dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r member_c_eval/requirements-evaluation.txt
```

The raw dataset partitions should be extracted into their corresponding directories before running the processing pipeline.

## Execution

### 1. Process the training data and train the models

```bash
python -m src.ml.train --train-root data/train --out results/ml --radius 50 --max-candidates 30 --max-label-distance 50 --gap-break 30 --seed 42
```

### 2. Train the additional sampled models

```bash
python -m src.ml.train_remaining
```

### 3. Prepare the evaluation datasets

```bash
python -m member_c_eval.prepare_evaluation_data
```

### 4. Run model comparison and cross-validation

```bash
python -m member_c_eval.run_model_comparison
```

### 5. Run Bayesian hyperparameter optimisation

```bash
python -m member_c_eval.run_bayesian_tuning
```

The evaluation scripts use the processed candidate-level training data and the student-testing candidate dataset.

## Experiment Outputs

The evaluation pipeline stores generated results under:

```text
results/ml/member_c_evaluation/
```

This directory contains:

- Consolidated model comparison tables.
- Individual cross-validation fold results.
- Model configurations.
- Model-selection summaries.
- Student-testing predictions.
- Confusion matrices and comparison plots.
- Bayesian optimisation trial records.
- Best hyperparameter configurations.
- Tuned model artifacts.
- Tuned-versus-untuned comparisons.

These outputs support the analysis and reporting of the experiments.

## Reproducibility and Data Handling

- Random seeds and model configurations are retained in the experiment pipeline.
- Cross-validation uses trip IDs as grouping variables.
- The student-testing partition is kept separate from the training folds.
- The evaluator-testing partition is reserved for final evaluation.
- API keys, passwords and other private credentials must not be committed.
- Raw datasets and large generated artifacts should be managed separately from source code in accordance with repository size and submission requirements.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Optuna
- Matplotlib
- Joblib
- Folium
- FastAPI
- HTML, CSS and JavaScript for the evaluator interface

## Project Outcome

PathSnap combines candidate-based machine learning with road-network-aware sequential route reconstruction to transform noisy GPS observations into plausible road-level trajectories.

The project brings together model experimentation, systematic evaluation, hyperparameter optimisation and geographical visualisation in an end-to-end distributed map-matching workflow.
