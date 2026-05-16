# Hybrid CNN–LSTM Architecture for Intelligent Machine Failure Prediction

An advanced deep learning framework for predictive maintenance using a hybrid CNN–LSTM architecture to predict industrial machine failures from multivariate sensor data.

---

## Overview

Unexpected machine failures in industrial environments lead to:

- Production downtime
- Increased maintenance costs
- Reduced operational efficiency
- Safety risks

This project introduces a **Hybrid CNN–LSTM predictive maintenance framework** capable of learning both:

- Spatial patterns using Convolutional Neural Networks (CNN)
- Temporal dependencies using Long Short-Term Memory Networks (LSTM)

The proposed model achieves up to **97.8% prediction accuracy**, outperforming traditional machine learning approaches such as Logistic Regression, Random Forest, and XGBoost.

---

## Key Features

- Hybrid CNN + LSTM deep learning architecture
- Predictive maintenance using industrial sensor data
- Time-series sequence modeling
- Sliding window sequence generation
- SMOTE-based class imbalance handling
- Min-Max feature normalization
- High accuracy and generalization performance
- GPU-accelerated TensorFlow implementation
- Real-time industrial deployment potential

---

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** containing industrial sensor readings such as:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The dataset is transformed into time-series sequences for temporal learning.

---

## Proposed Architecture

The framework combines:

### CNN Layer
Extracts local feature representations from sensor sequences.

### LSTM Layer
Captures long-term temporal dependencies and degradation patterns.

### Dense Output Layer
Performs binary classification:

- Failure
- Non-Failure

---

## Methodology

### Data Preprocessing

- Missing value handling
- Outlier removal
- One-hot encoding
- Min-Max normalization
- SMOTE balancing

### Sequence Generation

- Sliding window approach
- Sequence length = 10
- Time-series reshaping

### Model Pipeline

```text
Dataset → Preprocessing → Sequence Generation → CNN → LSTM → Dense Layer → Prediction
