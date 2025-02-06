# A2. Car Price Prediction Project
### By Lu Htoo Kyaw (ST124956)

This project predicts car prices using a Django-based web application integrated with a trained Polynomial Regression model. It includes Jupyter notebooks for exploratory data analysis (EDA) and model training. The training process is managed using MLflow, enabling experiment tracking, model versioning, and performance comparison. The entire setup is containerized with Docker for ease of deployment and reproducibility.

Polynomial Regression allows the model to capture complex relationships between car attributes and their prices, offering smoother predictions compared to linear models. The training process involves data preprocessing, feature scaling, polynomial feature transformation, k-fold cross-validation, and hyperparameter tuning. MLflow is used to log training metrics such as mean squared error (MSE) and R² score, helping to identify the best-performing model for deployment.

# Installation

## Clone the repository

```
git clone https://github.com/luhtookyaw/a2-car-price-prediction.git
```

## Install Python Packages
Go to app folder:

```
  cd app
```
```
  pip3 install -r requirements.txt
```

## Run Django Server In Local
Go to app folder:

```
  python3 manage.py runserver
```

## Build Docker Image From Scratch

Go to app folder:

```
  docker compose up -d
```

## Pull Image and Run
In root directory:

```
  docker compose up -d
```