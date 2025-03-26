# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms with hyperparameter optimization.

## 📋 Description

This repository contains a comprehensive machine learning solution for the famous Kaggle Titanic survival prediction competition. The project implements multiple classification algorithms with hyperparameter optimization to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, fare, cabin, and more.

The solution employs a systematic approach:
1. Data exploration and visualization
2. Feature engineering and preprocessing
3. Model training with hyperparameter optimization
4. Model evaluation and selection
5. Prediction generation for test data

## ✨ Features

- **Exploratory Data Analysis**: Comprehensive analysis of the Titanic dataset with visualizations to understand feature relationships and survival patterns
- **Feature Engineering**: Creation of new features like family size, title extraction from names, and family survival correlation
- **Hyperparameter Optimization**: Uses Hyperopt library to find optimal parameters for each model
- **Multiple Classification Algorithms**:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
  - K-Nearest Neighbors
  - Support Vector Machine (implementation available but not used in main script)
  - Neural Network with Keras (implementation available but not used in main script)
- **Model Comparison**: Automatic selection of the best performing model for final predictions

## 🛠️ Setup

### Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - hyperopt
  - xgboost
  - keras (optional, for neural network implementation)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/KaggleTitanticSurvivalClassify.git
cd KaggleTitanticSurvivalClassify
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn hyperopt xgboost keras
```

## 📊 Usage

1. Run the main script to train models and generate predictions:
```bash
python titanicPredictSurvival.py
```

2. The script will:
   - Load and preprocess the training and test data
   - Perform feature engineering
   - Train multiple models with hyperparameter optimization
   - Select the best performing model
   - Generate predictions for the test set
   - Save predictions to `test_set_prediction.csv`

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
