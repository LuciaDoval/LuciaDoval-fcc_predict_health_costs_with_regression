# LuciaDoval-fcc_predict_health_costs_with_regression

This project aims to predict healthcare costs of individuals using a dataset that contains information about their age, body mass index (BMI), number of children, smoking status, sex, region, and medical expenses.

We use a **regression model** trained with **TensorFlow** and **Keras** to predict medical costs based on the provided features.

## **Project Objectives**

1. **Data Preprocessing**: Convert categorical variables into numeric data.
2. **Build a Regression Model**: Use a neural network model to predict healthcare costs.
3. **Train the Model**: Optimize the model to achieve a mean absolute error (MAE) under $3500 in predictions.
4. **Implement Early Stopping**: Apply the **Early Stopping** technique to prevent overfitting by stopping training if the error stops improving.

## **Project Structure**

```
.
├── README.md               # This file
└── fcc_predict_health_costs_with_regression.ipynb        
```

## **Dataset Description**

The dataset contains 1338 records with the following columns:

- **age**: Age of the person
- **sex**: Gender of the person (male, female)
- **bmi**: Body Mass Index (BMI)
- **children**: Number of children
- **smoker**: Indicates whether the person smokes (yes, no)
- **region**: Region where the person lives (northeast, northwest, southeast, southwest)
- **expenses**: Medical expenses (the target variable)

## **Dependencies**

To run the project, make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include the following libraries:

```
tensorflow==2.10.0
numpy
pandas
matplotlib
scikit-learn
```

## **How to Run the Project**

1. **Loading and Preprocessing the Data**:
   - The data is loaded from a CSV file (`insurance.csv`).
   - Categorical columns are converted to dummy variables (using `pd.get_dummies()`).
   - The data is normalized to ensure that all features are on the same scale.

2. **Dataset Split**:
   - The dataset is split into an 80% training set and a 20% test set.

3. **Building and Training the Model**:
   - A neural network model is built with Keras to predict healthcare expenses.
   - The model is trained using the training set, with **EarlyStopping** applied to halt training if the error starts to increase.

4. **Model Evaluation**:
   - The model is evaluated using the test set. The goal is to achieve a **Mean Absolute Error (MAE)** of less than 3500 to pass the challenge.

## **Python Code**

The `model.py` file contains the necessary code for loading, preprocessing the data, building the model, and training it.

1. **Data Loading and Preprocessing**:

```python
import pandas as pd

# Load the dataset
dataset = pd.read_csv('insurance.csv')

# Convert categorical columns to dummy variables
dataset = pd.get_dummies(dataset)

# Separate the target variable 'expenses'
train_labels = dataset.pop('expenses')
```

2. **Data Normalization**:

```python
# Normalize the data
train_stats = dataset.describe().transpose()

def normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = normalize(dataset)
```

3. **Model Building**:

```python
from tensorflow.keras import layers, models

# Build the neural network model
def build_model():
    model = models.Sequential([
      keras.layers.Dense(64, activation='relu', input_shape=(train_dataset.shape[1],)),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(16, activation='relu'),
      keras.layers.Dense(1) 
    ])
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

model = build_model()
```

4. **Training with EarlyStopping**:

```python
from tensorflow.keras.callbacks import EarlyStopping

# Implement EarlyStopping to stop training if the error doesn't improve
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(normed_train_data, train_labels, epochs=100, validation_split=0.2, callbacks=[early_stop])
```

5. **Model Evaluation**:

```python
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print(f"Mean Absolute Error on the test set: {mae} expenses")
```

## **Expected Results**

The model should predict healthcare costs with a **Mean Absolute Error (MAE)** less than $3500 to pass the challenge.
