# tests/test_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.traffic_prediction.models.models import train_decision_tree_model

def test_train_decision_tree_model():
    # Create a simple dataset for testing


    data = {
        'Avg_Total': [10, 20, 30, 40, 50],
        'Avg_CarCount': [2, 4, 6, 8, 10],
        'Avg_BikeCount': [1, 2, 3, 4, 5],
        'Avg_BusCount': [0, 1, 2, 3, 4],
        'Avg_TruckCount': [0, 1, 1, 2, 2],
        'Traffic_Situation': ['low', 'low', 'normal', 'normal', 'high']
    }

    df = pd.DataFrame(data)

    # Split the data into features (X) and target variable (y)
    x = df[['Avg_Total', 'Avg_CarCount', 'Avg_BikeCount', 'Avg_BusCount', 'Avg_TruckCount']]
    y = df['Traffic_Situation']

    # Split the data into training and testing sets
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = train_decision_tree_model(X_train, y_train)

    # Add specific assertions based on the expected behavior of your train_decision_tree_model function
    assert model is not None, "Model is not created"
    assert str(type(model)) == "<class 'sklearn.tree._classes.DecisionTreeClassifier'>", "Model is not a DecisionTreeClassifier instance"
