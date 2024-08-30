# tests/test_evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.traffic_prediction.models.models import train_decision_tree_model
from src.evaluation import evaluate_model


def test_evaluate_model():
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = train_decision_tree_model(x_train, y_train)

    # Evaluate the model
    evaluation_result = evaluate_model(model, x_test, y_test)

    # Add specific assertions based on the expected behavior of your evaluate_model function
    # For example, you can check if certain metrics are present in the evaluation result
    assert 'accuracy' in evaluation_result, "Accuracy not found in the evaluation result"
    assert 'classification_report' in evaluation_result, "Classification report not found in the evaluation result"

    # You may want to check specific values or conditions based on your expected behavior

    # Example: Check if the accuracy is within a reasonable range
    assert 0.0 <= evaluation_result['accuracy'] <= 1.0, "Accuracy is outside the expected range"

    # Example: Check if the classification report has certain expected keys
    expected_keys = ['precision', 'recall', 'f1-score', 'support']
    assert all(key in evaluation_result['classification_report'] for key in expected_keys)

