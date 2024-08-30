# src/evaluation.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, _test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix