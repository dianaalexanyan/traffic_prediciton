# src/models.py
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_model(x, y):
    """Train a Decision Tree model."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    return model