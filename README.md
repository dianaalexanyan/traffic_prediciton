from src.data_preprocessing import load_data, preprocess_data

# Load data
df = load_data('data/Traffic.csv')

# Preprocess data
df = preprocess_data(df)
from src.exploratory_data_analysis import visualize_data

# Visualize data
visualize_data(df)
from src.models import train_decision_tree_model
from src.evaluation import evaluate_model

# Train a decision tree model
model = train_decision_tree_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)