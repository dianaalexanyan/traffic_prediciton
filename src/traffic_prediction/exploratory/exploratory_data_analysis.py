import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    """Visualize exploratory data analysis."""
    # Distribution of Traffic Situation
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Traffic Situation', data=df)
    plt.title('Distribution of Traffic Situation')
    plt.xlabel('Traffic Situation')
    plt.ylabel('Count')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Scatter plot between CarCount and BikeCount
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CarCount', y='BikeCount', data=df, hue='Traffic Situation', palette='viridis')
    plt.title('Scatter plot between CarCount and BikeCount')
    plt.xlabel('CarCount')
    plt.ylabel('BikeCount')
    plt.show()


