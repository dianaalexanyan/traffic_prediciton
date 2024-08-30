# tests/test_exploratory_data_analysis.py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from src.traffic_prediction.exploratory.exploratory_data_analysis import visualize_data


def test_visualize_data():
    # Call the visualize_data function
    # You may need to pass relevant data or modify the function to load the data internally
    fig, axes = visualize_data()

    # Render the plot to check if it can be created without errors
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Check if the axes are instances of Matplotlib AxesSubplot
    assert isinstance(axes, plt.Axes), "Unexpected type for axes"

    # Add more specific tests based on your visualization logic
    # For example, you can check if certain elements (lines, bars, etc.) are present on the plot
    # You may need to adjust this based on the nature of your visualizations

    # Example: Check if a line plot is present
    assert any(isinstance(ax, plt.Line2D) for ax in axes.get_lines()), "Line plot not found in the visualization"

    # Example: Check if a bar plot is present
    assert any(isinstance(ax, plt.Rectangle) for ax in axes.patches), "Bar plot not found in the visualization"

    # Example: Check if the plot title is set
    assert fig.get_title() != "", "Plot title is not set"

    # Example: Check if the x-axis label is set
    assert axes.get_xlabel() != "", "X-axis label is not set"

    # Example: Check if the y-axis label is set
    assert axes.get_ylabel() != "", "Y-axis label is not set"
