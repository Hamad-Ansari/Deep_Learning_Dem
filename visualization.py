import plotly.graph_objects as go
import numpy as np

def create_confusion_matrix():
    """Create a sample confusion matrix"""
    classes = ['Cat', 'Dog', 'Bird', 'Car']
    cm = np.array([[45, 5, 2, 1], [3, 48, 2, 0], [1, 2, 44, 3], [2, 1, 3, 47]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, 
        x=classes, 
        y=classes, 
        colorscale="Viridis"
    ))
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_confidence_scores(classes, probabilities):
    """Create confidence score visualization"""
    fig = go.Figure(data=[
        go.Bar(x=probabilities, y=classes, orientation='h')
    ])
    fig.update_layout(title="Prediction Confidence Scores")
    return fig