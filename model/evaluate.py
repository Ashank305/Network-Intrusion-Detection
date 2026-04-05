"""
Model Evaluation Module for Network Intrusion Detection System
Generates metrics, confusion matrix, ROC curve, feature importance
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from model.preprocess import COLUMN_NAMES


def get_all_metrics(y_true, y_pred):
    """Calculate all classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def get_classification_report(y_true, y_pred):
    """Get classification report as a string."""
    target_names = ['Normal', 'Attack']
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def plot_confusion_matrix(y_true, y_pred):
    """Create an interactive confusion matrix plotly figure."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Normal', 'Attack']
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create text annotations
    text = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm[i])):
            row.append(f"{cm[i][j]}<br>({cm_normalized[i][j]:.1%})")
        text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[
            [0, '#1a1a2e'],
            [0.5, '#e94560'],
            [1, '#0f3460']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Count", font=dict(color='#e0e0e0')),
            tickfont=dict(color='#e0e0e0')
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='Confusion Matrix',
            font=dict(size=20, color='#e0e0e0')
        ),
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        width=500,
        height=450
    )
    
    return fig


def plot_roc_curve(y_true, y_prob):
    """Create ROC curve plotly figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC Curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#e94560', width=3),
        fill='tozeroy',
        fillcolor='rgba(233, 69, 96, 0.15)'
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#555', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(
            text=f'ROC Curve (AUC = {roc_auc:.4f})',
            font=dict(size=20, color='#e0e0e0')
        ),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        font=dict(color='#e0e0e0'),
        legend=dict(
            x=0.5, y=0.05,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='#e0e0e0')
        ),
        width=600,
        height=450
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig, roc_auc


def plot_feature_importance(model, top_n=15):
    """Create feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features = [COLUMN_NAMES[i] for i in indices]
    top_importances = importances[indices]
    
    # Create gradient colors
    colors = [f'rgba(233, 69, 96, {0.4 + 0.6 * (1 - i/top_n)})' for i in range(top_n)]
    
    fig = go.Figure(data=go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(233, 69, 96, 0.8)', width=1)
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Feature Importances',
            font=dict(size=20, color='#e0e0e0')
        ),
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        font=dict(color='#e0e0e0'),
        height=500,
        width=700
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def plot_class_distribution(y, title="Class Distribution"):
    """Create a donut chart showing class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Normal' if u == 0 else 'Attack' for u in unique]
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=counts,
        hole=0.5,
        marker=dict(
            colors=['#0f3460', '#e94560'],
            line=dict(color='#16213e', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hoverinfo='label+value+percent'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#e0e0e0')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(font=dict(color='#e0e0e0')),
        width=450,
        height=400,
        annotations=[dict(
            text=f'{sum(counts):,}<br>Total',
            x=0.5, y=0.5,
            font_size=16, font_color='#e0e0e0',
            showarrow=False
        )]
    )
    
    return fig


def plot_metrics_radar(metrics):
    """Create a radar chart of model metrics."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(233, 69, 96, 0.2)',
        line=dict(color='#e94560', width=3),
        marker=dict(size=8, color='#e94560'),
        name='Model Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#e0e0e0')
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#e0e0e0', size=13)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        title=dict(
            text='Model Performance Radar',
            font=dict(size=20, color='#e0e0e0')
        ),
        width=500,
        height=450
    )
    
    return fig
