import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import cv2

def visualize_prediction(prediction_value):
    """Visualize prediction confidence as a gauge chart"""
    # Create a simple gauge visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create confidence levels
    confidence = prediction_value * 100
    
    # Create a horizontal bar chart showing confidence
    categories = ['Normal', 'Cancer']
    values = [100 - confidence, confidence]
    colors = ['green', 'red']
    
    bars = ax.barh(categories, values, color=colors, alpha=0.7)
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 1, i, f'{value:.1f}%', 
                va='center', fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Prediction Confidence', fontsize=16, fontweight='bold')
    
    # Add a vertical line at 50% for reference
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)

def visualize_model_performance(model_type):
    """Display model performance metrics"""
    # Simulated performance metrics based on model type
    if model_type == "InceptionV3 Transfer Learning":
        metrics = {
            'Accuracy': 0.94,
            'Precision': 0.91,
            'Recall': 0.89,
            'F1-Score': 0.90,
            'AUC': 0.95
        }
    else:
        metrics = {
            'Accuracy': 0.87,
            'Precision': 0.84,
            'Recall': 0.82,
            'F1-Score': 0.83,
            'AUC': 0.88
        }
    
    # Create a DataFrame for better display
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
    
    # Display as a table
    st.dataframe(df, use_container_width=True)
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), 
                  color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'{model_type} Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def visualize_activation_maps(model, image):
    """Visualize class activation maps (simplified version)"""
    st.subheader("Class Activation Maps")
    
    try:
        # For demonstration, create a heatmap overlay
        # In a real implementation, this would use GradCAM or similar techniques
        
        # Create a simulated heatmap
        heatmap = np.random.rand(224, 224)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Create overlay
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        im2 = ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Activation Map')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Overlay
        ax3.imshow(image)
        ax3.imshow(heatmap, alpha=0.4, cmap='jet')
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating activation maps: {str(e)}")
        st.info("Activation maps require a trained model. Using simulated visualization.")

def visualize_feature_maps(model, image):
    """Visualize feature maps from intermediate layers"""
    st.subheader("Feature Maps")
    
    try:
        # For demonstration, show simulated feature maps
        # In a real implementation, this would extract actual feature maps
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(8):
            # Create simulated feature map
            feature_map = np.random.rand(56, 56)
            feature_map = cv2.GaussianBlur(feature_map, (5, 5), 0)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating feature maps: {str(e)}")
        st.info("Feature maps require a trained model. Using simulated visualization.")

def plot_training_history():
    """Plot simulated training history"""
    # Simulated training data
    epochs = list(range(1, 21))
    
    # Simulated accuracy curves
    train_acc = [0.5 + 0.02*i + np.random.normal(0, 0.01) for i in epochs]
    val_acc = [0.5 + 0.018*i + np.random.normal(0, 0.015) for i in epochs]
    
    # Simulated loss curves
    train_loss = [2.0 - 0.08*i + np.random.normal(0, 0.05) for i in epochs]
    val_loss = [2.0 - 0.07*i + np.random.normal(0, 0.08) for i in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig
