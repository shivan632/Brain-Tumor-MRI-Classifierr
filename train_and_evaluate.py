import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from datetime import datetime
from data_preprocessing import get_data_generators
from models import build_custom_cnn, build_transfer_model

# Configuration
DATA_DIR = 'train-20250723T054459Z-1-001\\train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_DIR = 'saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def plot_history(history, model_name):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_training.png')
    plt.savefig(plot_path)
    plt.close()

def evaluate_model(model, val_gen, class_names):
    """Evaluate model and generate classification report."""
    # Get true labels and predictions
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

def main():
    # Load data  
    train_gen, val_gen, class_names = get_data_generators(
        DATA_DIR, IMG_SIZE, BATCH_SIZE
    )
    input_shape = IMG_SIZE + (3,)
    num_classes = len(class_names)
    
    # Custom CNN
    print("\nTraining Custom CNN...")
    cnn = build_custom_cnn(input_shape, num_classes)
    cnn_hist = cnn.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )
    cnn_path = os.path.join(MODEL_SAVE_DIR, 'custom_cnn.h5')
    cnn.save(cnn_path)
    plot_history(cnn_hist, 'Custom_CNN')
    evaluate_model(cnn, val_gen, class_names)
    
    # Transfer Learning
    print("\nTraining Transfer Learning Model...")
    tl = build_transfer_model(input_shape, num_classes)
    tl_hist = tl.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )
    tl_path = os.path.join(MODEL_SAVE_DIR, 'transfer_vgg16.h5')
    tl.save(tl_path)
    plot_history(tl_hist, 'Transfer_VGG16')
    evaluate_model(tl, val_gen, class_names)
    
    # Fine-tuning
    print("\nFine-tuning Transfer Learning Model...")
    tl_fine = build_transfer_model(input_shape, num_classes, fine_tune=True)
    fine_hist = tl_fine.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,  # Fewer epochs for fine-tuning
        verbose=1
    )
    fine_path = os.path.join(MODEL_SAVE_DIR, 'fine_tuned_vgg16.h5')
    tl_fine.save(fine_path)
    plot_history(fine_hist, 'Fine_Tuned_VGG16')
    evaluate_model(tl_fine, val_gen, class_names)

if __name__ == '__main__':
    main()