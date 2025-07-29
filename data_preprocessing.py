import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    # Validate data directory
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    # Create generators
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_gen.class_indices.keys())
    
    return train_gen, val_gen, class_names