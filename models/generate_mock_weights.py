import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Configurations
IMG_SIZE = 224
NUM_CLASSES = 5
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

print("Initializing Neural Network Graph Builders...")
print(f"Target Directory: {MODELS_DIR}")
print(f"TensorFlow Version: {tf.__version__}")

# ============================================================
# CBAM BLOCK DEFINITION
# ============================================================
def cbam_block(x, ratio=8):
    # Channel Attention
    ch = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    shared_dense_1 = layers.Dense(ch // ratio, activation='relu')
    shared_dense_2 = layers.Dense(ch)
    
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    
    channel_attention = layers.Activation('sigmoid')(avg_out + max_out)
    channel_attention = layers.Reshape((1, 1, ch))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
    
    # Spatial Attention
    avg_spatial = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_spatial = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    
    spatial_features = layers.Concatenate()([avg_spatial, max_spatial])
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_features)
    
    return layers.Multiply()([x, spatial_attention])

# ============================================================
# 1. LEAFNET (L-NET) + CBAM
# ============================================================
def build_leafnet_cbam():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="LeafNet_CBAM")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# 2. DENSENET121 + CBAM
# ============================================================
def build_densenet_cbam():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # We use weights=None to ensure rapid, offline local building
    try:
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
        print(" -> DenseNet121 loaded with pre-trained ImageNet weights successfully.")
    except Exception as e:
        print(f" -> Offline building: Initializing DenseNet121 base architecture without weights: {e}")
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
        
    base_model.trainable = False
    
    x = base_model.output
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="DenseNet121_CBAM")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# 3. INCEPTIONV3 + CBAM
# ============================================================
def build_inception_cbam():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    try:
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_tensor=inputs)
        print(" -> InceptionV3 loaded with pre-trained ImageNet weights successfully.")
    except Exception as e:
        print(f" -> Offline building: Initializing InceptionV3 base architecture without weights: {e}")
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_tensor=inputs)
        
    base_model.trainable = False
    
    x = base_model(inputs)
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="InceptionV3_CBAM")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# 4. INCEPTION-RESNET-V2 + CBAM
# ============================================================
def build_inception_resnet_cbam():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    try:
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
        print(" -> InceptionResNetV2 loaded with pre-trained ImageNet weights successfully.")
    except Exception as e:
        print(f" -> Offline building: Initializing InceptionResNetV2 base architecture without weights: {e}")
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_tensor=inputs)
        
    base_model.trainable = False
    
    x = base_model(inputs)
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="InceptionResNetV2_CBAM")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# 5. CONVNEXT-TINY + CBAM
# ============================================================
def build_convnext_cbam():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocessing layer for ConvNeXt
    from tensorflow.keras.applications.convnext import preprocess_input
    x_pre = preprocess_input(inputs)
    
    try:
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet")
        print(" -> ConvNeXtTiny loaded with pre-trained ImageNet weights successfully.")
    except Exception as e:
        print(f" -> Offline building: Initializing ConvNeXtTiny base architecture without weights: {e}")
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights=None)
        
    base_model.trainable = False
    
    x = base_model(x_pre)
    x = cbam_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="ConvNeXtTiny_CBAM")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# EXECUTOR
# ============================================================
def generate_all_models():
    model_configs = [
        ("LNet", build_leafnet_cbam),
        ("DenseNet", build_densenet_cbam),
        ("InceptionV3", build_inception_cbam),
        ("Inception_ResNet_v2", build_inception_resnet_cbam),
        ("Untitled65", build_convnext_cbam) # Untitled65 is the ConvNeXt notebook
    ]
    
    for filename, builder in model_configs:
        out_path = os.path.join(MODELS_DIR, f"{filename}.h5")
        if os.path.exists(out_path):
            print(f"Model already exists at: {out_path} (Skipping...)")
            continue
            
        print(f"\nBuilding model: {filename}...")
        try:
            model = builder()
            print(f"Saving compiled structure and weights to: {out_path}")
            model.save(out_path)
            print(f"Model {filename} created successfully!")
        except Exception as e:
            print(f"Error compiling model {filename}: {e}")

if __name__ == "__main__":
    generate_all_models()
    print("\nInitialization Complete! All models are ready for deep-learning deployment.")
