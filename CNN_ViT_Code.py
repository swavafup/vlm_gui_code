"""
Comprehensive CNN vs Transformer comparison for tunnel crack detection.

Dataset structure:
    DATA_DIR/
        crack/
            image_1.jpg
            image_2.jpg
            ...
        non crack/
            image_1.jpg
            image_2.jpg
            ...

This script includes:
    - All CNN models shown in your screenshot:
        Xception, VGG16, VGG19,
        ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
        InceptionV3, InceptionResNetV2,
        MobileNet, MobileNetV2,
        DenseNet121, DenseNet169, DenseNet201,
        NASNetMobile, NASNetLarge,
        EfficientNetB0-B7,
        EfficientNetV2B0, B1, B2, B3, S, M, L,
        ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge.

    - Extra Keras Applications CNNs:
        MobileNetV3Small, MobileNetV3Large.

    - Transformer-based models:
        ViT, DeiT, Swin, BEiT, MobileViT.

    - Evaluation:
        Accuracy, precision, recall/sensitivity, specificity,
        F1-score, F2-score, ROC-AUC, PR-AUC,
        balanced accuracy, MCC, Cohen kappa, log loss.

    - Computational analysis:
        Head training time,
        fine-tuning time,
        total training time,
        inference time,
        inference time per image,
        images per second,
        total parameters,
        trainable parameters,
        non-trainable parameters,
        estimated float32 model size.

    - Graphs:
        Accuracy/loss curves,
        ROC-AUC/PR-AUC curves,
        confusion matrices,
        ROC curves,
        precision-recall curves,
        model comparison bar charts,
        grouped metric comparison,
        CNN vs transformer family comparison,
        F1 vs parameter complexity,
        F1 vs inference time,
        ROC-AUC vs parameter complexity,
        radar chart for top models.

Author: Generated for tunnel crack detection research.
"""

# ============================================================
# 0. Compatibility settings
# ============================================================

import os

# Important for Hugging Face TensorFlow models with recent TensorFlow/Keras versions.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# 1. Imports
# ============================================================

import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications as ka

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

try:
    from transformers import TFAutoModel, AutoImageProcessor
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False
    print("WARNING: Hugging Face transformers could not be imported.")
    print("Transformer models will be skipped.")
    print("Import error:", e)


# ============================================================
# 2. Configuration
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # Change this path only
    # --------------------------------------------------------
    "DATA_DIR": r"D:\My Research 2026\swavaf",

    # --------------------------------------------------------
    # Image settings
    # Your original images are 225 x 225.
    # Most pretrained CNN/transformer models use 224 x 224.
    # --------------------------------------------------------
    "IMG_SIZE": (224, 224),
    "CHANNELS": 3,

    # --------------------------------------------------------
    # Splitting
    # --------------------------------------------------------
    "VAL_SIZE": 0.15,
    "TEST_SIZE": 0.15,
    "RANDOM_SEED": 42,

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------
    "BATCH_SIZE": 16,
    "HEAD_EPOCHS": 10,
    "FINE_TUNE_EPOCHS": 10,
    "USE_FINE_TUNING": True,

    "HEAD_LR": 1e-4,
    "FINE_TUNE_LR": 1e-5,

    "EARLY_STOPPING_PATIENCE": 5,

    # --------------------------------------------------------
    # Pretrained weights
    # Use "imagenet" for Keras CNNs.
    # Use None if you want to train CNNs from scratch.
    # --------------------------------------------------------
    "CNN_WEIGHTS": "imagenet",

    # --------------------------------------------------------
    # Fine-tuning depth
    # --------------------------------------------------------
    "CNN_FINE_TUNE_LAST_N_LAYERS": 30,

    # Transformer fine-tuning is memory-heavy.
    # False means only the transformer classification head is trained.
    # True means the full transformer backbone is fine-tuned.
    "FINE_TUNE_TRANSFORMERS": False,

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------
    "CLASSIFICATION_THRESHOLD": 0.5,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    "OUTPUT_DIR": "tunnel_crack_cnn_transformer_results",
}


# ============================================================
# 3. Model lists
# ============================================================

CNN_MODELS_TO_RUN = [
    "Xception",

    "VGG16",
    "VGG19",

    "ResNet50",
    "ResNet50V2",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",

    "InceptionV3",
    "InceptionResNetV2",

    "MobileNet",
    "MobileNetV2",

    # Extra Keras Applications CNNs
    "MobileNetV3Small",
    "MobileNetV3Large",

    "DenseNet121",
    "DenseNet169",
    "DenseNet201",

    "NASNetMobile",
    "NASNetLarge",

    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",

    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "EfficientNetV2L",

    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtXLarge",
]


TRANSFORMER_MODELS_TO_RUN = [
    # Vision Transformer family
    "ViT_B16_IN21K",
    "ViT_B32_IN21K",
    "ViT_L16_IN21K",

    # DeiT family
    "DeiT_Tiny",
    "DeiT_Small",
    "DeiT_Base",

    # Swin Transformer family
    "Swin_Tiny",
    "Swin_Small",
    "Swin_Base",

    # BEiT family
    "BEiT_Base",

    # MobileViT family
    "MobileViT_XXS",
    "MobileViT_XS",
    "MobileViT_Small",
]


TRANSFORMER_REGISTRY = {
    "ViT_B16_IN21K": {
        "checkpoint": "google/vit-base-patch16-224-in21k",
        "family": "ViT",
    },
    "ViT_B32_IN21K": {
        "checkpoint": "google/vit-base-patch32-224-in21k",
        "family": "ViT",
    },
    "ViT_L16_IN21K": {
        "checkpoint": "google/vit-large-patch16-224-in21k",
        "family": "ViT",
    },

    "DeiT_Tiny": {
        "checkpoint": "facebook/deit-tiny-patch16-224",
        "family": "DeiT",
    },
    "DeiT_Small": {
        "checkpoint": "facebook/deit-small-patch16-224",
        "family": "DeiT",
    },
    "DeiT_Base": {
        "checkpoint": "facebook/deit-base-patch16-224",
        "family": "DeiT",
    },

    "Swin_Tiny": {
        "checkpoint": "microsoft/swin-tiny-patch4-window7-224",
        "family": "Swin",
    },
    "Swin_Small": {
        "checkpoint": "microsoft/swin-small-patch4-window7-224",
        "family": "Swin",
    },
    "Swin_Base": {
        "checkpoint": "microsoft/swin-base-patch4-window7-224",
        "family": "Swin",
    },

    "BEiT_Base": {
        "checkpoint": "microsoft/beit-base-patch16-224",
        "family": "BEiT",
    },

    "MobileViT_XXS": {
        "checkpoint": "apple/mobilevit-xx-small",
        "family": "MobileViT",
    },
    "MobileViT_XS": {
        "checkpoint": "apple/mobilevit-x-small",
        "family": "MobileViT",
    },
    "MobileViT_Small": {
        "checkpoint": "apple/mobilevit-small",
        "family": "MobileViT",
    },
}


# ============================================================
# 4. Reproducibility and GPU setup
# ============================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        print(f"Detected {len(gpus)} GPU(s).")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("Could not set GPU memory growth:", e)
    else:
        print("No GPU detected. Training all models on CPU will be very slow.")


set_global_seed(CONFIG["RANDOM_SEED"])
setup_gpu_memory_growth()


# ============================================================
# 5. Output folders
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(CONFIG["OUTPUT_DIR"]) / f"run_{RUN_ID}"

PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "saved_models"
TABLES_DIR = OUTPUT_DIR / "tables"
HISTORIES_DIR = OUTPUT_DIR / "histories"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
ERRORS_DIR = OUTPUT_DIR / "errors"

for folder in [
    OUTPUT_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    HISTORIES_DIR,
    PREDICTIONS_DIR,
    ERRORS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=4)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# 6. Dataset loading
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_folder_name(name):
    return (
        name.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .strip()
    )


def discover_binary_class_folders(data_dir):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_path}")

    folders = [p for p in data_path.iterdir() if p.is_dir()]

    positive_candidates = []
    negative_candidates = []

    for folder in folders:
        name = normalize_folder_name(folder.name)
        compact = name.replace(" ", "")

        if name in ["crack", "cracks"] or compact in ["crack", "cracks"]:
            positive_candidates.append(folder)

        if ("non" in name and "crack" in name) or ("non" in compact and "crack" in compact):
            negative_candidates.append(folder)

    if len(positive_candidates) == 0:
        raise ValueError("Could not find the crack folder. Name it 'crack'.")

    if len(negative_candidates) == 0:
        raise ValueError("Could not find the non-crack folder. Name it 'non crack', 'non_crack', or 'non-crack'.")

    crack_dir = positive_candidates[0]
    non_crack_dir = negative_candidates[0]

    print("Detected crack folder:", crack_dir)
    print("Detected non-crack folder:", non_crack_dir)

    return crack_dir, non_crack_dir


def collect_image_paths_and_labels(data_dir):
    crack_dir, non_crack_dir = discover_binary_class_folders(data_dir)

    image_paths = []
    labels = []

    # Label convention:
    #   1 = crack
    #   0 = non-crack
    for path in crack_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(path))
            labels.append(1)

    for path in non_crack_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(path))
            labels.append(0)

    image_paths = np.array(image_paths)
    labels = np.array(labels).astype(np.int32)

    if len(image_paths) == 0:
        raise ValueError("No images found. Check your folder names and image extensions.")

    print("\nDataset summary")
    print("Total images:", len(image_paths))
    print("Crack images:", int(np.sum(labels == 1)))
    print("Non-crack images:", int(np.sum(labels == 0)))

    return image_paths, labels


image_paths, labels = collect_image_paths_and_labels(CONFIG["DATA_DIR"])


# ============================================================
# 7. Train/validation/test split
# ============================================================

def make_stratified_splits(paths, y, val_size, test_size, seed):
    temp_size = val_size + test_size

    train_paths, temp_paths, y_train, y_temp = train_test_split(
        paths,
        y,
        test_size=temp_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )

    relative_test_size = test_size / temp_size

    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths,
        y_temp,
        test_size=relative_test_size,
        random_state=seed,
        stratify=y_temp,
        shuffle=True,
    )

    return train_paths, val_paths, test_paths, y_train, y_val, y_test


train_paths, val_paths, test_paths, y_train, y_val, y_test = make_stratified_splits(
    image_paths,
    labels,
    CONFIG["VAL_SIZE"],
    CONFIG["TEST_SIZE"],
    CONFIG["RANDOM_SEED"],
)

print("\nSplit summary")
print("Train:", len(train_paths), "Crack:", int(np.sum(y_train == 1)), "Non-crack:", int(np.sum(y_train == 0)))
print("Val:  ", len(val_paths), "Crack:", int(np.sum(y_val == 1)), "Non-crack:", int(np.sum(y_val == 0)))
print("Test: ", len(test_paths), "Crack:", int(np.sum(y_test == 1)), "Non-crack:", int(np.sum(y_test == 0)))


# ============================================================
# 8. Class weights
# ============================================================

classes = np.array([0, 1])
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

CLASS_WEIGHTS = {
    int(cls): float(weight)
    for cls, weight in zip(classes, class_weight_values)
}

print("\nClass weights:", CLASS_WEIGHTS)


# ============================================================
# 9. tf.data pipeline
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def decode_resize_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, CONFIG["IMG_SIZE"], method="bilinear")
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.float32)

    return image, label


def make_dataset(paths, y, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=CONFIG["RANDOM_SEED"],
            reshuffle_each_iteration=True,
        )

    ds = ds.map(decode_resize_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = make_dataset(train_paths, y_train, CONFIG["BATCH_SIZE"], training=True)
val_ds = make_dataset(val_paths, y_val, CONFIG["BATCH_SIZE"], training=False)
test_ds = make_dataset(test_paths, y_test, CONFIG["BATCH_SIZE"], training=False)


# ============================================================
# 10. Data augmentation
# ============================================================

def get_data_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )


# ============================================================
# 11. CNN registry from Keras Applications
# ============================================================

CNN_SPECS = [
    ("Xception", "xception"),

    ("VGG16", "vgg16"),
    ("VGG19", "vgg19"),

    ("ResNet50", "resnet"),
    ("ResNet101", "resnet"),
    ("ResNet152", "resnet"),

    ("ResNet50V2", "resnet_v2"),
    ("ResNet101V2", "resnet_v2"),
    ("ResNet152V2", "resnet_v2"),

    ("InceptionV3", "inception_v3"),
    ("InceptionResNetV2", "inception_resnet_v2"),

    ("MobileNet", "mobilenet"),
    ("MobileNetV2", "mobilenet_v2"),
    ("MobileNetV3Small", "mobilenet_v3"),
    ("MobileNetV3Large", "mobilenet_v3"),

    ("DenseNet121", "densenet"),
    ("DenseNet169", "densenet"),
    ("DenseNet201", "densenet"),

    ("NASNetMobile", "nasnet"),
    ("NASNetLarge", "nasnet"),

    ("EfficientNetB0", "efficientnet"),
    ("EfficientNetB1", "efficientnet"),
    ("EfficientNetB2", "efficientnet"),
    ("EfficientNetB3", "efficientnet"),
    ("EfficientNetB4", "efficientnet"),
    ("EfficientNetB5", "efficientnet"),
    ("EfficientNetB6", "efficientnet"),
    ("EfficientNetB7", "efficientnet"),

    ("EfficientNetV2B0", "efficientnet_v2"),
    ("EfficientNetV2B1", "efficientnet_v2"),
    ("EfficientNetV2B2", "efficientnet_v2"),
    ("EfficientNetV2B3", "efficientnet_v2"),
    ("EfficientNetV2S", "efficientnet_v2"),
    ("EfficientNetV2M", "efficientnet_v2"),
    ("EfficientNetV2L", "efficientnet_v2"),

    ("ConvNeXtTiny", "convnext"),
    ("ConvNeXtSmall", "convnext"),
    ("ConvNeXtBase", "convnext"),
    ("ConvNeXtLarge", "convnext"),
    ("ConvNeXtXLarge", "convnext"),
]


def build_cnn_registry():
    registry = {}

    for model_name, module_name in CNN_SPECS:
        model_class = getattr(ka, model_name, None)
        module = getattr(ka, module_name, None)

        if model_class is None:
            print(f"WARNING: {model_name} is not available in this TensorFlow/Keras installation.")
            continue

        preprocess_fn = None
        if module is not None:
            preprocess_fn = getattr(module, "preprocess_input", None)

        if preprocess_fn is None:
            preprocess_fn = lambda x: x

        registry[model_name] = {
            "class": model_class,
            "preprocess": preprocess_fn,
        }

    return registry


CNN_REGISTRY = build_cnn_registry()


# ============================================================
# 12. Metrics and compile
# ============================================================

def get_keras_metrics():
    threshold = CONFIG["CLASSIFICATION_THRESHOLD"]

    return [
        keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
        keras.metrics.Precision(name="precision", thresholds=threshold),
        keras.metrics.Recall(name="recall", thresholds=threshold),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=get_keras_metrics(),
    )

    return model


# ============================================================
# 13. Parameter counting
# ============================================================

def count_params_from_weights(weights):
    total = 0
    for weight in weights:
        total += int(tf.keras.backend.count_params(weight))
    return total


def get_parameter_summary(model):
    trainable_params = count_params_from_weights(model.trainable_weights)
    non_trainable_params = count_params_from_weights(model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "total_params_million": total_params / 1_000_000.0,
        "trainable_params_million": trainable_params / 1_000_000.0,
        "non_trainable_params_million": non_trainable_params / 1_000_000.0,
        "estimated_float32_model_size_mb": total_params * 4 / (1024 ** 2),
    }


def get_backbone_layer_count(model):
    if hasattr(model, "base_model_ref"):
        return len(model.base_model_ref.layers)

    if hasattr(model, "backbone_ref"):
        try:
            return len(model.backbone_ref.layers)
        except Exception:
            return np.nan

    return np.nan


# ============================================================
# 14. CNN model builder
# ============================================================

def build_cnn_model(model_name):
    if model_name not in CNN_REGISTRY:
        raise ValueError(f"{model_name} is not available in CNN_REGISTRY.")

    model_info = CNN_REGISTRY[model_name]
    base_class = model_info["class"]
    preprocess_fn = model_info["preprocess"]

    inputs = keras.Input(
        shape=(CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], CONFIG["CHANNELS"]),
        name="image_input",
    )

    x = get_data_augmentation()(inputs)
    x = layers.Lambda(preprocess_fn, name=f"{model_name}_preprocessing")(x)

    base_model = base_class(
        include_top=False,
        weights=CONFIG["CNN_WEIGHTS"],
        input_shape=(CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], CONFIG["CHANNELS"]),
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="head_batch_norm_1")(x)
    x = layers.Dropout(0.35, name="head_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.BatchNormalization(name="head_batch_norm_2")(x)
    x = layers.Dropout(0.25, name="head_dropout_2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="crack_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.base_model_ref = base_model

    return model


# ============================================================
# 15. Transformer preprocessing and model builder
# ============================================================

def load_hf_image_processor(checkpoint):
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(processor, "image_std", [0.5, 0.5, 0.5])
    except Exception:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if not isinstance(mean, (list, tuple)):
        mean = [float(mean)] * 3

    if not isinstance(std, (list, tuple)):
        std = [float(std)] * 3

    return [float(v) for v in mean], [float(v) for v in std]


def make_hf_preprocess_layer(mean, std, name):
    mean_tensor = tf.constant(mean, dtype=tf.float32)
    std_tensor = tf.constant(std, dtype=tf.float32)

    def preprocess(x):
        x = tf.cast(x, tf.float32)
        x = tf.image.resize(x, CONFIG["IMG_SIZE"])
        x = x / 255.0
        x = (x - mean_tensor) / std_tensor

        # Hugging Face TensorFlow vision models generally expect:
        # [batch, channels, height, width]
        x = tf.transpose(x, perm=[0, 3, 1, 2])

        return x

    return layers.Lambda(preprocess, name=name)


def load_hf_backbone(checkpoint):
    if not HF_AVAILABLE:
        raise ImportError("transformers is not installed or could not be imported.")

    try:
        backbone = TFAutoModel.from_pretrained(checkpoint)
        return backbone
    except Exception as e1:
        print(f"Standard TensorFlow loading failed for {checkpoint}.")
        print("Trying from_pt=True conversion fallback.")
        print("Original loading error:", e1)

        backbone = TFAutoModel.from_pretrained(checkpoint, from_pt=True)
        return backbone


def extract_hf_features(outputs):
    """
    Generic feature extractor for Hugging Face vision model outputs.
    Handles pooled output, sequence output, and spatial output.
    """

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    hidden = None

    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        hidden = outputs[0]

    if hidden is None:
        raise ValueError("Could not extract features from Hugging Face model output.")

    rank = len(hidden.shape)

    if rank == 3:
        # Sequence format: [batch, tokens, channels]
        return tf.reduce_mean(hidden, axis=1)

    if rank == 4:
        shape = hidden.shape.as_list()

        # Many HF vision backbones use channels-first:
        # [batch, channels, height, width]
        if shape[1] is not None and shape[1] > 8:
            return tf.reduce_mean(hidden, axis=[2, 3])

        # Channels-last fallback:
        # [batch, height, width, channels]
        return tf.reduce_mean(hidden, axis=[1, 2])

    if rank == 2:
        return hidden

    raise ValueError(f"Unsupported hidden state rank: {rank}")


def build_transformer_model(model_name):
    if not HF_AVAILABLE:
        raise ImportError("Hugging Face transformers is not available.")

    if model_name not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Unknown transformer model: {model_name}")

    checkpoint = TRANSFORMER_REGISTRY[model_name]["checkpoint"]

    mean, std = load_hf_image_processor(checkpoint)

    inputs = keras.Input(
        shape=(CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], CONFIG["CHANNELS"]),
        name="image_input",
    )

    x = get_data_augmentation()(inputs)
    pixel_values = make_hf_preprocess_layer(
        mean=mean,
        std=std,
        name=f"{model_name}_hf_preprocessing",
    )(x)

    backbone = load_hf_backbone(checkpoint)
    backbone.trainable = bool(CONFIG["FINE_TUNE_TRANSFORMERS"])

    outputs = backbone(pixel_values=pixel_values, training=False)
    features = extract_hf_features(outputs)

    x = layers.BatchNormalization(name="head_batch_norm_1")(features)
    x = layers.Dropout(0.35, name="head_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.BatchNormalization(name="head_batch_norm_2")(x)
    x = layers.Dropout(0.25, name="head_dropout_2")(x)

    predictions = layers.Dense(1, activation="sigmoid", name="crack_probability")(x)

    model = keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.backbone_ref = backbone

    return model


# ============================================================
# 16. Fine-tuning helpers
# ============================================================

def unfreeze_cnn_last_layers(model, last_n_layers):
    if not hasattr(model, "base_model_ref"):
        return model

    base_model = model.base_model_ref
    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-last_n_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    return model


def unfreeze_transformer_backbone(model):
    if hasattr(model, "backbone_ref"):
        model.backbone_ref.trainable = True

    return model


# ============================================================
# 17. Callbacks
# ============================================================

def get_callbacks(model_name, stage):
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{model_name}_{stage}_best.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=CONFIG["EARLY_STOPPING_PATIENCE"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(HISTORIES_DIR / f"{model_name}_{stage}_training_log.csv"),
            append=False,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks


# ============================================================
# 18. Training history plots
# ============================================================

def merge_histories(histories):
    merged = {}

    for history in histories:
        if history is None:
            continue

        for key, values in history.history.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(values)

    return merged


def plot_metric_curve(history_dict, model_name, train_key, val_key, title, ylabel, filename):
    if train_key not in history_dict and val_key not in history_dict:
        return

    n_epochs = len(history_dict.get("loss", []))
    epochs = range(1, n_epochs + 1)

    plt.figure(figsize=(8, 6))

    if train_key in history_dict:
        plt.plot(epochs, history_dict[train_key], label=f"Training {ylabel}")

    if val_key in history_dict:
        plt.plot(epochs, history_dict[val_key], label=f"Validation {ylabel}")

    plt.title(f"{model_name} - {title}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_training_history(history_dict, model_name):
    if not history_dict:
        return

    plot_metric_curve(
        history_dict,
        model_name,
        "accuracy",
        "val_accuracy",
        "Training and Validation Accuracy",
        "Accuracy",
        f"{model_name}_accuracy_curve.png",
    )

    plot_metric_curve(
        history_dict,
        model_name,
        "loss",
        "val_loss",
        "Training and Validation Loss",
        "Loss",
        f"{model_name}_loss_curve.png",
    )

    plot_metric_curve(
        history_dict,
        model_name,
        "roc_auc",
        "val_roc_auc",
        "Training and Validation ROC-AUC",
        "ROC-AUC",
        f"{model_name}_roc_auc_curve.png",
    )

    plot_metric_curve(
        history_dict,
        model_name,
        "pr_auc",
        "val_pr_auc",
        "Training and Validation PR-AUC",
        "PR-AUC",
        f"{model_name}_pr_auc_curve.png",
    )


# ============================================================
# 19. Evaluation metrics and plots
# ============================================================

def get_predictions(model, dataset):
    y_true_all = []
    y_prob_all = []

    start_time = time.perf_counter()

    for batch_images, batch_labels in dataset:
        batch_probs = model.predict(batch_images, verbose=0).reshape(-1)
        y_prob_all.extend(batch_probs.tolist())
        y_true_all.extend(batch_labels.numpy().reshape(-1).tolist())

    inference_time = time.perf_counter() - start_time

    y_true_all = np.array(y_true_all).astype(int)
    y_prob_all = np.array(y_prob_all).astype(float)

    return y_true_all, y_prob_all, inference_time


def calculate_metrics(y_true, y_prob, threshold, inference_time_seconds):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    eps = 1e-8

    specificity = tn / (tn + fp + eps)
    sensitivity = tp / (tp + fn + eps)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = np.nan

    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except Exception:
        auc_pr = np.nan

    try:
        ll = log_loss(y_true, np.clip(y_prob, eps, 1.0 - eps))
    except Exception:
        ll = np.nan

    n_images = len(y_true)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "f2_score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": auc_roc,
        "pr_auc": auc_pr,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": ll,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "inference_time_seconds": inference_time_seconds,
        "inference_time_per_image_ms": (inference_time_seconds / max(n_images, 1)) * 1000.0,
        "inference_images_per_second": n_images / max(inference_time_seconds, eps),
    }

    return metrics, cm, y_pred


def plot_confusion_matrix(cm, model_name, normalized=False):
    labels = ["Non-crack", "Crack"]

    if normalized:
        cm_to_plot = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        filename = f"{model_name}_normalized_confusion_matrix.png"
        title = f"{model_name} - Normalized Confusion Matrix"
        text_format = ".2f"
    else:
        cm_to_plot = cm
        filename = f"{model_name}_confusion_matrix.png"
        title = f"{model_name} - Confusion Matrix"
        text_format = "d"

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30)
    plt.yticks(tick_marks, labels)

    threshold = cm_to_plot.max() / 2.0 if cm_to_plot.max() > 0 else 0.5

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = format(cm_to_plot[i, j], text_format)
            text_color = "white" if cm_to_plot[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                value,
                horizontalalignment="center",
                color=text_color,
                fontsize=12,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        plt.title(f"{model_name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{model_name}_roc_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve for {model_name}: {e}")


def plot_precision_recall_curve(y_true, y_prob, model_name):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, label=f"PR-AUC = {ap_value:.4f}")
        plt.title(f"{model_name} - Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{model_name}_precision_recall_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not plot precision-recall curve for {model_name}: {e}")


def save_prediction_table(model_name, y_true, y_prob, y_pred):
    prediction_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_probability_crack": y_prob,
            "predicted_label": y_pred,
            "true_class": np.where(y_true == 1, "crack", "non_crack"),
            "predicted_class": np.where(y_pred == 1, "crack", "non_crack"),
            "correct_prediction": y_true == y_pred,
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_DIR / f"{model_name}_test_predictions.csv",
        index=False,
    )


# ============================================================
# 20. Train and evaluate one model
# ============================================================

def train_and_evaluate_model(model_name, model_type):
    print("\n" + "=" * 90)
    print(f"Starting model: {model_name} | Type: {model_type}")
    print("=" * 90)

    tf.keras.backend.clear_session()
    set_global_seed(CONFIG["RANDOM_SEED"])

    if model_type == "CNN":
        model = build_cnn_model(model_name)
        family = "CNN"
    elif model_type == "Transformer":
        model = build_transformer_model(model_name)
        family = TRANSFORMER_REGISTRY.get(model_name, {}).get("family", "Transformer")
    else:
        raise ValueError("model_type must be 'CNN' or 'Transformer'.")

    model = compile_model(model, CONFIG["HEAD_LR"])

    initial_param_summary = get_parameter_summary(model)
    backbone_layer_count = get_backbone_layer_count(model)

    print("\nInitial parameter summary")
    print(pd.Series(initial_param_summary))
    print("Backbone layers:", backbone_layer_count)

    # -------------------------------
    # Stage 1: train classification head
    # -------------------------------
    print(f"\nStage 1: Training classification head for {model_name}")
    head_start = time.perf_counter()

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["HEAD_EPOCHS"],
        class_weight=CLASS_WEIGHTS,
        callbacks=get_callbacks(model_name, stage="head"),
        verbose=1,
    )

    head_training_time = time.perf_counter() - head_start

    # -------------------------------
    # Stage 2: fine-tuning
    # -------------------------------
    history_fine = None
    fine_tune_training_time = 0.0

    if CONFIG["USE_FINE_TUNING"]:
        if model_type == "CNN":
            print(f"\nStage 2: Fine-tuning last {CONFIG['CNN_FINE_TUNE_LAST_N_LAYERS']} CNN layers for {model_name}")

            model = unfreeze_cnn_last_layers(
                model,
                last_n_layers=CONFIG["CNN_FINE_TUNE_LAST_N_LAYERS"],
            )
            model = compile_model(model, CONFIG["FINE_TUNE_LR"])

            fine_start = time.perf_counter()

            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=CONFIG["FINE_TUNE_EPOCHS"],
                class_weight=CLASS_WEIGHTS,
                callbacks=get_callbacks(model_name, stage="fine_tune"),
                verbose=1,
            )

            fine_tune_training_time = time.perf_counter() - fine_start

        elif model_type == "Transformer":
            if CONFIG["FINE_TUNE_TRANSFORMERS"]:
                print(f"\nStage 2: Fine-tuning transformer backbone for {model_name}")

                model = unfreeze_transformer_backbone(model)
                model = compile_model(model, CONFIG["FINE_TUNE_LR"])

                fine_start = time.perf_counter()

                history_fine = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=CONFIG["FINE_TUNE_EPOCHS"],
                    class_weight=CLASS_WEIGHTS,
                    callbacks=get_callbacks(model_name, stage="fine_tune"),
                    verbose=1,
                )

                fine_tune_training_time = time.perf_counter() - fine_start
            else:
                print(f"\nSkipping full transformer fine-tuning for {model_name}. Only the classification head was trained.")

    total_training_time = head_training_time + fine_tune_training_time

    final_param_summary = get_parameter_summary(model)

    # -------------------------------
    # Save history and curves
    # -------------------------------
    history_dict = merge_histories([history_head, history_fine])

    if history_dict:
        history_df = pd.DataFrame(history_dict)
        history_df.to_csv(
            HISTORIES_DIR / f"{model_name}_merged_history.csv",
            index=False,
        )

        plot_training_history(history_dict, model_name)

    # -------------------------------
    # Test evaluation
    # -------------------------------
    print(f"\nEvaluating {model_name} on the test set.")
    y_true, y_prob, inference_time = get_predictions(model, test_ds)

    metrics, cm, y_pred = calculate_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=CONFIG["CLASSIFICATION_THRESHOLD"],
        inference_time_seconds=inference_time,
    )

    plot_confusion_matrix(cm, model_name, normalized=False)
    plot_confusion_matrix(cm, model_name, normalized=True)
    plot_roc_curve(y_true, y_prob, model_name)
    plot_precision_recall_curve(y_true, y_prob, model_name)
    save_prediction_table(model_name, y_true, y_prob, y_pred)

    # -------------------------------
    # Save model if possible
    # -------------------------------
    final_model_dir = MODELS_DIR / model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save(final_model_dir / f"{model_name}_final.keras")
    except Exception as e:
        print(f"Full model saving failed for {model_name}. Saving weights only.")
        print("Saving error:", e)
        try:
            model.save_weights(final_model_dir / f"{model_name}_final.weights.h5")
        except Exception as e2:
            print(f"Weight saving also failed for {model_name}: {e2}")

    # -------------------------------
    # Final metric dictionary
    # -------------------------------
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "model_family": family,

        "num_train_images": len(train_paths),
        "num_val_images": len(val_paths),
        "num_test_images": len(test_paths),

        "head_training_time_seconds": head_training_time,
        "fine_tune_training_time_seconds": fine_tune_training_time,
        "total_training_time_seconds": total_training_time,

        "initial_total_params": initial_param_summary["total_params"],
        "initial_trainable_params": initial_param_summary["trainable_params"],
        "initial_non_trainable_params": initial_param_summary["non_trainable_params"],

        "final_total_params": final_param_summary["total_params"],
        "final_trainable_params": final_param_summary["trainable_params"],
        "final_non_trainable_params": final_param_summary["non_trainable_params"],

        "final_total_params_million": final_param_summary["total_params_million"],
        "final_trainable_params_million": final_param_summary["trainable_params_million"],
        "final_non_trainable_params_million": final_param_summary["non_trainable_params_million"],
        "estimated_float32_model_size_mb": final_param_summary["estimated_float32_model_size_mb"],

        "keras_model_layers": len(model.layers),
        "backbone_layers": backbone_layer_count,
    }

    result.update(metrics)

    with open(TABLES_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nCompleted model: {model_name}")
    print(pd.Series(result))

    return result


# ============================================================
# 21. Final comparison plots
# ============================================================

def plot_comparison_bar(results_df, metric_name, higher_is_better=True):
    if metric_name not in results_df.columns:
        return

    df = results_df.sort_values(metric_name, ascending=not higher_is_better)

    plt.figure(figsize=(14, 7))
    plt.bar(df["model_name"], df[metric_name])
    plt.title(f"Model Comparison - {metric_name}")
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.xticks(rotation=60, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"comparison_{metric_name}.png", dpi=300)
    plt.close()


def plot_grouped_metrics(results_df):
    metrics_to_plot = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ]

    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

    if len(available_metrics) == 0:
        return

    df = results_df[["model_name"] + available_metrics].copy()
    df = df.sort_values("f1_score", ascending=False)

    x = np.arange(len(df["model_name"]))
    width = 0.09

    plt.figure(figsize=(18, 8))

    for i, metric in enumerate(available_metrics):
        plt.bar(x + i * width, df[metric], width=width, label=metric)

    plt.title("Comprehensive Model Metrics Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(
        x + width * len(available_metrics) / 2,
        df["model_name"],
        rotation=60,
        ha="right",
    )
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_grouped_metrics.png", dpi=300)
    plt.close()


def plot_family_average(results_df):
    metric_cols = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "inference_time_per_image_ms",
        "final_total_params_million",
    ]

    available = [m for m in metric_cols if m in results_df.columns]

    summary = results_df.groupby("model_type")[available].mean().reset_index()
    summary.to_csv(TABLES_DIR / "cnn_vs_transformer_average_summary.csv", index=False)

    score_metrics = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ]

    score_metrics = [m for m in score_metrics if m in summary.columns]

    if len(score_metrics) > 0:
        x = np.arange(len(summary["model_type"]))
        width = 0.10

        plt.figure(figsize=(12, 7))

        for i, metric in enumerate(score_metrics):
            plt.bar(x + i * width, summary[metric], width=width, label=metric)

        plt.title("Average Performance: CNN Models vs Transformer Models")
        plt.xlabel("Model Type")
        plt.ylabel("Average Score")
        plt.xticks(x + width * len(score_metrics) / 2, summary["model_type"])
        plt.legend(loc="lower right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "cnn_vs_transformer_average_performance.png", dpi=300)
        plt.close()


def plot_scatter_complexity(results_df, x_col, y_col, filename, title, xlabel, ylabel):
    if x_col not in results_df.columns or y_col not in results_df.columns:
        return

    df = results_df.dropna(subset=[x_col, y_col]).copy()

    if df.empty:
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(df[x_col], df[y_col])

    for _, row in df.iterrows():
        plt.annotate(
            row["model_name"],
            (row[x_col], row[y_col]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_radar_top_models(results_df, top_k=5):
    metrics = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ]

    available = [m for m in metrics if m in results_df.columns]

    if len(available) < 3:
        return

    df = results_df.sort_values("f1_score", ascending=False).head(top_k)

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = row[available].astype(float).tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["model_name"])
        ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available)
    ax.set_ylim(0, 1)
    plt.title(f"Radar Chart of Top {top_k} Models")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar_top_models.png", dpi=300)
    plt.close()


# ============================================================
# 22. Save final comparison tables
# ============================================================

def generate_final_outputs(all_results, failures):
    if len(all_results) == 0:
        print("No successful model results to summarize.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    preferred_columns = [
        "model_name",
        "model_type",
        "model_family",

        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "f2_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "log_loss",

        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",

        "head_training_time_seconds",
        "fine_tune_training_time_seconds",
        "total_training_time_seconds",

        "inference_time_seconds",
        "inference_time_per_image_ms",
        "inference_images_per_second",

        "initial_total_params",
        "initial_trainable_params",
        "initial_non_trainable_params",

        "final_total_params",
        "final_trainable_params",
        "final_non_trainable_params",
        "final_total_params_million",
        "final_trainable_params_million",
        "final_non_trainable_params_million",
        "estimated_float32_model_size_mb",

        "keras_model_layers",
        "backbone_layers",

        "num_train_images",
        "num_val_images",
        "num_test_images",
    ]

    existing = [c for c in preferred_columns if c in results_df.columns]
    remaining = [c for c in results_df.columns if c not in existing]

    results_df = results_df[existing + remaining]

    results_df = results_df.sort_values(
        by=["f1_score", "roc_auc", "accuracy"],
        ascending=False,
    )

    csv_path = TABLES_DIR / "final_model_comparison_results.csv"
    excel_path = TABLES_DIR / "final_model_comparison_results.xlsx"

    results_df.to_csv(csv_path, index=False)

    failures_df = pd.DataFrame(failures)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="All Results")

        cnn_df = results_df[results_df["model_type"] == "CNN"]
        transformer_df = results_df[results_df["model_type"] == "Transformer"]

        if not cnn_df.empty:
            cnn_df.to_excel(writer, index=False, sheet_name="CNN Results")

        if not transformer_df.empty:
            transformer_df.to_excel(writer, index=False, sheet_name="Transformer Results")

        results_df.head(10).to_excel(writer, index=False, sheet_name="Top 10 Models")

        if not failures_df.empty:
            failures_df.to_excel(writer, index=False, sheet_name="Failed Models")

    if not failures_df.empty:
        failures_df.to_csv(TABLES_DIR / "failed_models.csv", index=False)

    # Metric comparison plots
    high_is_good = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "f2_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "inference_images_per_second",
    ]

    low_is_good = [
        "log_loss",
        "head_training_time_seconds",
        "fine_tune_training_time_seconds",
        "total_training_time_seconds",
        "inference_time_seconds",
        "inference_time_per_image_ms",
        "final_total_params_million",
        "estimated_float32_model_size_mb",
    ]

    for metric in high_is_good:
        plot_comparison_bar(results_df, metric, higher_is_better=True)

    for metric in low_is_good:
        plot_comparison_bar(results_df, metric, higher_is_better=False)

    plot_grouped_metrics(results_df)
    plot_family_average(results_df)
    plot_radar_top_models(results_df, top_k=min(5, len(results_df)))

    # Complexity-performance scatter plots
    plot_scatter_complexity(
        results_df,
        x_col="final_total_params_million",
        y_col="f1_score",
        filename="scatter_f1_vs_parameters.png",
        title="F1-Score vs Model Parameters",
        xlabel="Total Parameters in Millions",
        ylabel="F1-Score",
    )

    plot_scatter_complexity(
        results_df,
        x_col="inference_time_per_image_ms",
        y_col="f1_score",
        filename="scatter_f1_vs_inference_time.png",
        title="F1-Score vs Inference Time per Image",
        xlabel="Inference Time per Image in ms",
        ylabel="F1-Score",
    )

    plot_scatter_complexity(
        results_df,
        x_col="final_total_params_million",
        y_col="roc_auc",
        filename="scatter_roc_auc_vs_parameters.png",
        title="ROC-AUC vs Model Parameters",
        xlabel="Total Parameters in Millions",
        ylabel="ROC-AUC",
    )

    plot_scatter_complexity(
        results_df,
        x_col="estimated_float32_model_size_mb",
        y_col="f1_score",
        filename="scatter_f1_vs_model_size.png",
        title="F1-Score vs Estimated Float32 Model Size",
        xlabel="Estimated Float32 Model Size in MB",
        ylabel="F1-Score",
    )

    print("\nFinal CSV saved to:", csv_path)
    print("Final Excel saved to:", excel_path)

    print("\nTop 10 models by F1-score:")
    display_cols = [
        "model_name",
        "model_type",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "inference_time_per_image_ms",
        "final_total_params_million",
        "total_training_time_seconds",
    ]
    display_cols = [c for c in display_cols if c in results_df.columns]

    print(results_df[display_cols].head(10))

    return results_df


# ============================================================
# 23. Main experiment loop
# ============================================================

def main():
    all_results = []
    failures = []

    print("\n" + "#" * 90)
    print("STARTING CNN MODEL COMPARISON")
    print("#" * 90)

    for model_name in CNN_MODELS_TO_RUN:
        try:
            if model_name not in CNN_REGISTRY:
                raise ValueError(f"{model_name} is not available in your installed Keras version.")

            result = train_and_evaluate_model(model_name, model_type="CNN")
            all_results.append(result)

            pd.DataFrame(all_results).to_csv(
                TABLES_DIR / "partial_results.csv",
                index=False,
            )

        except Exception as e:
            print("\n" + "!" * 90)
            print(f"ERROR while running CNN model {model_name}")
            print("Error:", e)
            print("Continuing with the next model.")
            print("!" * 90)

            failures.append(
                {
                    "model_name": model_name,
                    "model_type": "CNN",
                    "error": str(e),
                }
            )

            pd.DataFrame(failures).to_csv(
                TABLES_DIR / "failed_models_partial.csv",
                index=False,
            )

    print("\n" + "#" * 90)
    print("STARTING TRANSFORMER MODEL COMPARISON")
    print("#" * 90)

    if HF_AVAILABLE:
        for model_name in TRANSFORMER_MODELS_TO_RUN:
            try:
                if model_name not in TRANSFORMER_REGISTRY:
                    raise ValueError(f"{model_name} is not available in TRANSFORMER_REGISTRY.")

                result = train_and_evaluate_model(model_name, model_type="Transformer")
                all_results.append(result)

                pd.DataFrame(all_results).to_csv(
                    TABLES_DIR / "partial_results.csv",
                    index=False,
                )

            except Exception as e:
                print("\n" + "!" * 90)
                print(f"ERROR while running Transformer model {model_name}")
                print("Error:", e)
                print("Continuing with the next model.")
                print("!" * 90)

                failures.append(
                    {
                        "model_name": model_name,
                        "model_type": "Transformer",
                        "error": str(e),
                    }
                )

                pd.DataFrame(failures).to_csv(
                    TABLES_DIR / "failed_models_partial.csv",
                    index=False,
                )
    else:
        failures.append(
            {
                "model_name": "ALL_TRANSFORMERS",
                "model_type": "Transformer",
                "error": "Hugging Face transformers package is not available.",
            }
        )

    final_df = generate_final_outputs(all_results, failures)

    print("\nExperiment completed.")
    print("All results are saved in:", OUTPUT_DIR)

    return final_df


if __name__ == "__main__":
    final_results = main()
