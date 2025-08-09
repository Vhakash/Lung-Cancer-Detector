import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import numpy as np

# Load dorsar/lung-cancer dataset
ds = load_dataset("dorsar/lung-cancer")

IMG_SIZE = 350

def preprocess_example(example):
    image = example['image']
    label = example['label']

    # Convert PIL image to numpy array
    image = np.array(image)

    # Handle grayscale images (2D) by adding channel dimension
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    # Convert 4-channel images (RGBA) to 3 channels by dropping alpha channel
    if image.shape[-1] == 4:
        image = image[..., :3]

    # For grayscale single-channel images, repeat to create 3 channels
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # Convert to tf.Tensor
    image = tf.convert_to_tensor(image)

    # Resize image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize pixels to [0,1]
    image = image / 255.0

    return image, label

def prepare_tf_dataset(hf_dataset, batch_size=32, shuffle=False):
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: (preprocess_example(x) for x in hf_dataset),
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
    )
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return tf_dataset

train_dataset = prepare_tf_dataset(ds['train'], batch_size=32, shuffle=True)
valid_dataset = prepare_tf_dataset(ds['validation'], batch_size=32)
test_dataset = prepare_tf_dataset(ds['test'], batch_size=32)

num_classes = len(ds['train'].features['label'].names)

base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 20

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset
)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

model.save("lung_cancer_model.h5")
