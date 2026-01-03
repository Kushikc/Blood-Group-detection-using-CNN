import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import os

BATCH_SIZE = 32
IMG_SIZE = (64, 64)

dataset_path = "dataset_blood_group"

raw_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=True
)

class_names = raw_dataset.class_names

imgs, labels = [], []
counts = Counter()
for img, lbl in raw_dataset:
    imgs.append(img.numpy())
    labels.append(lbl.numpy())
    counts[int(lbl.numpy())] += 1

imgs = np.array(imgs)
labels = np.array(labels)

max_count = max(counts.values())
balanced_indices = []

for cls_id, cls_count in counts.items():
    indices = np.where(labels == cls_id)[0]
    reps = max_count // cls_count + (max_count % cls_count > 0)
    balanced_indices.extend(np.tile(indices, reps)[:max_count].tolist())

balanced_indices = np.random.permutation(balanced_indices)
imgs_bal = imgs[balanced_indices]
labels_bal = labels[balanced_indices]

total_len = imgs_bal.shape[0]
train_len = int(0.7 * total_len)
val_len = int(0.2 * total_len)

train_imgs, train_labels = imgs_bal[:train_len], labels_bal[:train_len]
val_imgs, val_labels = imgs_bal[train_len:train_len + val_len], labels_bal[train_len:train_len + val_len]
test_imgs, test_labels = imgs_bal[train_len + val_len:], labels_bal[train_len + val_len:]

def create_ds(images, labels, augment=False, shuffle=False, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(len(images))
    # Normalize pixel values to [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    if repeat:
        ds = ds.repeat()
    # No augmentation in this initial setup
    return ds

train_dataset = create_ds(train_imgs, train_labels, shuffle=True, repeat=True)
val_dataset = create_ds(val_imgs, val_labels)
test_dataset = create_ds(test_imgs, test_labels)

steps_per_epoch = train_len // BATCH_SIZE
validation_steps = val_len // BATCH_SIZE
test_steps = test_imgs.shape[0] // BATCH_SIZE

def original_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced LR for stable learning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = original_cnn_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('best_original_cnn.h5', save_best_only=True, monitor='val_loss', verbose=1)
]

history = model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
)

test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

y_true, y_pred = [], []
for i, (imgs_batch, labels_batch) in enumerate(test_dataset):
    if i >= test_steps:
        break
    preds = model.predict(imgs_batch)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(preds.argmax(axis=1))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

num_images = 10
plt.figure(figsize=(15, 6))

for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    img = test_imgs[i] / 255.0  # Normalize pixel values to [0, 1]
    true_label = class_names[test_labels[i]]
    pred_label = class_names[y_pred[i]]
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

