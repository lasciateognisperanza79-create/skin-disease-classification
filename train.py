import os, random, gc, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from google.colab import drive
from datetime import datetime

# Монтирование Drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/pediatric_3class_resnet_best_full'
os.makedirs(SAVE_DIR, exist_ok=True)

# Параметры
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
TARGET_PER_CLS  = 800
EPOCHS_STAGE1   = 12
EPOCHS_STAGE2   = 20
NUM_CLASSES     = 3
CLASS_NAMES     = ['allergy', 'chickenpox', 'normal']

# Загрузка данных 
print("Загрузка датасетов...")
!kaggle datasets download -d dipuiucse/monkeypoxskinimagedataset -p /content/monkeypox_data --unzip -q

RAW_DIR = '/content/datasets_raw'          
EXTRA_DIR = '/content/monkeypox_data'     

def find_images(base_dir, target_class):
    images = []
    sick_kw = ['atopic','eczema','dermatitis','chickenpox','varicella','chicken','urticaria','hives','orticaria','mpox']
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root).lower()
        if target_class == 'normal':
            if any(kw in folder for kw in sick_kw):
                continue
        if target_class == 'allergy':
            if any(kw in folder for kw in ['atopic','eczema','dermatitis','urticaria','hives','orticaria']):
                for f in files:
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        images.append(os.path.join(root, f))
        elif target_class == 'chickenpox':
            if any(kw in folder for kw in ['chickenpox','varicella','chicken']):
                for f in files:
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        images.append(os.path.join(root, f))
        elif target_class == 'normal':
            if any(kw in folder for kw in ['nv','nevus','nevi','melanocytic','healthy','normal','benign']):
                for f in files:
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                        images.append(os.path.join(root, f))
    return list(set(images))

def find_images_extra(base_dir, target_class):
    images = []
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root).lower()
        if target_class == 'normal' and ('normal' in folder or 'healthy' in folder):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                    images.append(os.path.join(root, f))
    return images

raw_data = {}
for cls in CLASS_NAMES:
    raw_data[cls] = find_images(RAW_DIR, cls)
extra_normal = find_images_extra(EXTRA_DIR, 'normal')
raw_data['normal'].extend(extra_normal)
raw_data['normal'] = list(set(raw_data['normal']))

print("Raw image counts:")
for cls in CLASS_NAMES:
    print(f"  {cls}: {len(raw_data[cls])}")

# Балансировка
balanced_data = {}
for cls in CLASS_NAMES:
    imgs = raw_data[cls]
    random.shuffle(imgs)
    if len(imgs) >= TARGET_PER_CLS:
        balanced_data[cls] = imgs[:TARGET_PER_CLS]
    else:
        repeats = (TARGET_PER_CLS // len(imgs)) + 1
        balanced_data[cls] = (imgs * repeats)[:TARGET_PER_CLS]

paths, labels = [], []
for idx, cls in enumerate(CLASS_NAMES):
    for p in balanced_data[cls]:
        paths.append(p)
        labels.append(idx)
paths = np.array(paths); labels = np.array(labels)

X_train, X_tmp, y_train, y_tmp = train_test_split(
    paths, labels, test_size=0.2, random_state=SEED, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
del raw_data, balanced_data, paths, labels, X_tmp, y_tmp
gc.collect()

# Датасеты
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_online(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    crop_ratio = tf.random.uniform([], 0.85, 1.0)
    img = tf.image.central_crop(img, crop_ratio)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.08)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_online, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(3000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val,   y_val,   training=False)
test_ds  = make_dataset(X_test,  y_test,  training=False)

# Веса классов
cw_arr  = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

def smooth_sparse_ce(y_true, y_pred, smoothing=0.1):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    y_true_smooth = y_true_oh * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

loss_fn = smooth_sparse_ce
mixed_precision.set_global_policy('mixed_float16')

# Модель
base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.002))(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.002))(x)
x = Dropout(0.5)(x)
out = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
model.summary()

# Callbacks
callbacks1 = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('/tmp/best1.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]

print("\n=== Stage 1 ===")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1,
               class_weight=cw_dict, callbacks=callbacks1, verbose=1)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss=loss_fn, metrics=['accuracy'])

callbacks2 = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('/tmp/best2.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
]

print("\n=== Stage 2 ===")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2,
               class_weight=cw_dict, callbacks=callbacks2, verbose=1)

# Оценка
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

y_true, y_probs = [], []
for imgs, lbls in test_ds:
    probs = model.predict(imgs, verbose=0)
    y_true.extend(lbls.numpy())
    y_probs.extend(probs)
y_true = np.array(y_true); y_probs = np.array(y_probs); y_pred = np.argmax(y_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

ohe = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
roc = roc_auc_score(ohe, y_probs, multi_class='ovr', average='macro')
print(f"ROC-AUC (macro): {roc:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — 3 classes (87.08%)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# Графики обучения
def merge_hist(h1, h2):
    return {k: h1.history[k] + h2.history[k] for k in h1.history}
full = merge_hist(h1, h2)
ep = range(1, len(full['accuracy']) + 1)
split = len(h1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, (tr, val), title in zip(
    axes,
    [('accuracy', 'val_accuracy'), ('loss', 'val_loss')],
    ['Accuracy', 'Loss']):
    ax.plot(ep, full[tr], label='Train')
    ax.plot(ep, full[val], label='Val')
    ax.axvline(split, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()

# Grad-CAM
GRADCAM_LAYER = 'conv5_block3_out'
def gradcam_heatmap(model, img_tensor, layer_name):
    g_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = g_model(img_tensor, training=False)
        pred_idx = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_idx]
    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), pred_idx

def overlay_cam(img_np, heatmap):
    h = cv2.resize(heatmap.astype(np.float32), (img_np.shape[1], img_np.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.6, colored, 0.4, 0)

fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(13, NUM_CLASSES*3.5))
shown = {i: False for i in range(NUM_CLASSES)}
for path, true_lbl in zip(X_test, y_test):
    if shown[true_lbl]:
        continue
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img_array = tf.expand_dims(img, 0)
    img_np = (img.numpy() * 255).astype(np.uint8)

    heatmap, pred_idx = gradcam_heatmap(model, img_array, GRADCAM_LAYER)
    overlay = overlay_cam(img_np, heatmap)
    correct = '✓' if pred_idx == true_lbl else '✗'
    row = true_lbl

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title(f"True: {CLASS_NAMES[true_lbl]}", fontsize=11)
    axes[row, 0].axis('off')
    axes[row, 1].imshow(heatmap, cmap='jet')
    axes[row, 1].set_title("Grad-CAM", fontsize=11)
    axes[row, 1].axis('off')
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Pred: {CLASS_NAMES[pred_idx]} {correct}", fontsize=11)
    axes[row, 2].axis('off')

    shown[true_lbl] = True
    if all(shown.values()):
        break

plt.suptitle("Grad-CAM — по одному примеру на класс (87.08%)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'gradcam_examples.png'), dpi=150)
plt.show()

# Сохранение модели и информации
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_filename = f"resnet_best_{acc*100:.2f}_{timestamp}.keras"
model_path = os.path.join(SAVE_DIR, model_filename)
model.save(model_path)
print(f"Модель сохранена как: {model_filename}")

with open(os.path.join(SAVE_DIR, 'model_info.txt'), 'w') as f:
    f.write(
        f"Architecture: ResNet50 + Monkeypox normal data\n"
        f"Dropout: 0.6 / 0.5, Stage2 LR: 1e-5\n"
        f"Classes: {CLASS_NAMES}\n"
        f"Target per class: {TARGET_PER_CLS}\n"
        f"Test Accuracy: {acc*100:.2f}%\n"
        f"ROC-AUC macro: {roc:.4f}\n"
        f"Model file: {model_filename}\n"
    )

print(f"\nAll files saved to: {SAVE_DIR}")

from google.colab import files
files.download(model_path)
print("Скачивание модели началось.")
