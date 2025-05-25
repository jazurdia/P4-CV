import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Parámetros configurables
# ----------------------------------------------------
ROOT_DIR           = "./mall_dataset"
MAT_PATH           = os.path.join(ROOT_DIR, "mall_gt.mat")
FRAMES_DIR         = os.path.join(ROOT_DIR, "frames")
IMG_SIZE           = (224, 224)
MODEL_SAVE_PATH    = "mall_cnn.h5"                             # Ruta donde se guardará / cargará el modelo
PREDICT_IMAGE_PATH = os.path.join(FRAMES_DIR, "seq_000970.jpg")# Imagen para predecir y visualizar
MODE               = 'compare'                                 # 'train', 'predict' o 'compare'
BATCH_SIZE         = 16
EPOCHS             = 150
N_COMPARE          = 5                                         # Cantidad de imágenes a comparar en modo 'compare'


def load_and_preprocess(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    return img

mat    = scipy.io.loadmat(MAT_PATH)
frames = mat["frame"]
n      = frames.shape[1]
paths  = [os.path.join(FRAMES_DIR, f"seq_{i+1:06d}.jpg") for i in range(n)]
counts = np.array([frames[0,i]["loc"][0,0].shape[0] for i in range(n)], dtype=np.float32)

idxs      = np.arange(n)
np.random.seed(42)
np.random.shuffle(idxs)
split_i   = int(0.8 * n)
train_idx = idxs[:split_i]
val_idx   = idxs[split_i:]
train_paths, train_counts = np.array(paths)[train_idx], counts[train_idx]
val_paths,   val_counts   = np.array(paths)[val_idx],   counts[val_idx]


if MODE == 'train':
    def make_dataset(paths, counts, batch_size, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, counts))
        if shuffle: ds = ds.shuffle(len(paths))
        ds = ds.map(lambda p,c: (load_and_preprocess(p), c),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_paths, train_counts, BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(val_paths,   val_counts,   BATCH_SIZE)

    model = models.Sequential([
        layers.Conv2D(16,3,padding="same",activation="relu",input_shape=(*IMG_SIZE,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding="same",activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding="same",activation="relu",name="last_conv"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(1,activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    model.summary()
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Modelo guardado en '{MODEL_SAVE_PATH}'")


elif MODE == 'predict':
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    img = load_and_preprocess(PREDICT_IMAGE_PATH)
    raw_pred = model.predict(tf.expand_dims(img,0))[0,0]
    pred = int(np.rint(raw_pred))
    print(f"🔢 Conteo estimado: {pred}")
    plt.figure(figsize=(6,4))
    plt.imshow(img)
    plt.title(f"Predicho: {pred:.2f}")
    plt.axis("off")
    plt.show()


elif MODE == 'compare':
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    for i in range(min(N_COMPARE, len(val_paths))):
        path = val_paths[i]
        real = val_counts[i]
        img  = load_and_preprocess(path)
        raw_pred = model.predict(tf.expand_dims(img,0))[0,0]
        pred = int(np.rint(raw_pred))
        print(f"Imagen: {os.path.basename(path)} | Real: {int(real)} | Predicho: {pred}")
        plt.figure(figsize=(6,4))
        plt.imshow(img)
        plt.title(f"Real: {int(real)}  Predicho: {pred}")
        plt.axis("off")
        plt.show()

else:
    raise ValueError("MODE debe ser 'train', 'predict' o 'compare'")

