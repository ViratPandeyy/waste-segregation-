import argparse, json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).parent
DATA = ROOT / "dataset"
OUT = ROOT / "models"
CLASSES = ["glass", "organic", "paper", "plastic"]

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12)
args = parser.parse_args()

for name in CLASSES:
    folder = DATA / name
    if not folder.exists() or not any(folder.iterdir()):
        raise SystemExit(f"Add training images to {folder} before training.")

train = tf.keras.utils.image_dataset_from_directory(DATA, validation_split=.2, subset="training", seed=123, image_size=(224,224), batch_size=32, class_names=CLASSES)
val = tf.keras.utils.image_dataset_from_directory(DATA, validation_split=.2, subset="validation", seed=123, image_size=(224,224), batch_size=32, class_names=CLASSES)
auto = tf.data.AUTOTUNE
train = train.prefetch(auto); val = val.prefetch(auto)
base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False
inputs = layers.Input(shape=(224,224,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(.2)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=args.epochs)
OUT.mkdir(exist_ok=True)
model.save(OUT / "waste_mobilenetv2.keras")
(OUT / "class_names.json").write_text(json.dumps(CLASSES))
print("Saved model in models/")
