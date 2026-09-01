"""Train a per-user model on stroke features and export it as quantised TFLite.

Produces app/src/main/assets/touch_model.tflite plus the size and latency
figures the deployment section of the paper needs.

The model is deliberately small. It has to run on every stroke, inside a
foreground service, without draining the battery -- so capacity is traded for
a footprint measured in kilobytes.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", default="strokes_touchalytics.csv",
                    help="output of the notebook's Phase 1")
    ap.add_argument("--user", type=int, default=None,
                    help="which user to enrol; defaults to the one with most strokes")
    ap.add_argument("--out", default="../app/src/main/assets/touch_model.tflite")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    import touchauth as ta
    strokes = pd.read_csv(args.strokes)
    data = ta.normalise_devices(strokes, "rank")
    cols = ta.feature_columns(data)

    user = args.user if args.user is not None else data.user_id.value_counts().index[0]
    print(f"enrolling user {user} over {len(cols)} features")

    # Enrolment and test never share a session, matching the offline protocol.
    train = data[data.session.isin([0, 1, 2])]
    test = data[data.session.isin([3, 4])]

    X_train = train[cols].to_numpy(np.float32)
    y_train = (train.user_id == user).to_numpy(np.float32)
    X_test = test[cols].to_numpy(np.float32)
    y_test = (test.user_id == user).to_numpy(np.float32)

    # One person against forty is heavily imbalanced. Without this the network
    # learns to answer "impostor" every time and scores well doing it.
    pos = max(y_train.sum(), 1)
    class_weight = {0: 1.0, 1: float((len(y_train) - pos) / pos)}

    model = build(len(cols))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=64,
              class_weight=class_weight, verbose=0)
    auc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"held-out AUC {auc:.3f}")

    def representative():
        for row in X_train[np.random.choice(len(X_train), min(500, len(X_train)), False)]:
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {args.out}  ({len(tflite_model) / 1024:.1f} KB)")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    sample = X_test[:1].astype(inp["dtype"]) if inp["dtype"] != np.float32 else X_test[:1]

    for _ in range(20):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
    start = time.perf_counter()
    for _ in range(500):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
    print(f"desktop inference {(time.perf_counter() - start) / 500 * 1000:.3f} ms per stroke")
    print("measure the real figure on the handset with Classifier.benchmark()")


if __name__ == "__main__":
    main()
