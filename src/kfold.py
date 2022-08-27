import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from mpose import MPOSE

import sys
sys.path.append("/home/jeff/school/msc/fyp/ematm55_act/AcT")
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.data import random_flip, random_noise, one_hot
from utils.tools import CustomSchedule

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, KFold

def load_mpose(dataset, split, verbose=False, data=None, frames=30):
    dataset = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=True, 
                    remove_zip=False,
                    verbose=verbose)
    dataset.T = frames

    # custom dataset - wow new feature
    if data is not None:
        dataset.load_data(data)
        dataset.load_list()
        dataset.apply_transforms()

    dataset.reduce_keypoints()
    dataset.scale_and_center()
    dataset.remove_confidence()
    dataset.flatten_features()
    
    return dataset.get_data()

def to_dataset(X, y, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda x,y : one_hot(x,y,20), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    if augment:
        ds = ds.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(X.shape[0])
    ds = ds.batch(512)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def build_act(
    # transformer params
    n_heads,
    n_layers,
    dropout,
    mlp_head_size,
    # data params
    frames=30,
    keypoints=13,
    channels=4,
    classes=20
):
    d_model = 64 * n_heads
    d_ff = d_model * 4
    transformer = TransformerEncoder(d_model, n_heads, d_ff, dropout, tf.nn.gelu, n_layers)

    inputs = tf.keras.layers.Input(shape=(frames, keypoints*channels))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = PatchClassEmbedding(d_model, frames)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(classes)(x)
    return tf.keras.models.Model(inputs, outputs)

def act_micro(frames=30):
    n_heads = 1
    n_layers = 4
    dropout = 0.3
    mlp_head_size = 256
    return build_act(n_heads, n_layers, dropout, mlp_head_size, frames)

def sliding_window(array, size, stride):
    windows = [array[i:i+size] for i in range(0, array.shape[0]-size, stride)]
    return windows

def preprocess_data(X, y, frameskip=30, size=30, stride=1):
    X = [sliding_window(array[::frameskip], size, stride) for X_actor in X for array in X_actor]
    y = [sliding_window(array[::frameskip], size, stride) for y_actor in y for array in y_actor]
    y = [np.unique(chunk) for chunks in y for chunk in chunks]
    
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


if __name__ == "__main__":
    import os
    X = []
    y = []

    # dataset
    dir_path = "/home/jeff/school/msc/fyp/ematm55_act/docker/openpose/outputs_thermal"
    for file_name in sorted(os.listdir(dir_path)):
        if not file_name.endswith(".npz"):
            continue

        file_path = os.path.join(dir_path, file_name)
        dataset = np.load(file_path)

        X_actor = []
        y_actor = []
        for key in dataset:
            if key.startswith("X"):
                X_actor.append(dataset[key])
            elif key.startswith("y"):
                y_actor.append(dataset[key])
        
        X.append(X_actor)
        y.append(y_actor)

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    frames = 30
    kf = KFold(n_splits=6, shuffle=True, random_state=11331)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_train, y_train = preprocess_data(X_train, y_train)

        data = (X_train, y_train, X_train, y_train)
        X_train, y_train, _, _ = load_mpose('openpose', 1, verbose=False, data=data, frames=frames)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
        
        ds_train = to_dataset(X_train, y_train)
        ds_val = to_dataset(X_val, y_val)

        # mpose hyperparams
        epochs = 50
        lr = CustomSchedule(64, 
            warmup_steps=len(ds_train)*epochs*0.3,
            decay_step=len(ds_train)*epochs*0.8)
        weight_decay=1e-4

        # create model
        model = act_micro(frames=frames)
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # training
        history = model.fit(
            ds_train,
            epochs=epochs,
            initial_epoch=0,
            validation_data=ds_val,
            # callbacks=[checkpointer],
            verbose=1
        )

        # compute metrics
        test_results = {}
        weighted_accuracy = 0
        weighted_balanced_accuracy = 0

        for index in test_index:
            print("Test actor: {}".format(index))

            X_test, y_test = X[index], y[index]
            X_test, y_test = preprocess_data([X_test], [y_test])
            
            data = (X_test, y_test, X_test, y_test)
            X_test, y_test, _, _ = load_mpose('openpose', 1, verbose=False, data=data, frames=frames)

            ds_test = to_dataset(X_test, y_test)
            _, accuracy_test = model.evaluate(ds_test)
            X_tf, y_tf = tuple(zip(*ds_test))
            predictions = tf.nn.softmax(model.predict(tf.concat(X_tf, axis=0)), axis=-1)
            y_pred = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            text = f"Accuracy Test: {accuracy} <> Balanced Accuracy: {balanced_accuracy}\n"
            print(text)

            test_results["actor_{}_X_test".format(index)] = X_test
            test_results["actor_{}_y_test".format(index)] = y_test
            test_results["actor_{}_y_pred".format(index)] = y_pred
        
        print("Test fold: {} actors: {}".format(fold, test_index))
        y_test = np.concatenate([test_results["actor_{}_y_test".format(index)] for index in test_index])
        y_pred = np.concatenate([test_results["actor_{}_y_pred".format(index)] for index in test_index])
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        text = f"Accuracy Test: {accuracy} <> Balanced Accuracy: {balanced_accuracy}\n"
        print(text)

        test_results["indices"] = test_index
        np.savez_compressed("results/actors/fold_{}.npz".format(fold), **test_results)