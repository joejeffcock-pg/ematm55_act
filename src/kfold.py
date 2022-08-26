import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from mpose import MPOSE

import sys
sys.path.append("/home/jeff/school/msc/fyp/ematm55_act/AcT")
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.data import random_flip, random_noise, one_hot
from utils.tools import CustomSchedule

from sklearn.metrics import balanced_accuracy_score
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

def preprocess_data(X, y, augment=False, shuffle=False):
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

def n_sized_chunks(ary, n):
    splits = [ary[i:i+n] for i in range(0, ary.shape[0]-n)]
    if not len(splits[-1]) == n:
        splits = splits[:-1]
    return splits

def stack_data(X, y, frames=30):
    X = [n_sized_chunks(array[::30], frames) for X_subset in X for array in X_subset]
    y = [n_sized_chunks(array[::30], frames) for y_subset in y for array in y_subset]
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

        X_subset = []
        y_subset = []
        for key in dataset:
            if key.startswith("X"):
                X_subset.append(dataset[key])
            elif key.startswith("y"):
                y_subset.append(dataset[key])
        
        X.append(X_subset)
        y.append(y_subset)

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    frames = 30
    kf = KFold(n_splits=6, shuffle=True, random_state=11331)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = stack_data(X[train_index], y[train_index], frames)
        X_test, y_test = stack_data(X[test_index], y[test_index], frames)
        data = (X_train, y_train, X_test, y_test)

        X_train, y_train, X_test, y_test = load_mpose('openpose', 1, verbose=False, data=data, frames=frames)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=0.1,
                                                        shuffle=True)
        
        ds_train = preprocess_data(X_train, y_train)
        ds_val = preprocess_data(X_val, y_val)
        ds_test = preprocess_data(X_test, y_test)

        # mpose hyperparams
        epochs = 350
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
            epochs=50,
            initial_epoch=0,
            validation_data=ds_val,
            # callbacks=[checkpointer],
            verbose=1
        )

        # compute metrics
        _, accuracy_test = model.evaluate(ds_test)
        X_tf, y_tf = tuple(zip(*ds_test))
        predictions = tf.nn.softmax(model.predict(tf.concat(X_tf, axis=0)), axis=-1)
        y_pred = np.argmax(predictions, axis=1)
        y_scores = np.amax(predictions, axis=1)
        balanced_accuracy = balanced_accuracy_score(tf.math.argmax(tf.concat(y_tf, axis=0), axis=1), y_pred)

        text = f"Accuracy Test: {accuracy_test} <> Balanced Accuracy: {balanced_accuracy}\n"
        print(text)

        # np.save("results/frameskip/y_test_{}.npy".format(fold), y_test)
        # np.save("results/frameskip/y_pred_{}.npy".format(fold), y_pred)