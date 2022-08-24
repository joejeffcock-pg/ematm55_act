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

#TODO this function can be way way smaller
def preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test):
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.map(lambda x,y : one_hot(x,y,20), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(X_train.shape[0])
    ds_train = ds_train.batch(512)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.map(lambda x,y : one_hot(x,y,20), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(512)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.map(lambda x,y : one_hot(x,y,20), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(512)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_val, ds_test

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
    splits = [ary[i:i+n] for i in range(0, ary.shape[0], n)]
    if not len(splits[-1]) == n:
        splits = splits[:-1]
    return splits

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
        for key in dataset:
            if key.startswith("X"):
                X.append(dataset[key])
            elif key.startswith("y"):
                y.append(dataset[key])

    frames = 30
    X = [n_sized_chunks(array, frames) for array in X]
    y = [np.unique(chunk) for array in y for chunk in (n_sized_chunks(array, frames))]
    X = np.vstack(X)
    y = np.hstack(y)

    kf = KFold(n_splits=6)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        data = (X_train, y_train, X_test, y_test)

        X_train, y_train, X_test, y_test = load_mpose('openpose', 1, verbose=False, data=data, frames=frames)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        shuffle=True)
        
        ds_train, ds_val, ds_test = preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test)

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
            epochs=epochs,
            initial_epoch=0,
            validation_data=ds_val,
            # callbacks=[checkpointer],
            verbose=0
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

        # # qualitative results
        # import cv2
        # import time
        # video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (240, 320))
        # X_test = np.load("/home/jeff/school/msc/fyp/ematm55_act/docker/openpose/outputs_thermal/X_test.npy")
        # y_test = np.load("/home/jeff/school/msc/fyp/ematm55_act/docker/openpose/outputs_thermal/y_test.npy")
        # labels = {0:"alert", 1:"sedated", 2:"agitated"}

        # pairs = [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16, 16, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]
        # for i, data in enumerate(X_test):
        #     for j, keypoints in enumerate(data):
        #         frame = np.zeros((320,240,3), dtype=np.uint8)
        #         for k in range(0, len(pairs)-1, 2):
        #             p1 = pairs[k]
        #             p2 = pairs[k+1]
        #             if keypoints[p1][2] > 0.0 and keypoints[p2][2] > 0.0:
        #                 start = [int(v) for v in keypoints[p1][0:2]]
        #                 end = [int(v) for v in keypoints[p2][0:2]]
        #                 cv2.line(frame,start,end,(255,255,255),2)
                
        #         cv2.putText(frame, "label: {} prediction: {} ({:.3f})".format(labels[y_test[i]], labels[y_pred[i]], y_scores[i]), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        #         cv2.imshow("frame", frame)
        #         cv2.waitKey(1)
        #         # time.sleep(0.03)
        #         video_writer.write(frame)
        # video_writer.release()

        # np.save("y_test_{}.npy".format(fold), y_test)
        # np.save("y_pred_{}.npy".format(fold), y_pred)