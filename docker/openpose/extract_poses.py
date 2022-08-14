import os
import numpy as np
import cv2
import pyopenpose as op

INPUTS_PATH = "/root/inputs"

def compute_keypoints(frame, op_wrapper):
    datum = op.Datum()
    datum.cvInputData = frame
    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    key = cv2.waitKey(15)
    return datum

dirs = {}

print("searching the following directories for .mp4 files:")
for class_name in os.listdir(INPUTS_PATH):
    dir_path = os.path.join(INPUTS_PATH, class_name)

    if os.path.isdir(dir_path):
        print(dir_path)

        for file_name in os.listdir(dir_path):
            if file_name.endswith(".mp4"):
                dirs[class_name] = dir_path

print("\nthe following classes were found:")
print(dirs)

# openpose wrapper
params = {"model_folder": "/root/openpose/models"}
op_wrapper = op.WrapperPython()
op_wrapper.configure(params)
op_wrapper.start()

kwargs = {}
count = 0

for class_index, class_name in enumerate(dirs):
    dir_path = dirs[class_name]

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(dir_path, file_name)
            cap = cv2.VideoCapture(file_path)

            X = []
            y = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                datum = compute_keypoints(frame, op_wrapper)

                if datum.poseKeypoints is not None:
                    X.append(datum.poseKeypoints[0])
                    y.append(class_index)
                
                if len(X) % 300 == 0:
                    print("video: {} video_time: {:.2f}s frame_no: {} X_shape: {}".format(count, len(X)/30.0, len(X), np.stack(X, axis=0).shape))
            
            cap.release()

            kwargs["X_{}".format(count)] = np.array(X)
            kwargs["y_{}".format(count)] = np.array(y, dtype=np.int32)
            count += 1

np.savez_compressed("/root/outputs/dataset.npz", **kwargs)

dataset = np.load("/root/outputs/dataset.npz")
for key in dataset:
    print(key, dataset[key].shape)