import os
import numpy as np
import cv2
import pyopenpose as op
from collections import defaultdict

INPUTS_PATH = "/root/inputs"

def compute_keypoints(frame, op_wrapper):
    datum = op.Datum()
    datum.cvInputData = frame
    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

    cv2.imshow("OpenPose", datum.cvOutputData)
    cv2.waitKey(15)
    return datum

sources = defaultdict(list)
labels_text = set()

print("searching the following directories for .mp4 files:")
for dir_name in os.listdir(INPUTS_PATH):
    dir_path = os.path.join(INPUTS_PATH, dir_name)

    if os.path.isdir(dir_path):
        print(dir_path)

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith(".mp4"):
                label_text = file_name[:-4]
                labels_text.add(label_text)
                sources[dir_name].append((file_path, label_text))

labels_text = dict(zip(labels_text, range(len(labels_text))))

print("\nthe following classes were found:")
print(labels_text)
print("\nThe following data sources were found:")
for name in sources:
    for file_path, label_text in sources[name]:
        print("name: {} file_path: {:8} label_text: {}".format(name, label_text, file_path))
print()

# openpose wrapper
params = {"model_folder": "/root/openpose/models"}
op_wrapper = op.WrapperPython()
op_wrapper.configure(params)
op_wrapper.start()

for name in sources:
    kwargs = {}

    for file_path, label_text in sources[name]:
        X = []
        y = []

        cap = cv2.VideoCapture(file_path)
        label = labels_text[label_text]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # most of my data is portrait, so I'm rotating the image here
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            datum = compute_keypoints(frame, op_wrapper)
            if datum.poseKeypoints is not None:
                best_score = 0.0
                best_pose = datum.poseKeypoints[0]

                # keep the best pose
                for poseKeypoints in datum.poseKeypoints:
                    avg_score = np.sum(poseKeypoints[:,2])
                    if avg_score >= best_score:
                        best_score = avg_score
                        best_pose = poseKeypoints

                X.append(best_pose)
            else:
                X.append(np.zeros((25,3)))
            
            y.append(label)
            
            if len(X) % 300 == 0:
                print("video: {} video_time: {:.2f}s frame_no: {} X_shape: {}".format(file_path, len(X)/30.0, len(X), np.stack(X, axis=0).shape))
        
        cap.release()

        kwargs["X{}".format(label)] = np.array(X)
        kwargs["y{}".format(label)] = np.array(y, dtype=np.int32)

    np.savez_compressed("/root/outputs/{}.npz".format(name), **kwargs)

for name in sources:
    npz_path = "/root/outputs/{}.npz".format(name)
    print(npz_path)
    dataset = np.load(npz_path)
    for key in dataset:
        print(key, dataset[key].shape)
    print()