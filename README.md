# Automated Detection of Delirium using Human-robot Interaction and Transformers for Action Recognition on Thermal Images

## Pepper (HRI) scripts

To run the interactive component of the CAM-ICU:

Run the hand state prediction server
  
  cd scripts
  python cam_icu_feature_4_server.py

Obtain Pepper's IP adress by pressing the button on its chest. Then run the CAM-ICU

  cd scripts
  python cam_icu.py --ip <Pepper's IP address>

## OpenPose 2D Skeleton extraction

Build the OpenPose container:

  cd docker/openpose/
  ./build.sh

Run the container:

  ./run.sh <input dir> <output dir>
  
Input directories must follow this structure:
-Dataset
  - Actor 1
    - label1.mp4
    - label2.mp4
    - label3.mp4
  - Actor 2
    - label1.mp4
    - label2.mp4
    - label3.mp4

## AcT Model Benchmarking

  cd src
  python kfold.py

The following changes should be made to `kfold.py`:
- the AcT path on `line 8` to a local clone of AcT: https://github.com/PIC4SeR/AcT
- the directory path to an output directory from Openpose 2D Skeleton Extraction
