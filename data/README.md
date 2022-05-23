# Hypoactivity dataset datasheet
<!-- ref: https://arxiv.org/abs/1803.09010 -->

## Motivation

Hypoactivity is symptomatic of delirium, where its detection by computer vision algorithms can create a positive impact when deployed in hospitals.

This iteration of the Hypoactivity dataset is a toy dataset targeted at testing the feasibility of hypoactivity detection by finetuning a transformer to distinguish between moving and non-moving persons.

This dataset was collected by Joe Jeffcock at the Bristol Robotics Lab, for the purpose of the EMATM0055 Dissertation 2021 module of the MSc Robotics at the University of Bristol.

## Composition

- Instances of human activity over time, grouped by hypoactive and non-hypoactive labels
- 10-45 instances per class (based on Weizmann dataset)
- Instances consist of sequences of 2D human pose keypoints, each captured at 30fps for approximately 2-5 seconds
- Instances are labelled either hypoactive or non-hypoactive
- data split 3:1:1 train:validation:test
- data is not confidential (images are of the author)
- data comprises 2D human pose keypoints; individuals are not identifiable

## Collection Process

- An RGB camera is set up 1m from the bed. Preferably an overhead camera view.
    - Camera view is fixed for this iteration of the dataset
- Camera resolution will be set to 640x480 for postprocessing using OpenPose
- Camera framerate is set to 30fps
- Camera is set to record
- Participants (the author, in this case) will be tasked with exhibiting hypoactive/non-hypoactive conditions
    - hypoactive conditions may involve breathing, movement of the wrists, or other slow movements
    - non-hypoactive conditions may involve actions such as sitting/standing up, looking around, or picking up nearby items
- Camera is set to stop once sufficient raw video data has been collected
- 2-5 second slices of hypoactive/non-hypoactive movements are cut from the recording and selected for the dataset

## Preprocessing/cleaning/labeling

- sequences are postprocessed by computing 2D human pose keypoints from each frame using OpenPose
- sequences will be indirectly labeled by storage in the appropriate directory (hypoactive/non-hypoactive)

## Uses

TODO

## Distribution

TODO

## Maintenance

TODO
