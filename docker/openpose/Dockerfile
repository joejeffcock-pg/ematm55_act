FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# OpenPose pre-requisites
RUN apt-get update && \
	apt-get install -y cmake-qt-gui && \
	apt-get install -y libopencv-dev && \
	apt-get install -y python3-dev python3-pip && \
	apt-get install -y git && \
	apt-get install -y libgoogle-glog-dev protobuf-compiler libprotobuf-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev && \
	pip3 install --upgrade pip && \
	pip3 install --upgrade cmake && \
	pip3 install numpy opencv-python
