FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# install requirements
RUN apt-get update && \
	apt-get install -y cmake-qt-gui && \
	apt-get install -y libopencv-dev && \
	apt-get install -y python3-dev python3-pip && \
	apt-get install -y git && \
	apt-get install -y libgoogle-glog-dev protobuf-compiler libprotobuf-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev && \
	pip3 install --upgrade pip && \
	pip3 install --upgrade cmake && \
	pip3 install numpy opencv-python

# build openpose
# replace CUDA_ARCH_BIN and CUDA_ARCH_PTX values with the target Nvidia architecture
# my GTX 1050 Ti GPU is supported by the sm_61 architecture
RUN cd ~ && \
	git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose && \
	cd openpose/ && \
	git submodule update --init --recursive --remote && \
	mkdir build && \
	cd build && \
	cmake -DBUILD_PYTHON=ON -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=61 -DCUDA_ARCH_PTX=61 .. && \
	make -j `nproc`
