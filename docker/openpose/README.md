# OpenPose Docker container

## Build OpenPose for Nvidia

	cd ~
	git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
	cd openpose/
	git submodule update --init --recursive --remote
	mkdir build
	cd build
	cmake -DBUILD_PYTHON=ON -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=61 -DCUDA_ARCH_PTX=61 ..
	make -j `nproc`

Replace `-DCUDA_ARCH_BIN=61 -DCUDA_ARCH_PTX=61` with the appropriate architecture code: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list