FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt update && \
    apt install -y curl git python3.8 python3.8-dev python3.8-distutils && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/PIC4SeR/AcT.git && \
    cd AcT && \
    python3.8 -m pip install -r requirements.txt
