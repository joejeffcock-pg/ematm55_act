FROM ubuntu:20.04

RUN apt update && \
    apt install -y curl python-dev python3-dev python3-pip && \
    cd ~ && \
    curl -sSL https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install pyzmq && \
    curl -sSL https://community-static.aldebaran.com/resources/2.5.10/Python%20SDK/pynaoqi-python2.7-2.5.7.1-linux64.tar.gz -o pynaoqi-python2.7-2.5.7.1-linux64.tar.gz && \
    tar -zxvf pynaoqi-python2.7-2.5.7.1-linux64.tar.gz && \
    echo "export PYTHONPATH=${PYTHONPATH}:~/pynaoqi-python2.7-2.5.7.1-linux64/lib/python2.7/site-packages" >> ~/.bashrc && \
    python3 -m pip install mediapipe pyzmq
