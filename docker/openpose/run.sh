if [ -z "$1" ]
    then
        echo "usage: ./run.sh <input directory>"
else
    nvidia-docker run \
        -it \
        --rm \
        --gpus=all \
        --env=PYTHONPATH=/root/openpose/build/python/openpose \
        --volume $1:/root/inputs \
        --volume $PWD/outputs:/root/outputs \
        --volume $PWD/extract_poses.py:/root/find_videos.py \
        --env=DISPLAY \
        --env=QT_X11_NO_MITSHM=1 \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
        joejeffcockpg/ematm55_openpose:latest \
        python3 /root/find_videos.py
fi
