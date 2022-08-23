if [ "$#" -eq 0 ]
    then
        echo "usage: ./run.sh <input directory> [output directory]"
else
    # set output directory
    output_dir=$PWD/outputs
    if [ "$#" -eq 2 ]
        then
            output_dir="$2"
    fi

    # run OpenPose container mounting input/output/scripts
    # before executing pose extraction
    nvidia-docker run \
        -it \
        --rm \
        --gpus=all \
        --env=PYTHONPATH=/root/openpose/build/python/openpose \
        --volume $1:/root/inputs \
        --volume $output_dir:/root/outputs \
        --volume $PWD/extract_poses.py:/root/extract_poses.py \
        --env=DISPLAY \
        --env=QT_X11_NO_MITSHM=1 \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
        joejeffcockpg/ematm55_openpose:latest \
        python3 /root/extract_poses.py
fi
