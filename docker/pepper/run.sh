docker run \
    -it \
    --network host \
    --name ematm55_pepper \
    --env=DISPLAY \
    --env=QT_X11_NO_MITSHM=1 \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
    joejeffcockpg/ematm55_pepper:latest
