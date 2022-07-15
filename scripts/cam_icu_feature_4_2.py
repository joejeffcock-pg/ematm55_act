# -*- encoding: UTF-8 -*-
# Get an image from NAO. Display it and save it using PIL.

import sys
import time

# Python Image Library
from PIL import Image
import qi
import cv2
import numpy as np
import zmq
import time


def showNaoImage(IP, PORT):
    """
    First get an image from Nao, then show it on the screen with PIL.
    """
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:{}".format(IP, PORT)])
    app.start()
    session = app.session
    camProxy = session.service("ALVideoDevice")
    tts = session.service("ALTextToSpeech")
    ap = session.service("ALAudioPlayer")

    resolution = 2    # VGA
    colorSpace = 11   # RGB
    fps = 5
    # videoClient = camProxy.subscribe("python_client", resolution, colorSpace, fps) # Pepper
    videoClient = camProxy.subscribeCamera("python_client",0, resolution, colorSpace, fps) # NAO 6

    #  Prepare our context and sockets
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5560")

    tts.say("Please hold up 2 fingers")
    results = []
    while len(results) < 10:
        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        t0 = time.time()
        naoImage = camProxy.getImageRemote(videoClient)
        t1 = time.time()

        # Time the image transfer.
        print("acquisition delay ", t1 - t0)

        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]

        socket.send(str(array))
        if not socket.poll(200):
            break
        message = socket.recv()

        if message == "":
            results = []
        else:
            results.append(np.sum([int(v) for v in message]))
            ap.playSine(500, 20, 0, 0.1)
            time.sleep(0.1)
        
    camProxy.unsubscribe(videoClient)
    print(results)

    result = int(round(np.mean(results)))
    if result == 2:
        tts.say("good!")
    else:
        tts.say("oh no!")
    tts.say("{} fingers are up".format(result))



if __name__ == '__main__':
    IP = "169.254.213.34"  # Replace here with your NaoQi's IP address.
    PORT = 9559

    # Read IP address from first argument if any.
    if len(sys.argv) > 1:
        IP = sys.argv[1]

    naoImage = showNaoImage(IP, PORT)