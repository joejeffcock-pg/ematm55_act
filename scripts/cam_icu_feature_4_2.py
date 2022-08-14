import time
import qi
import numpy as np
import zmq
import time
from collections import Counter
import argparse

HANDEDNESS_MAP = {"Left": 0, "Right": 1}

class DisorganisedThinkingFingers:
    def __init__(self, session, socket, auto_start=True):
        self.camProxy = session.service("ALVideoDevice")
        self.tts = session.service("ALTextToSpeech")
        self.ap = session.service("ALAudioPlayer")
        self.socket = socket
        self.tablet = session.service("ALTabletService")
        if auto_start:
            self.start()
    
    def start(self):
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        fps = 5
        self.videoClient = self.camProxy.subscribeCamera("python_client",0, resolution, colorSpace, fps)
    
    def stop(self):
        self.camProxy.unsubscribe(self.videoClient)
    
    def say_introduction(self):
        self.tts.say("I need you to make gestures with your hands")
        self.tts.say("At one arms length away, hold up")
        # self.tablet.showImage("https://joejeffcock-pg.github.io/ematm55_act/images/two_fingers.png")
        self.tablet.showImage("http://198.18.0.1/apps/images_folder_for_joe-5e9343/two_fingers.png")
        self.tts.say("this")
        self.tts.say("many fingers where I can see them")
    
    def detect_gesture(self):
        counts = []
        labels = []
        start_time = time.time()
        while len(counts) < 10 and time.time() - start_time < 10.0:
            # image[6] contains the image data passed as an array of ASCII chars.
            naoImage = self.camProxy.getImageRemote(self.videoClient)
            array = naoImage[6]

            self.socket.send(str(array))
            if not self.socket.poll(250):
                continue
            
            message = self.socket.recv()
            if message == "":
                counts = []
                labels = []
            else:
                fingers, label = message.split()
                counts.append(np.sum([int(v) for v in fingers]))
                labels.append(label)
                self.ap.playSine(500, 20, 0, 0.1)
                time.sleep(0.1)
        
        if len(labels) == 10:
            count = Counter(counts).most_common(1)[0][0]
            label = Counter(labels).most_common(1)[0][0]
            return count, HANDEDNESS_MAP[label]
        else:
            return -1, 0
    
    def interview(self):
        self.say_introduction()
        results = []
        results.append(self.detect_gesture())
        self.tablet.hideImage()
        time.sleep(0.5)
        if not results[0][0] == -1:
            self.tts.say("Okay. Now do the same thing with the other hand")
            time.sleep(0.5)
            results.append(self.detect_gesture())

        print(results)
        results = np.array(results)
        result = (results[:,0] == 2).all() and (np.sum(results[:,1]) == 1)
        errors = int(not result)
        return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", dest="ip", action="store", type=str, default="pepper.local")
    parser.add_argument("--port", dest="port", action="store", type=int, default=9559)
    args = parser.parse_args()

    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:{}".format(args.ip, args.port)])
    app.start()
    session = app.session

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5560")

    try:
        feature_4_2 = DisorganisedThinkingFingers(session, socket)
        errors = feature_4_2.interview()
        print(errors)
    except:
        pass
    feature_4_2.stop()
