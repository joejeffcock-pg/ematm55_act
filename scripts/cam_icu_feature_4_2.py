import time
import qi
import numpy as np
import zmq
import time
from collections import Counter

class DisorganisedThinkingFingers:
    def __init__(self, session, socket):
        self.camProxy = session.service("ALVideoDevice")
        self.tts = session.service("ALTextToSpeech")
        self.ap = session.service("ALAudioPlayer")
        self.socket = socket
    
    def say_introduction(self):
        self.tts.say("I need you to make gestures with your hands")
        self.tts.say("At one arms length away, please hold up 2 fingers where I can see them")
    
    def interview(self):
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        fps = 5
        videoClient = self.camProxy.subscribeCamera("python_client",0, resolution, colorSpace, fps)

        self.say_introduction()
        results = []
        while len(results) < 10:
            # image[6] contains the image data passed as an array of ASCII chars.
            naoImage = self.camProxy.getImageRemote(videoClient)
            array = naoImage[6]

            self.socket.send(str(array))
            if not self.socket.poll(250):
                continue
            
            message = self.socket.recv()
            if message == "":
                results = []
            else:
                results.append(np.sum([int(v) for v in message]))
                self.ap.playSine(500, 20, 0, 0.1)
                time.sleep(0.1)
        
        self.camProxy.unsubscribe(videoClient)

        counter = Counter(results)
        result = counter.most_common(1)[0][0]
        return result == 2

if __name__ == "__main__":
    ip = "169.254.45.131"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:9559".format(ip, port)])
    app.start()
    session = app.session

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5560")

    feature_4_2 = DisorganisedThinkingFingers(session, socket)
    feature_4_2.interview()
