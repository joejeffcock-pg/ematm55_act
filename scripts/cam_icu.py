import qi
import zmq
import time
from cam_icu_feature_2 import *
from cam_icu_feature_4_1 import *
from cam_icu_feature_4_2 import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", dest="ip", action="store", type=str, default="pepper.local")
    parser.add_argument("--port", dest="port", action="store", type=int, default=9559)
    args = parser.parse_args()

    ip = "169.254.179.1"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:{}".format(args.ip, args.port)])
    app.start()
    session = app.session

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5560")

    feature_2 = Inattention(session)
    feature_4_1 = DisorganisedThinkingYesNo(session, auto_start=False)
    feature_4_2 = DisorganisedThinkingFingers(session, socket)

    posture = session.service("ALRobotPosture")
    posture.goToPosture("Stand", 1.0)

    tts = session.service("ALTextToSpeech")
    tts.setParameter("speed", 85)
    tts.say("Hello, my name is Pepper and I am an assistive robot.")
    tts.say("Today I am going to go through the cam I.C.U. interview process with you.")
    tts.say("This will involve touching my hand, answering a few questions, and showing me a gesture.")
    time.sleep(1.0)

    tts.say("First,")
    feature_2.interview(LETTERS)
    time.sleep(1.0)

    posture.goToPosture("Stand", 1.0)
    tts.say("Now,")
    try:
        feature_4_1.start()
        feature_4_1.interview(QUESTIONS, ANSWERS)
    except RuntimeError as e:
        print(e)
    feature_4_1.stop()
    time.sleep(1.0)

    posture.goToPosture("Stand", 1.0)
    tts.say("Finally,")
    feature_4_2.interview()
