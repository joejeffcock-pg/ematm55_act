from cam_icu_feature_2 import *
from cam_icu_feature_4_1 import *
from cam_icu_feature_4_2 import *

if __name__ == "__main__":
    ip = "169.254.45.131"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:9559".format(ip, port)])
    app.start()
    session = app.session

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5560")

    feature_2 = Inattention(session)
    feature_4_1 = DisorganisedThinkingYesNo(session)

    feature_2.interview(LETTERS)
    feature_4_1.interview(QUESTIONS, ANSWERS)
    feature_4_2 = DisorganisedThinkingFingers(session, socket)
    feature_4_2.interview()

    feature_4_1.stop()
