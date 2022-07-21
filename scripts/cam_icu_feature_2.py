from naoqi import ALProxy
import qi
import time

LETTERS = "SAVEAHAART"

class Inattention:
    def __init__(self, session):
        self.tts = session.service("ALTextToSpeech")
        self.mem = session.service("ALMemory")
        self.ap = session.service("ALAudioPlayer")
        self.posture = session.service("ALRobotPosture")
    
    def say_introduction(self):
        self.tts.say("I am going to read you a series of 10 letters.")
        self.tts.say("Whenever you hear the letter")
        time.sleep(0.25)
        self.tts.say("A.")
        time.sleep(0.25)
        self.tts.say("indicate by touching the top of my hand.")
        self.tts.say("I will make this sound:")
        time.sleep(0.25)
        self.ap.playSine(500, 50, 0, 0.1)
        time.sleep(0.25)
        self.tts.say("once you touch my hand.")
        self.posture.goToPosture("StandZero", 0.5)

        self.tts.say("Remember to only touch my hand when you hear the letter. A.")
        self.tts.say("Let's begin in")
        self.tts.say("3")
        time.sleep(0.5)
        self.tts.say("2")
        time.sleep(0.5)
        self.tts.say("1")
        time.sleep(0.5)
    
    def interview(self, letters):
        letters = letters.lower()
        letters = [letter for letter in letters]
        timeout = 2.0
        responses = []

        self.say_introduction()

        for letter in letters:
            print(letter)
            self.tts.say(letter)
            start_time = time.time()
            touched = 0

            while time.time() - start_time < timeout:
                value_l = self.mem.getData("Device/SubDeviceList/LHand/Touch/Back/Sensor/Value")
                value_r = self.mem.getData("Device/SubDeviceList/RHand/Touch/Back/Sensor/Value")
                if value_l or value_r:
                    touched = 1
                    self.ap.playSine(500, 50, 0, 0.1)
                    time.sleep(timeout - (time.time() - start_time))
                    break

                time.sleep(0.1)
            
            print(letter, touched)
            responses.append(touched)

        errors = 0
        for i, response in enumerate(responses):
            if (letters[i] == 'a' and not response) or (not letters[i] == 'a' and response):
                errors +=1

        self.posture.goToPosture("Stand", 0.5)
        return errors

if __name__ == "__main__":
    ip = "169.254.165.104"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:9559".format(ip, port)])
    app.start()
    session = app.session

    feature_2 = Inattention(session)
    feature_2.interview(LETTERS)
