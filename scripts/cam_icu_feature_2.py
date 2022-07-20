from naoqi import ALProxy
import qi
import time

LETTERS = "SAVEAHAART"

class Inattention:
    def __init__(self, session):
        self.tts = session.service("ALTextToSpeech")
        self.mem = session.service("ALMemory")
        self.ap = session.service("ALAudioPlayer")
    
    def interview(self, letters):
        letters = letters.lower()
        letters = [letter for letter in letters]
        timeout = 2.0
        responses = []
        self.tts.say("I am going to read you a series of 10 letters.")
        self.tts.say("Whenever you hear the letter A, indicate by touching the top of my hand.")
        self.tts.say("I will make a beep sound once you touch my hand.")

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

        print("errors: {}".format(errors))
        if errors > 2:
            self.tts.say("oh no, {} errors. looks like you are delirious".format(errors))
        else:
            self.tts.say("{} errors, you are fine".format(errors))
        
        return errors


if __name__ == "__main__":
    ip = "169.254.165.104"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:9559".format(ip, port)])
    app.start()
    session = app.session

    feature_2 = Inattention(session)
    feature_2.interview(LETTERS)
