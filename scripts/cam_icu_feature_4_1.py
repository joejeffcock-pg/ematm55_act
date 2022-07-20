import qi
import time

QUESTIONS = [
    "Will a stone float on water?",
    "Are there fish in the sea?",
    "Does one pound weigh more than two pounds?",
    "Can you use a hammer to pound a nail?"
]

ANSWERS = [
    "no",
    "yes",
    "no",
    "yes"
]

class DisorganisedThinkingYesNo:
    def __init__(self, session, auto_start=True):
        self.asr = session.service("ALSpeechRecognition")
        self.mem = session.service("ALMemory")
        self.tts = session.service("ALTextToSpeech")
        self.ap = session.service("ALAudioPlayer")
        if auto_start:
            self.start()
        
    def start(self):
        self.asr.subscribe("Test_ASR")
    
    def stop(self):
        self.asr.unsubscribe("Test_ASR")
    
    def beep_on(self, high=750, low=500):
        self.ap.playSine(low, 25, 0, 0.05)
        time.sleep(0.1)
        self.ap.playSine(low, 25, 0, 0.05)
        time.sleep(0.1)
        self.ap.playSine(high, 25, 0, 0.1)
        time.sleep(0.15)

    def beep_off(self, high=750, low=500):
        self.ap.playSine(high, 25, 0, 0.1)
        time.sleep(0.15)
        self.ap.playSine(low, 25, 0, 0.1)
        time.sleep(0.15)

    def interview(self, questions, answers):
        self.asr.pause(1)
        self.asr.setLanguage("English")
        vocabulary = list(set(answers))
        self.asr.setVocabulary(vocabulary, False)

        self.tts.say("Could you please answer the following questions with yes or no responses?")
        time.sleep(0.5)
        self.asr.pause(0)

        failures = 0
        for i, question in enumerate(questions):
            self.tts.say(question)
            time.sleep(0.5)
            self.beep_on()
            start_time = time.time()
            while 1:
                word, score = self.mem.getDataOnChange("WordRecognized", 0)
                # word, score = mem.getData("WordRecognized")

                print(word, score)
                if not word == "" and score > 0.5:
                    break
                time.sleep(0.1)
            
            if not word == ANSWERS[i]:
                failures += 1
            
            self.beep_off()
            time.sleep(0.5)

        self.tts.say("{} errors".format(failures))
        if failures > 1:
            self.tts.say("oh no, you have disorganised thinking")
        else:
            self.tts.say("oh no, you have organised thinking")

        self.asr.pause(1)

        return failures

if __name__ == "__main__":
    ip = "169.254.165.104"
    port = 9559
    app = qi.Application(["CAMICU", "--qi-url=tcp://{}:9559".format(ip, port)])
    app.start()
    session = app.session

    feature_4_1 = DisorganisedThinkingYesNo(session)
    feature_4_1.interview(QUESTIONS, ANSWERS)
    feature_4_1.stop()