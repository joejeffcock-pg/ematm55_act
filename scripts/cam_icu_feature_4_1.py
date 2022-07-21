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
    
    def say_introduction(self):
        self.tts.say("Could you please answer the following questions with")
        time.sleep(0.25)
        self.tts.say("yes")
        time.sleep(0.25)
        self.tts.say("or")
        time.sleep(0.25)
        self.tts.say("no")
        time.sleep(0.25)
        self.tts.say("responses?")
        time.sleep(0.5)

    def interview(self, questions, answers):
        self.asr.pause(1)
        self.asr.setLanguage("English")
        vocabulary = list(set(answers))
        self.asr.setVocabulary(vocabulary, False)

        self.say_introduction()

        failures = 0
        for i, question in enumerate(questions):
            self.asr.pause(1)
            self.tts.say(question)
            time.sleep(0.5)

            self.mem.insertData("WordRecognized", ("", 0.0))
            self.asr.pause(0)

            start_time = time.time()
            while time.time() - start_time < 5.0:
                word, score = self.mem.getData("WordRecognized")
                print(word, score)
                if not word == "" and score >= 0.4:
                    break
                time.sleep(0.1)
            
            if not (word == ANSWERS[i] and score < 0.4):
                failures += 1
            time.sleep(0.5)

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