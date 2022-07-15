import qi
import time

app = qi.Application(["CAMICU", "--qi-url=tcp://169.254.27.249:9559"])
app.start()
session = app.session
asr = session.service("ALSpeechRecognition")
mem = session.service("ALMemory")
tts = session.service("ALTextToSpeech")
ap = session.service("ALAudioPlayer")
asr.subscribe("Test_ASR")

def beep_on(high=750, low=500):
    ap.playSine(low, 25, 0, 0.05)
    time.sleep(0.1)
    ap.playSine(low, 25, 0, 0.05)
    time.sleep(0.1)
    ap.playSine(high, 25, 0, 0.1)
    time.sleep(0.15)

def beep_off(high=750, low=500):
    ap.playSine(high, 25, 0, 0.1)
    time.sleep(0.15)
    ap.playSine(low, 25, 0, 0.1)
    time.sleep(0.15)

# the pause is causing issues!!! ):<
asr.pause(1)
asr.setLanguage("English")
vocabulary = ["yes", "no"]
asr.setVocabulary(vocabulary, False)

timeout = 20

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

failures = 0
tts.say("Could you please answer the following questions with yes or no responses?")
time.sleep(0.5)

for i, question in enumerate(QUESTIONS):
    # if mem.getData("WordRecognized"):
    #     mem.removeData("WordRecognized")

    tts.say(question)
    asr.pause(0)
    beep_on()
    start_time = time.time()
    while time.time() - start_time < timeout:
        word, score = mem.getDataOnChange("WordRecognized", 0)
        
        # word, score = mem.getData("WordRecognized")
        print(word, score)
        if not word == "" and score > 0.5:
            print(word)
            break
        time.sleep(0.1)
    
    if not word == ANSWERS[i]:
        failures += 1
    
    asr.pause(1)
    beep_off()
    time.sleep(0.5)

tts.say("{} errors".format(failures))
if failures > 1:
    tts.say("oh no, you have disorganised thinking")
else:
    tts.say("oh no, you have organised thinking")

asr.unsubscribe("Test_ASR")
