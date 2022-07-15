from naoqi import ALProxy
import qi
import time

app = qi.Application(["CAMICU", "--qi-url=tcp://169.254.30.10:9559"])
app.start()
session = app.session
tts = session.service("ALTextToSpeech")
mem = session.service("ALMemory")
ap = session.service("ALAudioPlayer")

letters = "SAVEAHAART"
letters = letters.lower()
letters = [letter for letter in letters]
timeout = 2.0
responses = []
tts.say("I am going to read you a series of 10 letters.")
tts.say("Whenever you hear the letter A, indicate by touching the top of my hand.")
tts.say("I will make a beep sound once you touch my hand.")

for letter in letters:
    print(letter)
    tts.say(letter)
    start_time = time.time()
    touched = 0

    while time.time() - start_time < timeout:
        value_l = mem.getData("Device/SubDeviceList/LHand/Touch/Back/Sensor/Value")
        value_r = mem.getData("Device/SubDeviceList/RHand/Touch/Back/Sensor/Value")
        if value_l or value_r:
            touched = 1
            ap.playSine(500, 50, 0, 0.1)
            time.sleep(timeout - (time.time() - start_time))
            break

        time.sleep(0.1)
    
    print(letter, touched)
    responses.append(touched)

errors = 0
for i, response in enumerate(responses):
    if letters[i] == 'a' and not response:
        errors +=1

print("errors: {}".format(errors))
if errors > 2:
    tts.say("oh no, {} errors. looks like you are delirious".format(errors))
else:
    tts.say("{} errors, you are fine".format(errors))