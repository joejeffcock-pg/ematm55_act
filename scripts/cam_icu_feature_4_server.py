import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import zmq
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from math import atan2, pi, degrees

def get_angle(p1, p2):
  return atan2(p2[1] - p1[1], p2[0] - p1[0])

def is_up(base, p1, p2):
  offset = get_angle((base.x, base.y), (p1.x, p1.y))
  angle = get_angle((p1.x, p1.y), (p2.x, p2.y))
  # print('{:.3f} {:.3f} {:.3f}'.format(degrees(angle), degrees(offset), degrees(angle - offset)))
  angle = angle - offset
  if angle > -pi/4.0 and angle < pi/4.0:
    return 1
  return 0

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5560")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while 1:
    # get image
    array = socket.recv()
    image = Image.frombuffer("RGB", (640, 480), array)
    image = np.array(image)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_str = None
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # construct result string
        crop = 0.5
        crop_side = crop/2.0
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        if wrist.x > crop_side and wrist.x < 1 - crop_side and wrist.y > crop_side and wrist.y < 1 - crop_side:
          result_str = '{}{}{}{}{}'.format(is_up(*thumb), is_up(*index), is_up(*middle), is_up(*ring), is_up(*pinky))
          print(result_str)
          break
        
    # send result
    if result_str is not None:
      socket.send(str.encode(result_str))
    else:
      socket.send(str.encode(""))

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()