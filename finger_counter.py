import cv2
import mediapipe as mp
import numpy as np


def not_lies_inside(x, c, r, pos):
    dis = np.sqrt((c[0]-pos[x][0])**2+(c[1]-pos[x][1])**2)
    if dis > r:
        return True
    return False


Hands = mp.solutions.hands
hands = Hands.Hands(static_image_mode=False, min_tracking_confidence=0.5, max_num_hands=3, min_detection_confidence=0.8)
Draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(1):
    _, frame = cap.read()
    hand_pos = []
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            for pos in hand.landmark:
                cx, cy = int(pos.x*frame.shape[1]), int(pos.y*frame.shape[0])
                hand_pos.append(np.array([cx, cy], np.int32))
        #Draw.draw_landmarks(frame, hand, Hands.HAND_CONNECTIONS)
    if len(hand_pos):
        circle_pos = np.array([hand_pos[0], hand_pos[2], hand_pos[6], hand_pos[10], hand_pos[14],
                               hand_pos[18]], np.int32)
        c, r = cv2.minEnclosingCircle(circle_pos)
        tips = [4, 8, 12, 16, 20]
        tip_pos = np.array([not_lies_inside(x, c, r, hand_pos) for x in tips], np.bool)
        num = np.sum(tip_pos[:])
        rw = 200 + (num == 10)*200
        frame = cv2.rectangle(frame, (0, 0), (rw, 200), (255, 0, 0), -1)
        cv2.putText(frame, str(num), (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 20)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()