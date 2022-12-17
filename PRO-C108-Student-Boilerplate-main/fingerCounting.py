import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)
tipIds = [4, 8, 12, 16, 20]

#Defining a functions to draw connections
def drawHandLandmarks(image, hand_landmarks):

    #Draw Connections between landmark points
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        #get all landmarks of the first hand visible
        landmarks = hand_landmarks[handNo].landmark
        #print(landmarks)

        #Count Fingers
        fingers = []

        for lm_index in tipIds:
            #Get Finger Tips and Bottoms Y position value
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index-2].y

            #Check if any finger is open or closed
            if lm_index != 4 :
                if finger_tip_y < finger_bottom_y :
                    fingers.append(1)
                    print('FINGER with id', lm_index, 'is open')

                if finger_tip_y > finger_bottom_y :
                    fingers.append(0)
                    print('FINGER with id', lm_index, 'is closed')

        print(fingers)
        totalFingers = fingers.count(1)

        #display Text
        text = f'Fingers : {totalFingers}'
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

while True :
    success, img = video.read()
    image = cv2.flip(img, 1)

    #Detect the hands landmarks
    results = hands.process(img)

    #get Landmark positions from the processed results
    hand_landmarks = results.multi_hand_landmarks

    #Draw Landmarks
    drawHandLandmarks(img, hand_landmarks)

    #Get hand fingers positions
    countFingers(image, hand_landmarks)

    cv2.imshow('Media Controller', img)
    key = cv2.waitKey(1)
    if key == 32 :
        break

cv2.destroyAllWindows()
video.release()