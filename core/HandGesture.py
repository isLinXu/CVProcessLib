import cv2
import mediapipe as mp
import time
import math
import numpy as np


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionConfidence,
                                        min_tracking_confidence=self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # Converting to RGB image for mediapipe to process
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Detecting Landmark points on Hand and drawing them on input image
        if self.results.multi_hand_landmarks:
            for hand_lndmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lndmks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """

        :param img:
        :param handNo:
        :param draw:
        :return:
        """
        lndmkList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lndmkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)

        return lndmkList


def get_frame_rate_info(pTime):
    """
    Compute and Return FPS Information
    :return: int -> fps
    """

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    return fps, cTime, pTime


def main():
    pTime = 0
    # Capture input Video Stream:
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        # Capturing input from Webcam
        success, img = cap.read()
        img = detector.findHands(img)
        fps, cTime, pTime = get_frame_rate_info(pTime)
        lndmarkList = detector.findPosition(img)
        if len(lndmarkList) != 0:
            print(lndmarkList[4])
        # Display CV Window on screen
        cv2.putText(img, "Frame Rate: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
