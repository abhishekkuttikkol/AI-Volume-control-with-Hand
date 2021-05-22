import cv2
import time
import mediapipe as mp 
import numpy
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm

def main():
    ################################
    wCam, hCam = 640, 480
    ################################

    cap = cv2.VideoCapture(0)
    # cap.open('http://192.168.1.199:8080/video') 
    cap.set(3, wCam)    
    cap.set(4, hCam)
    pTime = 0
    CTime = 0

    detector = htm.handDetector(detectionCon=0.7)

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]

    volBar = 0
    volPer = 0
    area = 0

    while cap.isOpened():
        success, image = cap.read()
        image = detector.findhands(image)
        lmLists, bbox = detector.findPosition(image, draw=False)
        fingers = detector.fingersUp()
        if len(lmLists) != 0:
            area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))//100
            if 250 < area < 750:

                length, image, lineInfo = detector.findDistance(4, 8, image)
                if length < 50:
                    cv2.circle(image, (lineInfo[4], lineInfo[5]), 7, (0, 255, 0), cv2.FILLED)
        
                if fingers[1] == 1 and fingers[2] != 1:
                    volBar = numpy.interp(length, [50, 150], [400, 150])
                    volPer = numpy.interp(length, [50, 150], [0, 100])
                    volume.SetMasterVolumeLevelScalar(volPer/100, None)

                cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(image, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(image, f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
                if fingers[1] == 1 and fingers[2] == 1:
                    cv2.circle(image, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                
        CTime = time.time()
        fps = 1/(CTime-pTime)
        pTime = CTime
        cv2.putText(image, f'FPS:{int(fps)}',(20,40),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),4)

        cv2.imshow("image",image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()  

if __name__ == '__main__':
    main()