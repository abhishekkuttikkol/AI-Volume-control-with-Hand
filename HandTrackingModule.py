import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipId = [ 4, 8, 12, 16, 20]

    def findhands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLMS, self.mpHands.HAND_CONNECTIONS)
        return image

    def findPosition(self, image, handNo=0, draw=True):
        yList = []
        xList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHands = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHands.landmark):
                # print(id,lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                # print(id,cx,cy)
                # if id == 4:
                # print(id,cx,cy)
                self.lmList.append([id,cx,cy])

                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(image, (bbox[0] -20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    def findDistance(self, p1, p2, image, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        
        if draw:
            cv2.circle(image, (x1, y1), 7, (255,0,255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 7, (255,0,255), cv2.FILLED)
            cv2.circle(image, (cx, cy), 7, (255,0,255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x1 - x2, y1 - y2)
        lineInfo = [x1, y1, x2, y2, cx, cy]
        return length, image, lineInfo
    
    def fingersUp(self):
        fingers = []
        if len(self.lmList) !=0:
            if self.lmList[self.tipId[0]][2] < self.lmList[self.tipId[0]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1,5):
                if self.lmList[self.tipId[id]][2] < self.lmList[self.tipId[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    
def main():
    pTime = 0
    CTime = 0
    cam = cv2.VideoCapture()
    cam.open('http://192.168.1.166:8080/video')   
    detector = handDetector()

    while cam.isOpened():
        success,image = cam.read()
        image =  detector.findhands(image)
        lmList = detector.findPosition(image)
        if len(lmList) != 0:
            print(lmList[4])

        CTime = time.time()
        fps = 1/(CTime-pTime)
        pTime = CTime
        cv2.putText(image,str(int(fps)),(20,40),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("image",image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()    


