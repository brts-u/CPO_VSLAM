import os
import numpy as np
import cv2

def fast(zdj): # TODO: coś nie działa z detectAndCompute
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)  # Konwersja obrazu do postaci zdjęcia w odcieniach szarości
    fast = cv2.FastFeatureDetector.create()
    kp, des = fast.detectAndCompute(szareZdjecie, None)
    img = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255, 0, 0))
    return img, kp, des

def orb(zdj):
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kp, des = orb.detectAndCompute(szareZdjecie, None)
    img = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255, 0, 0))
    return img, kp, des

cap = cv2.VideoCapture(r'C:\aszkola\6 sem\cyfrowe przetwarzanie obrazow\CPO_VSLAM\szczeki2.mp4')
if os.environ['COMPUTERNAME'] == 'LAPTOP-5E0LJ6KE': # dla laptopa Bartka
    cap = cv2.VideoCapture(r'szczeki1.mp4')

def main(detector = orb):
    # Capture first frame to initialize
    ret_prev, frame_prev = cap.read()
    img_prev, kp_prev, des_prev = detector(frame_prev)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img, kp, des = detector(frame)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_prev, des)
        matches = sorted(matches, key=lambda x: x.distance)

        kp_prev, des_prev = kp, des

        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # wyjście za pomocą klawisza 'Q'
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(orb)