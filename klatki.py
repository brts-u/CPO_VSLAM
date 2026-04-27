import cv2
import numpy as np
import matplotlib.pyplot as plt



# 2. Wczytanie wideo
cap = cv2.VideoCapture(r"C:\Users\domin\Pictures\Camera Roll\WIN_20260427_17_13_44_Pro.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 20) 
_, frame1 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 30) 
_, frame2 = cap.read()

h, w = frame1.shape[:2]

fx = fy = 0.8 * w   
cx = w / 2
cy = h / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

def fast(zdj): 
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)  # Konwersja obrazu do postaci zdjęcia w odcieniach szarości
    fast = cv2.FastFeatureDetector.create()
    kp = fast.detect(szareZdjecie, None)
    orb = cv2.ORB_create()
    kp, des = orb.compute(szareZdjecie, kp)
    img = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255, 0, 0))
    return img, kp, des


_,kp1, des1 = fast(frame1)
_, kp2, des2 = fast(frame2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches=[]
pts1, pts2 = [], []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

inliers = mask.ravel().astype(bool) 
pts1_in = pts1[inliers] 
pts2_in = pts2[inliers] 

_, R, T, maskpose = cv2.recoverPose(E, pts1_in, pts2_in, K)

good_matches_inliers = []
for i, m in enumerate(good_matches):
    if mask.ravel()[i] == 1: # Punkt został uznany za inlier w RANSAC
        good_matches_inliers.append(m) # Przechowujemy obiekt dopasowania

img3 = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches_inliers, None, 
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3)
plt.show()