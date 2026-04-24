import numpy as np 
import cv2 

def fast(zdj1):
    szareZdjecie = cv2.cvtColor(zdj1, cv2.COLOR_BGR2GRAY) # Konwersja obrazu do postaci zdjęcia w odcieniach szarości 
    fast = cv2.FastFeatureDetector.create() 
    kp1, des1 = fast.detectAndCompute(szareZdjecie, None) 
    img2 = cv2.drawKeypoints(szareZdjecie, kp1, None, color=(255,0,0)) 
    return img2, kp1, des1

   
def orb(zdj):
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000,scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20 ) 
    kp = orb.detect(szareZdjecie, None) 
    img2 = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255,0,0)) 
    return img2

cap = cv2.VideoCapture(r'C:\aszkola\6 sem\cyfrowe przetwarzanie obrazow\CPO_VSLAM\szczeki2.mp4') 

kp_prev, des_prev = None, None
while (True): 
    if des_prev is None:
        ret, frame_prev = cap.read()
        _, kp_prev, des_prev = fast(frame_prev,None)
        continue

    # Capture frame-by-frame 
    ret, frame = cap.read() 

    # Display the resulting frame
 
    _, kp, des = fast(frame,None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_prev,des) 
    matches = sorted(matches, key = lambda x:x.distance) 
    
    kp_prev, des_prev = kp, des



    #cv2.imshow('frame', img2) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #wyjście za pomocą klawisza 'Q'
        break 
    
# When everything done, release the capture 
cap.release() 
cv2.destroyAllWindows()

