import os
import numpy as np
import cv2
import laspy

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
    cap = cv2.VideoCapture(r'szczeki2.mp4')

def write_las(points, file_path='pcd.laz'):
    # Calculate offsets and scales
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    range_vals = max_vals - min_vals

    # Set scale to fit the range within LAS limits
    scale = range_vals / (2**31 - 1)  # Adjust scale to avoid overflow
    scale[scale == 0] = 1  # Avoid division by zero for constant dimensions

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = min_vals
    header.scales = scale

    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]

    las.write(file_path)

# K = np.array([ 
# [1080,0,972], #fx, 0, cx 
# [0,1080,1296], #0, fy, cy 
# [0,0,1]], #0,0,1 
# dtype = np.float32) 
# 

_, frame = cap.read()

w, h = frame.shape[:2]

fx = fy = 0.8 * w   
cx = w / 2
cy = h / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

distCoeffs = np.zeros(5)

def main(detector = orb):
    # Capture first frame to initialize
    ret_prev, frame_prev = cap.read()
    img_prev, kp_prev, des_prev = detector(frame_prev)
   
    global_points = []
    # Macierz transformacji (4x4) opisująca, gdzie jest kamera w świecie
    global_pose = np.eye(4) 
    prev_pose = np.eye(4)

    #macierz projekcji : rotacja jednostkowa - brak obrotu, translacja zerowa (brak przesunięcia)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1)))) # => 4x4 
    print(P1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img, kp, des = detector(frame)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_prev, des, k=2)
                
        pts1 = [] 
        pts2 = [] 
     
        for i,(m,n) in enumerate(matches): 
            if m.distance < 0.75*n.distance: 
                pts2.append(kp[m.trainIdx].pt) 
                pts1.append(kp_prev[m.queryIdx].pt)

        pts1 = np.asarray(pts1, dtype=np.float32)  
        pts2 = np.asarray(pts2, dtype=np.float32)

        if np.any(distCoeffs != 0): 
            pts1_und = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, distCoeffs) 
            pts2_und = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, distCoeffs) 

        E, inliers = cv2.findEssentialMat( 
        pts1, pts2, 
        cameraMatrix=K, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0 
        ) 

        inliers = inliers.ravel().astype(bool) 
        pts1_in = pts1[inliers] 
        pts2_in = pts2[inliers] 

        retval, R, T, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K) 

        #macierz transformacji miedzy klatkami
        Trel = np.eye(4)
        Trel[:3,:3]=R
        Trel[:3,3] = T.flatten()

        #global_pose = global_pose @ np.linalg.inv(Trel)
        global_pose = global_pose @ Trel

        P1 = K @ prev_pose[:3, :]
        P2 = K @ global_pose[:3, :]

        points3D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T) #zwraca P = XYZW (4 x N)
        #normalizacja:
        points3D /= points3D[3]   # dzielę przez 4 wiersz (W) 
        points3D = points3D[:3]   # zostawiam XYZ => 3xN
              
        #points3D_h = np.vstack((points3D, np.ones(points3D.shape[1]))) # => 4 x N
        #points3d_global = (global_pose @ points3D_h).T[:, :3] # 4x4 @ 4xN => 4xN.T => Nx4
        #global_points.append(points3d_global)

        global_points.append(points3D.T)

        prev_pose = global_pose.copy()
        kp_prev, des_prev = kp, des

        
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # wyjście za pomocą klawisza 'Q'
             break
    cap.release()
    cv2.destroyAllWindows()

    print('Zapisywanie chmury punktów')
    all_pts = np.vstack(global_points)
    write_las(all_pts)


if __name__ == "__main__":
    main(orb)