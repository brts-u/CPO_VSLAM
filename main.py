import os
import numpy as np
import cv2
import laspy
import matplotlib.pyplot as plt


video_file_name = r'szczeki2'

cap = cv2.VideoCapture(r'C:\aszkola\6 sem\cyfrowe przetwarzanie obrazow\CPO_VSLAM\\' + video_file_name + '.mp4')
if os.environ['COMPUTERNAME'] == 'LAPTOP-5E0LJ6KE': # dla laptopa Bartka
    cap = cv2.VideoCapture(video_file_name + '.mp4')

def fast(zdj):
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)  # Konwersja obrazu do postaci zdjęcia w odcieniach szarości
    fast = cv2.FastFeatureDetector.create()
    kp = fast.detect(szareZdjecie, None)
    orb = cv2.ORB_create()
    kp, des = orb.compute(szareZdjecie, kp)
    img = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255, 0, 0))
    return img, kp, des

def orb(zdj):
    szareZdjecie = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kp, des = orb.detectAndCompute(szareZdjecie, None)
    img = cv2.drawKeypoints(szareZdjecie, kp, None, color=(255, 0, 0))
    return img, kp, des

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


def plot_trajectory(trajectory):
    traj = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], marker='o')
    plt.show()

_, frame = cap.read()

h, w = frame.shape[:2]
print(f'w:{w}')
print(f'h:{h}')


fx = fy = 0.8 * w   
cx = w / 2
cy = h / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)


def main(detector = orb):
    count = 0
    # Capture first frame to initialize
    ret_prev, frame_prev = cap.read()
    img_prev, kp_prev, des_prev = detector(frame_prev)
   
    global_points = []
    trajectory =[]
    curr_xyz = np.array([0,0,0])
    trajectory.append(curr_xyz)

    global_pose = np.hstack((np.eye(3), np.zeros((3, 1))))
    prev_pose = np.hstack((np.eye(3), np.zeros((3, 1))))

    #macierz projekcji : rotacja jednostkowa - brak obrotu, translacja zerowa (brak przesunięcia)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1)))) # => 3x4 

    while True:
        count += 1
        if not count % 15 == 1:
            continue
        
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

        # macierz transformacji miedzy klatkami
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T.reshape(3)

        # global_pose = global_pose @ np.linalg.inv(Trel)
        global_pose = global_pose @ H

        cam_center = global_pose[:3, 3] #ostatni kolumna
        trajectory.append(cam_center)

        P1 = K @ prev_pose
        P2 = K @ global_pose

        points3D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T) 
        points3D = points3D/points3D[3] 
        points3D = points3D[:3,:].T 
                    
        global_points.append(points3D)

        prev_pose = global_pose.copy()
        kp_prev, des_prev = kp, des

        
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # wyjście za pomocą klawisza 'Q'
            break

    cap.release()
    cv2.destroyAllWindows()
    all_pts = np.vstack(global_points)
    
    plot_trajectory(trajectory)

    print('Zapisywanie chmury punktów')
    try:
        write_las(all_pts, f'{video_file_name}_{detector.__name__}.laz')
    except Exception as e:
        print(f"Nie można zapisać do LAZ: {e}")
        np.savetxt(f'{video_file_name}_{detector.__name__}.txt', all_pts, fmt='%.6f')

if __name__ == "__main__":
    main(orb)