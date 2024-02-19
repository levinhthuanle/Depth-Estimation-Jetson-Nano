# calibration camera

import numpy as np
import cv2
import glob

baseline = 0.016 # m

# calibration camera
def calibration_camera():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('chessboards/*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(50)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # save camera matrix and distortion coefficients
    with open("calibration.txt", "w") as f:
        f.write("camera matrix:\n")
        f.write(str(mtx[0, 0]) + " " + str(mtx[0, 1]) + " " + str(mtx[0, 2]) + "\n")
        f.write(str(mtx[1, 0]) + " " + str(mtx[1, 1]) + " " + str(mtx[1, 2]) + "\n")
        f.write(str(mtx[2, 0]) + " " + str(mtx[2, 1]) + " " + str(mtx[2, 2]) + "\n")
        f.write("distortion coefficients:\n")
        f.write(str(dist[0, 0]) + " " + str(dist[0, 1]) + " " + str(dist[0, 2]) + " " + str(dist[0, 3]) + " " + str(dist[0, 4]) + "\n")

    return mtx, dist

# load camera matrix and distortion coefficients
def load_camera_matrix(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        mtx = lines[1:4]
        dist = lines[5:6]
        mtx = [line.split(" ") for line in mtx]
        mtx = [[float(x) for x in line] for line in mtx]
        mtx = np.array(mtx)
        dist = [line.split(" ") for line in dist]
        dist = [[float(x) for x in line] for line in dist]
        dist = np.array(dist)
    return mtx, dist

# calculate distance
def calculate_distance(mtx, dist, image):
    # load image
    img = cv2.imread(image)
    img = cv2.resize(img, (640, 480))

    # undistort image
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibration/calibresult.png', dst)

    # convert to gray
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(dst, (9, 6), corners, ret)
        cv2.imshow('img', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # calculate distance
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        objpoints.append(objp)
        imgpoints.append(corners)
        ret, rvecs, tvecs = cv2.solvePnP(objpoints[0], imgpoints[0], mtx, dist)
        distance = np.linalg.norm(tvecs) * baseline
        print("distance: ", distance, "m")

        return distance

    else:
        print("Error: chessboard not found")
        return 0

if __name__ == '__main__':
    # calibration camera
    mtx, dist = calibration_camera()

    # load camera matrix and distortion coefficients
    # mtx, dist = load_camera_matrix("calibration.txt")

    # calculate distance
    calculate_distance(mtx, dist, "chessboards/chessboard_4.png")
    