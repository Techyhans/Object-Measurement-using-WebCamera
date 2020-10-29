import cv2
import numpy as np

def getContors(img, cThr=[100, 100], showCanny = False, minArea=1000, filter=0, draw=False):
    # convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) # 5 is kernal size, 1 is sigma

    # find edges
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1]) #cThr = threshold

    # dilation and erodion
    kernal = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernal, iterations=3)
    imgThres = cv2.erode(imgDial, kernal, iterations=2)

    if showCanny:
        cv2.imshow("Canny", imgThres)
    
    #find contours
    contours, hiearchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    #area of contours [current detected object]
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)

            # just need rectangle [4 corner], filter == num of corver
            if filter > 0:
                if len(approx) == filter:
                    # put into final list
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])
    
    #sort contours based on size
    finalContours = sorted(finalContours, key=lambda x:x[1], reverse=True)

    #draw contours
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0,0,255), 3) #(0,255,255) = red, 3 = thinkness
    
    return img, finalContours

# make ture the rectangle point sequence always [1 2 3 4], 1 always the smallest value, 4 is w+h
def reorder(myPoints):
    print(myPoints.shape) # (4, 1, 2), 4 is num of points, 2 is x and y of each point, 1 is useless
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] # get minimun value, and that value is for point 1
    myPointsNew[3] = myPoints[np.argmax(add)] # get maximun value, and that value is for point 4
    diff = np.diff(myPoints, axis=1) # do differenciation
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew # at this point the rectangle will be in order of [1 2 3 4]

def wrapImg(img, points, w, h, pad=20):
    #print(points)
    #print(reorder(points))

    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWrap = cv2.warpPerspective(img, matrix, (w,h))

    # remove some padding
    imgWrap = imgWrap[pad:imgWrap.shape[0]-pad, pad:imgWrap.shape[1]-pad]

    return imgWrap

# find length of object
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2)**0.5


webcam = False
path = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160) #brightness
cap.set(3, 1920) #width
cap.set(4, 1080) #height

# size of paper and scale up
scale = 3
wP = 210 * scale
hP = 297 * scale


while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    img, conts = getContors(img, minArea=50000, filter=4, draw=True)

    # make sure contours list not empty
    if len(conts) != 0:
        biggest = conts[0][2] # get approx
        print(biggest)
        imgWrap = wrapImg(img, biggest, wP, hP)
        cv2.imshow("A4 paper", imgWrap)

        # find contours in A4 paper
        imgContours2, conts2 = getContors(imgWrap, minArea=2000, filter=4, draw=False, cThr=[50, 50])
        if len(conts2) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2) #draw lines in green color and thickness of 2, draw properly
                nPoints = reorder(obj[2])

                #width
                nW = round(findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10), 1 # devide 10 to get in CM unit, witn 1 dp

                #height
                nH = round(findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10), 1 # devide 10 to get in CM unit, witn 1 dp

                print(nW, nH)

        cv2.imshow("Img Contours2", imgContours2)

    img = cv2.resize(img, (0,0), None, 0.5, 0.5) #scale down 0.5
    cv2.imshow("Original", img)
    cv2.waitKey(1) # 1 milisecond