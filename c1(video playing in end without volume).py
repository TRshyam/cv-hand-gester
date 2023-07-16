from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
gestureThreshold = 300
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False

# Get list of presentation files
pathFiles = sorted(os.listdir(folderPath), key=len)
print(pathFiles)

# Initialize video variable
video = None
videoPlaying = False

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Display image or video
    currentFile = pathFiles[imgNumber]
    filePath = os.path.join(folderPath, currentFile)
    fileExtension = os.path.splitext(currentFile)[1]

    if fileExtension.lower() in ['.jpg', '.jpeg', '.png']:  # Image file
        imgCurrent = cv2.imread(filePath)
    elif fileExtension.lower() in ['.mp4', '.avi', '.mov']:  # Video file
        if not buttonPressed and not videoPlaying:
            video = cv2.VideoCapture(filePath)
            buttonPressed = True
            videoPlaying = True

        if videoPlaying:
            success, frame = video.read()
            if not success:
                video.release()
                buttonPressed = False
                videoPlaying = False
                continue

            imgCurrent = frame

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:  # If hand is detected
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            elif fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathFiles) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop()
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for annotation in annotations:
        for j in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    small_img_scale = 0.2  # Adjust this scale to control the size of the small image
    small_img_width = int(width * small_img_scale)
    small_img_height = int(height * small_img_scale)

    imgSmall = cv2.resize(img, (small_img_width, small_img_height))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:small_img_height, w - small_img_width:w] = imgSmall

#responsive design
    window_name = "Slides"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = imgCurrent.shape[1]
    window_height = imgCurrent.shape[0]

    # Calculate the scale factor for resizing
    scale_factor = min(window_width / imgCurrent.shape[1], window_height / imgCurrent.shape[0])

    # Resize the image
    resized_width = int(imgCurrent.shape[1] * scale_factor)
    resized_height = int(imgCurrent.shape[0] * scale_factor)
    imgCurrent = cv2.resize(imgCurrent, (resized_width, resized_height))


    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

