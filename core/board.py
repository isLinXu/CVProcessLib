import numpy as np
import cv2
from collections import deque
from datetime import datetime
import os
#import pyautogui
import img2pdf

directory= r'/home/linxu/PycharmProjects/CVProcessLib/core/'
nw = datetime.now()
curr_time = nw.strftime("%H:%M:%S")
dirname='files_'+str(curr_time)
path=os.path.join(directory,dirname)
os.mkdir(path)
os.chdir(path)

state=1
flag=1
theme=0
#sc_width,sc_height = pyautogui.size()


# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup deques to store separate colors in separate arrays
redpoints = [deque(maxlen=512)]
yellowpoints = [deque(maxlen=512)]
greenpoints = [deque(maxlen=512)]
bluepoints = [deque(maxlen=512)]

greenindex = 0
redindex = 0
yellowindex = 0
blueindex = 0

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0),(255, 0, 0)]
colorIndex = 0
x_width=605
y_height=796
virtualWhiteBoard = np.zeros((x_width,y_height,3)) + 255
# Setup the Paint interface
def whiteBoard (color = 255,):
    global virtualWhiteBoard
    virtualWhiteBoard = np.zeros((x_width,y_height,3)) + color
    if(color==255):
        virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (40+80,x_width-35), 30, (0,0,0), -1)
        virtualWhiteBoard= cv2.circle(virtualWhiteBoard, (120+80,x_width-35), 30, (0,0,0), -1)
        cv2.putText(virtualWhiteBoard, "THEME", (20+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(virtualWhiteBoard, "CLEAR", (100+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (40+80,x_width-35), 30, (255,255,255), -1)
        virtualWhiteBoard= cv2.circle(virtualWhiteBoard, (120+80,x_width-35), 30, (255,255,255), -1)
        cv2.putText(virtualWhiteBoard, "THEME", (20+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(virtualWhiteBoard, "CLEAR", (100+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    #virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (40+80,x_width-35), 30, (0,255,255), -1)
    #virtualWhiteBoard= cv2.circle(virtualWhiteBoard, (120+80,x_width-35), 30, (255,0,255), -1)
    virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (200+80,x_width-35),30, colors[0], -1)
    virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (280+80,x_width-35), 30, colors[1], -1)
    virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (360+80,x_width-35),30, colors[2], -1)
    virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (440+80,x_width-35), 30,colors[3], -1)
    virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (520+80,x_width-35), 30, (255,0,255), -1)
    #virtualWhiteBoard = cv2.circle(virtualWhiteBoard, (600+80,x_width-35), 30, (255,0,255), -1)
    virtualWhiteBoard = cv2.rectangle(virtualWhiteBoard, (765,10), (790,40), colors[0], -1)
    #cv2.putText(virtualWhiteBoard, "THEME", (20+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(virtualWhiteBoard, "CLEAR", (100+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "RED", (180+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "YELLOW", (260+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "GREEN", (340+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "BLUE", (420+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "SAVE", (500+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    #cv2.putText(virtualWhiteBoard, "ERASE", (580+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0,0), 1, cv2.LINE_AA)
    cv2.putText(virtualWhiteBoard, "X", (767, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

whiteBoard()
#cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

def track():
    global Tracking

    Tracking = cv2.circle(Tracking, (40+80,x_width-35), 30, (0,0,0), -1)
    Tracking = cv2.circle(Tracking, (120+80,x_width-35), 30, (0,0,0), -1)
    Tracking = cv2.circle(Tracking, (200+80,x_width-35),30, colors[0], -1)
    Tracking = cv2.circle(Tracking, (280+80,x_width-35), 30, colors[1], -1)
    Tracking = cv2.circle(Tracking, (360+80,x_width-35),30, colors[2], -1)
    Tracking = cv2.circle(Tracking, (440+80,x_width-35), 30, colors[3], -1)
    Tracking = cv2.circle(Tracking, (520+80,x_width-35), 30, (255,0,255), -1)
    #Tracking = cv2.circle(Tracking, (600+80,x_width-35), 30, (255,0,255), -1)
    Tracking = cv2.rectangle(Tracking, (765,10), (790,40), colors[0], -1)
    cv2.putText(Tracking, "THEME", (20+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "CLEAR", (100+80,x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "RED", (180+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "YELLOW", (260+80,x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "GREEN", (340+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "BLUE", (420+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "SAVE", (500+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    #cv2.putText(Tracking, "ERASE", (580+80, x_width-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(Tracking, "X", (767, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



# Load the video
camera = cv2.VideoCapture(-1)

# Keep looping
while True:
    # Grab the current paintWindow
    (grabbed, Tracking) = camera.read()
    Tracking = cv2.resize(Tracking, (800, 600))
    # Check to see if we have reached the end of the video
    if not grabbed:
        break
    Tracking = cv2.flip(Tracking, 1)
    hsv = cv2.cvtColor(Tracking, cv2.COLOR_BGR2HSV)

    # Add the coloring options to the Tracking
    track()
    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours in the image
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
    	# Sort the contours and find the largest one -- we
    	# will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(Tracking, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1]>540:
            if 10+80 <= center[0] <= 70+80 and state:
                if theme == 0:
                    whiteBoard(000)
                    theme=1
                else:
                    whiteBoard(255)
                    theme=0
                state=0
            elif 90+80 <= center[0] <= 150+80: # Clear All
                redpoints = [deque(maxlen=512)]
                yellowpoints = [deque(maxlen=512)]
                greenpoints = [deque(maxlen=512)]
                bluepoints = [deque(maxlen=512)]


                redindex = 0
                yellowindex = 0
                greenindex = 0
                blueindex = 0

                if theme == 0:
                    whiteBoard(255)
                else:
                    whiteBoard(0)

            elif 170+80 <= center[0] <= 230+80:
                    colorIndex = 0 # Blue
            elif 250+80 <= center[0] <= 310+80:
                    colorIndex = 1 # Green
            elif 330+80 <= center[0] <= 390+80:
                    colorIndex = 2 # Red
            elif 410+80 <= center[0] <= 470+80:
                    colorIndex = 3 # Yellow
            elif (490+80 <= center[0] <= 550+80) and flag:

                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    filename='imgat '+str(current_time)+'.jpg'
                    #os.chdir(directory)
                    cv2.imwrite(filename, virtualWhiteBoard)
                    flag=0
            #elif 570+80 <= center[0] <= 630+80:

             #   if theme == 0:
                #    colorIndex=(255,0,255)
              #  else:
               #     colorIndex=(0,0,255) # Earase


        elif(center[1]>=10 and center[1]<=40 and center[0]<=790 and center[0]>=765):
            l1=list(os.listdir('.'))
            l1.sort()
            if len(l1)==0:
                break
            with open("output.pdf", "wb") as f:
                f.write(img2pdf.convert([i for i in l1 if i.endswith(".jpg")]))
            break

        else :
            state=1
            flag=1
            if colorIndex == 0:
                redpoints[redindex].appendleft(center)
            elif colorIndex == 1:
                yellowpoints[yellowindex].appendleft(center)
            elif colorIndex == 2:
                greenpoints[greenindex].appendleft(center)
            elif colorIndex == 3:
                bluepoints[blueindex].appendleft(center)

    # Append the next deque when no contours are detected (i.e., bottle cap reversed)
    else:
        redpoints.append(deque(maxlen=512))
        redindex += 1
        yellowpoints.append(deque(maxlen=512))
        yellowindex += 1
        greenpoints.append(deque(maxlen=512))
        greenindex += 1
        bluepoints.append(deque(maxlen=512))
        blueindex += 1

    # Draw lines of all the colors (Blue, Green, Red and Yellow)
    points = [redpoints, yellowpoints, greenpoints, bluepoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(Tracking, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(virtualWhiteBoard, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show the Tracking and the VirtualWhiteBoard image
    cv2.imshow("Tracking", Tracking)
    cv2.imshow("Virtual White Board", virtualWhiteBoard)

	# If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()