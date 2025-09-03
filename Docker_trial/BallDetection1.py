import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def frame_capture(camIdx):
    # video capture

    cam = cv2.VideoCapture(camIdx)

    # read frame
    bol, frame = cam.read()

    if bol == False:
        print("Frame not available.")
    else:
        return frame


def convertColr(fr, clrSpace):
    if clrSpace == "hsv":
        frm = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    elif clrSpace == "rgb":
        frm = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    elif clrSpace == "gray":
        frm = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return frm


def cntr_outlne_boundbox(fr, msk):
    c, hir = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(c))
    if len(c) == 0:
        print("Ball not found")
        cv2.imshow("Bounding Rectangle", fr)
    else:
        framecpy = fr.copy()
        for i, ci in enumerate(c):
            area = cv2.contourArea(ci)
            # print(area)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(ci)
                framecpy = cv2.rectangle(
                    framecpy, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                framecpy = cv2.drawContours(framecpy, [ci], 0, (0, 255, 255), 2)
            else:
                framecpy = fr
        cv2.imshow("Bounding Rectangle", framecpy)


with open("/home/dhvani/Subhiksha/Opencv/Docker_trial/Color Range.json", "r") as f:
    clrRang = json.load(f)


# rectified
cam = cv2.VideoCapture(2)

balCord = {}  # key idx (ith) ball - x,y,w,h of ball
frameballCord = {}  # key idx (ith) ball - frame after drawing rectangle

font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2


# read frame
while True:
    allCntrs = []
    bol, frame = cam.read()
    framecpy = frame.copy()

    # convert frame to hsv
    hsvframe = cv2.cvtColor(framecpy, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsvframe)

    # frame not available
    if bol == False:
        print("Video not available")

    # on key press q break
    if cv2.waitKey(1) == ord("q"):
        break

    # cv2.imwrite("balls.png",frame)
    # break
    output_file = "ball_positions.txt"  # File to write the output

    with open(output_file, "a") as file:
        # iterate color ball ranges
        for i, k in enumerate(clrRang["balls"]):
            start, stop = clrRang["balls"][k]

            # print(start, stop)
            # apply thresholding for converted hsv frame
            mask = cv2.inRange(hsvframe, np.array(start), np.array(stop)) # type: ignore
            # cv2.imshow("mask"+str(i),mask)

            # passing mask get contours
            c, hir = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # when no contour available
            if len(c) != 0:

                # Countour area for each contour
                for j, ci in enumerate(c):
                    area = cv2.contourArea(ci)

                    # draw bounding box of contour if greater than thresholded area
                    if area > 10000:
                        x, y, w, h = cv2.boundingRect(ci)

                        # positions of each balls
                        if i not in balCord:
                            balCord[i] = []
                            balCord[i].append([x, y, w, h])

                        # Write k, x, y, w, h to the file
                        file.write(f"{k} {x} {y} {w} {h}\n")

                        # draw rectangle border for each ball
                        framecpy = cv2.rectangle(
                            framecpy, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )

                        # frame dict with rectangle border drawn
                        image = cv2.putText(
                            framecpy,
                            k,
                            (x, y),
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )

                        if i not in frameballCord:
                            frameballCord[i] = []
                            frameballCord[i].append(framecpy)

                        allCntrs.append(ci)
                        # draw contours on frame with rectangle border drawn
                        # framecpy = cv2.drawContours(
                        #     framecpy, [ci], 0, (0, 255, 255), 2
                        # )

                    # when area does not meet threshold criteria
                    else:
                        framecpy = frame

            # when no contour available
            else:
                framecpy = frame

    # show the frame
    framecpy = cv2.drawContours(framecpy, allCntrs, 0, (0, 255, 255), 2)
    cv2.imshow("bounding", framecpy)


cam.release()
cv2.destroyAllWindows()
