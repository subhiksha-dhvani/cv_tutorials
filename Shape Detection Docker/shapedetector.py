import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#Capture Video 
cam = cv2.VideoCapture(2)

while True:
    bol, frame = cam.read()

    # frame not available
    if bol == False:
        print("Video not available")

    framecpy = frame.copy()

    # framecpy= framecpy[5:, 100:500]

    # press q to quit
    if cv2.waitKey(1) == ord("q"):
        break

    #save
    if cv2.waitKey(1) == ord("s"):
        cv2.imwrite("ImgCapture.png", frame)
        break

    #Countour -> Binary img with foreground 
    #
    grayframe = cv2.cvtColor(framecpy, cv2.COLOR_BGR2GRAY)

    ret, binimg = cv2.threshold(grayframe, 105, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Mask", binimg)

    cntr, hir = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(framecpy, cntr, -1, (0, 255, 255), 3)

    if len(cntr) != 0:
        for j, ci in enumerate(cntr):
            approx = cv2.approxPolyDP(ci, 0.02 * cv2.arcLength(ci, True), True)
            cv2.drawContours(framecpy, [approx], 0, (0, 255, 255), 3)
            vertices = len(approx)
            # print("No. of vertices ", vertices)
            if vertices > 2:
                if vertices == 3:
                    # print(approx)
                    cv2.putText(
                        framecpy,
                        "Triangle",
                        (approx.ravel()[0], approx.ravel()[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                    )
                elif vertices == 4:
                    x, y, w, h = cv2.boundingRect(ci)
                    ar = w / float(h)
                    # print("w/h: ",ar)
                    if 0.95 <= ar < 1.2:
                        # if ar == 1:
                        shape = "Square"
                    else:
                        shape = "Rectangle"
                    cv2.putText(
                        framecpy,
                        shape,
                        (approx.ravel()[0], approx.ravel()[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                    )

                elif vertices == 5:
                    cv2.putText(
                        framecpy,
                        "Pentagon",
                        (approx.ravel()[0], approx.ravel()[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                    )

                else:

                    ellipse = cv2.fitEllipse(approx)
                    if ellipse[1][0] > 0:
                        ar = ellipse[1][1] / ellipse[1][0]
                        # print("ar: ", ar)
                        if 0.95 <= ar < 1.1:
                            shapec = "circle"
                        else:
                            shapec = "ellipse"
                        cv2.putText(
                            framecpy,
                            shapec,
                            (approx.ravel()[0], approx.ravel()[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                        )

    else:
        framecpy = frame

    cv2.imshow("bounding", framecpy)


cam.release()
cv2.destroyAllWindows()
