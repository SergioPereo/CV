import cv2
import numpy as np
import os
import argparse
import math
from operator import xor

clear = lambda: os.system('cls')

class Scaler:
    angle = 0.0
    def __init__(self, min, max, a, b):
        self.min = min
        self.max = max
        self.a = a
        self.b = b
        print("Min: " + str(self.min))
        print("Max: " + str(self.max))
        print("A: " + str(self.a))
        print("B: " + str(self.b))
    def scaleX(self, x):
        angle = (((self.b-self.a)*(x-self.min))/(self.max-self.min))+self.a
        return angle

leftScaler = Scaler(min=0.0, max=160.0, a=-21.80140949, b=0)
rightScaler = Scaler(min=160.0, max=320.0, a=0, b=21.80140949)



def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)
    cv2.createTrackbar("MIN_CANNY", "Trackbars", 0, 500, callback)
    cv2.createTrackbar("MAX_CANNY", "Trackbars", 280, 500, callback)

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range Filter. RGB or HSV')
    ap.add_argument('-hsv', '--hsvvalues', required=False,
                    help='Filter Values')
    ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-cvinf', '--cvinformation', required=False,
                    help='Use if you want to apply the contours, centroids, and area',
                    action='store_true')
    ap.add_argument('-wi', '--webcamindex', required=False,
                    help='Webcam Index')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    ap.add_argument('-cs', '--camerasetsettings', required=False,
                    help='Set camera settings',
                    action='store_true')
    ap.add_argument('-cg', '--cameragetsettings', required=False,
                    help='Get camera settings',
                    action='store_true')
    args = vars(ap.parse_args())

    if not xor(bool(args['image']), bool(args['webcam'])):
        ap.error("Please specify only one image source")

    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please speciy a correct filter.")

    return args

def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    v = cv2.getTrackbarPos("MIN_CANNY", "Trackbars")
    values.append(v)
    v = cv2.getTrackbarPos("MAX_CANNY", "Trackbars")
    values.append(v)
    return values

def labelCenter(image, c):
    # Places a red circle on the centers of contours
    cx = 0
    M = cv2.moments(c)
    if M['m00'] is not 0 and M['m10'] is not 0 and M['m01']:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image,(cx,cy), 3, (0,0,255), -1)
    # Draw the countour number on the image
    return image, cx

def contoursFinding(image, canny_min_thresh, canny_max_thresh):

    #Grayscaling that original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Finding canny edges
    edged = cv2.Canny(gray, canny_min_thresh, canny_max_thresh)

    #Find the contours
    im2, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Returning the contours

    return contours

def getCenterCentroid(c1x, c2x):
    minX = min(c1x,c2x)
    maxX = max(c1x,c2x)
    distance = maxX - minX
    center = distance/2 + minX
    return center

def labelingCentroidsAndAreas(image, contours):
    sortedContours =  sorted(contours, key=cv2.contourArea, reverse=True)
    cxs = []
    center = 0;
    for (i, c) in enumerate(sortedContours):
        if i is 0 or i is 1:
            blank, cx = labelCenter(image, c)
            area = cv2.contourArea(c)
            cv2.putText(image, "Area: " + str(area), (image.shape[0]-80, image.shape[1]-100 + i*15)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cxs.append(cx)

    if len(cxs) >=  2:
            center = getCenterCentroid(cxs[0], cxs[1])
            center = int(center)
            angle = 0
            cv2.circle(image,(center,200), 3, (0,0,255), -1)
            cv2.putText(image, "Center X:  " + str(center), (image.shape[0]-250, image.shape[1]-100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            if center >= 160:
                angle = rightScaler.scaleX(x=center)
            else:
                angle = leftScaler.scaleX(x=center)
            cv2.putText(image, "Angle: " + str(angle), (image.shape[0]-320, image.shape[1]-85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    #dilation = cv2.dilate(blank_image, kernel, iterations = 3)
    cv2.imshow('CV Information', image)

def main():
    args = get_arguments()
    range_filter = args['filter'].upper()
    if args['image']:
        image = cv2.imread(args['image'])

        if range_filter == 'RGB':
            frame_to_thresh = image.copy()
        else:
            frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        if args['webcamindex']:
            int_webcam_index = int(args['webcamindex'])
            camera = cv2.VideoCapture(int_webcam_index)
            camera.set(3, 320)
            camera.set(4, 240)
            camera.set(5, 15)
            if args['camerasetsettings']:
                camera.set(11, 2)
                camera.set(12, 120)
                camera.set(13, 1)
                camera.set(15, -7)

        else:
            camera = cv2.VideoCapture(0)
            camera.set(3, 320)
            camera.set(4, 240)
            camera.set(5, 15)

            if args['camerasetsettings']:
                camera.set(11, 2)
                camera.set(12, 120)
                camera.set(13, 1)
                camera.set(15, -7)
    if args['hsvvalues'] is None:
        setup_trackbars(range_filter)

    while True:
        if args['webcam']:
            ret, image = camera.read()


            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if args['hsvvalues']:
            arrvalues = args['hsvvalues'].split(',')
            v1_min = arrvalues[0]
            v2_min = arrvalues[1]
            v3_min = arrvalues[2]
            v1_max = arrvalues[3]
            v2_max = arrvalues[4]
            v3_max = arrvalues[5]
            min_canny = arrvalues[6]
            max_canny = arrvalues[7]
            min_canny = int(min_canny)
            max_canny = int(max_canny)

            mask = cv2.inRange(frame_to_thresh, (int(v1_min), int(v2_min), int(v3_min)), (int(v1_max), int(v2_max), int(v3_max)))
        else:
            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max, min_canny, max_canny = get_trackbar_values(range_filter)
            mask = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        blank_image_preview = np.zeros((image.copy().shape[0], image.shape[1], 3))
        blank_image_cvinfo = np.zeros((image.copy().shape[0]+100, image.shape[1]+100, 3))
        if args['preview']:
            preview = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("Preview", preview)
        else:
            image = cv2.flip(image, 1)
            cv2.imshow("Original", image)

            thresh = cv2.flip(thresh, 1)
            cv2.imshow("Thresh", thresh)

        if args['cvinformation']:
            filtered_img = cv2.bitwise_and(image, image, mask=mask)
            contours = contoursFinding(filtered_img, min_canny, max_canny)
            labelingCentroidsAndAreas(blank_image_cvinfo, contours)
            if args['cameragetsettings']:
                print ( " CV_CAP_PROP_FORMAT:  "+str(camera.get(9)) + "" )
                print ( " CV_CAP_PROP_MODE:  "+str(camera.get(10)) + "" )
                print ( " CV_CAP_PROP_BRIGHTNESS:  "+str(camera.get(11)) + "" )
                print ( " CV_CAP_PROP_CONTRAST:  "+str(camera.get(12)) + "" )
                print ( " CV_CAP_PROP_SATURATION:  "+str(camera.get(13)) + "" )
                print ( " CV_CAP_PROP_HUE:  "+str(camera.get(14)) + "" )
                print ( " CV_CAP_PROP_GAIN:  "+str(camera.get(15)) + "" )
                print ( " CV_CAP_PROP_EXPOSURE:  "+str(camera.get(16)) + "" )
        if cv2.waitKey(1) == 13:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
