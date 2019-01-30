# CV

## Table of contents

* [General Information](#general-information)
* [Technologies](#technologies)
* [Setup](#setup)

---

## General Information
Python code to personalize your mask of color you want to apply, and recognize the centroids of the largest contours.

---

## Technologies
Project is created with:
* Python: 3.7.1
* OpenCV: 3.4.5
* Numpy: 1.15.4
* Argparse: 1.1

---

## Setup
Clone this repository (you can use the next example line if you have git installed in your pc)
* `git clone https://github.com/SergioPereo/CV.git`

To work with the script use the following flags and example lines (Preferably in an Anaconda Prompt)

Command Line Example (In the file download directory):
`python TecbotCV.py --filter HSV --webcam --webcamindex 0 --preview --cvinformation`

The script have the following flags
* `-f` | `--filter` (required):
  Range Filter, it could be RGB or HSV. Example: --filter HSV

* `-i` | `--image` (non-required):
  Path to an image (if needed).
  Example: `--image "image_CV.jpg"`

* `-w` | `--webcam` (non-required):
  Use it if you want to use a webcam capture.
  Example: `--webcam`

* `-cvinf` | `--cvinformation` (non-required):
  Use it if you want to apply the contours, centroids and area calculations.
  Example: `--cvinformation`

* `-wi` | `--webcamindex` (non-required):
  Webcam index (0 at default).
  Example: `--webcamindex 0`

* `-p` | `--preview` (non-required):
  Show preview of the image after applying the mask.
  Example: `--preview`

* `-cs` | `--camerasetsettings` (non-required):
  Set some camera settings as default in the code (You can change them if you want).
  Example: `--camerasetsettings`

* `-cg` | `--cameragetsettings` (non-required):
  Get your actual camera settings.
  Example: `--cameragetsettings`
  * CV_CAP_PROP_FORMAT
  * CV_CAP_PROP_FORMAT
  * CV_CAP_PROP_MODE
  * CV_CAP_PROP_BRIGHTNESS
  * CV_CAP_PROP_CONTRAST
  * CV_CAP_PROP_SATURATION
  * CV_CAP_PROP_HUE
  * CV_CAP_PROP_GAIN
  * CV_CAP_PROP_EXPOSURE

* `-hsv` | `--hsvvalues` (non-required):
  To set the filter mask values. Example:
  `--hsvvalues 0,0,0,255,255,255,0,280`
  Min H Value, Min S Value, Min V Value, Max H Value, Max S Value, Max V Value, Min Canny Threshold Value, Max Canny Threshold Value
