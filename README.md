# WebCam LightGun

Uses a webcam with a coral AI edgeTPU to create a lightgun.

It uses an object detection model to identify the monitor in the view (selecting the highest confidence score object if multiple are detected). It then moves the mouse cursor to where the camera is pointed relative to the bounding box.

Currently it runs at about 3 actions per second (limited by opencv frame grab, monitor detection takes about 10ms).

## See it in action

[Youtube video showing some gameplay](https://youtu.be/7g3i7UJV5Zg), buttons pressed by aiming out of the screen with XYAB buttons depending if above, below etc the monitor.

![Screenshot showing cursor following webcam focus](screenshot.png)

![Screenshot showing virtual gamepad mode](screenshot-vgp.png)

Basic tips:
* Check the `--debug` image to see if anything in the view is confusing the model
* The webcam should be held as vertical as possible as so far the image isn't being rotated to compensate for oblique monitors
* A CPU model should be supported and files are available should be simple [TODO]

## Dependencies

### Hardware Required
* Webcam
* Coral.ai EdgeTPU

### Software
* Python 3.9
* pycoral
* opencv-python
#### Windows, Linux and MacOS
* pyautogui - lg.py
#### Windows Only Gamepad mode
* vgamepad (uses Nefarious gamepad emulator) - lg-vgamepad.py

### Models
From https://coral.ai/models/object-detection/ get an object detection model that can label TVs or monitors. Anything trained on the COCO dataset should work.

I used SSD MobileNet V2 from [tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite](https://raw.githubusercontent.com/google-coral/test_data/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite) and label file [COCO labels](https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt).

## Running

`python .\lg.py --model .\model\tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite --label .\model\coco_labels.txt`

You will need to map something to click, rightclick or other buttons. You probably want the RawInput mode in game.