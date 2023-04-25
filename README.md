# WebCam LightGun

Uses a webcam with a coral AI edgeTPU to create a lightgun.

## Dependencies

### Hardware Required
* Webcam
* Coral.ai EdgeTPU

### Software
* Python 3.9
* pycoral
* opencv-python
* pyautogui

### Models
From https://coral.ai/models/object-detection/ get an object detection model that can label TVs or monitors. Anything trained on the COCO dataset should work.

I used SSD MobileNet V2 from [tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite](https://raw.githubusercontent.com/google-coral/test_data/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite) and label file [COCO labels](https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt).

## Running

`python .\lg.py --model .\model\tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite --label .\model\coco_labels.txt`

You will need to map something to click, rightclick or other buttons. You probably want the RawInput mode in game.