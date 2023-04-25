import argparse
import cv2
import time
import numpy as np
import vgamepad as vg
from PIL import Image
from collections import deque
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pykalman import KalmanFilter

def get_center_coordinates(obj):
    x_center = (obj.bbox.xmin + obj.bbox.xmax) / 2
    y_center = (obj.bbox.ymin + obj.bbox.ymax) / 2
    return (x_center, y_center)

def get_relative_coordinates(center, obj):
    x_relative = (center[0] - obj.bbox.xmin) / (obj.bbox.xmax - obj.bbox.xmin)
    y_relative = (center[1] - obj.bbox.ymin) / (obj.bbox.ymax - obj.bbox.ymin)
    return (x_relative, y_relative)

class LightGun:
    def __init__(self, videodevice=0, width=640, height=480, filter_len=5):
        self.cap = cv2.VideoCapture(videodevice)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.measurements_x = deque(maxlen=filter_len)
        self.measurements_x.append(0.5)
        self.measurements_y = deque(maxlen=filter_len)
        self.measurements_y.append(0.5)

        initial_state_mean = [self.measurements_x[0], self.measurements_y[0]]
        initial_state_covariance = [[1, 0], [0, 1]]
        transition_matrix = [[1, 0], [0, 1]]
        observation_matrix = [[1, 0], [0, 1]]
        observation_covariance = [[1, 0], [0, 1]]
        process_covariance = [[1, 0], [0, 1]]

        self.kf = KalmanFilter(transition_matrices=transition_matrix,
                               observation_matrices=observation_matrix,
                               initial_state_mean=initial_state_mean,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=process_covariance)

    def getFrame(self):
        ret, frame = self.cap.read()
        return frame

    def getPosition(self, frame, frame_center, highest_scoring_obj):
        relative_coordinates = get_relative_coordinates(frame_center, highest_scoring_obj)
        self.measurements_x.append(relative_coordinates[0])
        self.measurements_y.append(relative_coordinates[1])
        smoothed_coords, _ = self.kf.smooth(np.column_stack((self.measurements_x, self.measurements_y)))
        filtered_coords = smoothed_coords[-1] * 2.0
        return filtered_coords

    def release(self):
        self.cap.release()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
    parser.add_argument('-v', '--videodevice', type=int, default=0, help='Video device index')
    parser.add_argument('-w', '--width', type=int, default=640, help='Video capture width')
    parser.add_argument('-ht', '--height', type=int, default=480, help='Video capture height')
    parser.add_argument('--filter_len', type=int, default=5, help='Smoothing filter length')
    parser.add_argument('--object', default='tv', help='Label to use for detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for image output and drawing')
    parser.add_argument('--duration', type=float, default=0.1, help='Duration of mouse movement')
    args = parser.parse_args()

    labels = read_label_file(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    light_gun = LightGun(videodevice=args.videodevice, width=args.width, height=args.height, filter_len=args.filter_len)

    gamepad = vg.VX360Gamepad()

    while True:
        frame = light_gun.getFrame()
        if frame is None:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

        interpreter.invoke()
        objs = detect.get_objects(interpreter, args.threshold, scale)

        tv_objs = [obj for obj in objs if labels[obj.id] == args.object]
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        if args.debug:
            cv2.drawMarker(frame, frame_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 2) # Mark centre of frame

        if tv_objs:
            highest_scoring_obj = max(tv_objs, key=lambda tv: tv.score)
            filtered_coords = light_gun.getPosition(frame, frame_center, highest_scoring_obj)

            if args.debug:
                cv2.rectangle(frame, (highest_scoring_obj.bbox.xmin, highest_scoring_obj.bbox.ymin),
                              (highest_scoring_obj.bbox.xmax, highest_scoring_obj.bbox.ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"({filtered_coords[0]:.2f}, {filtered_coords[1]:.2f})",
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            gamepad.left_joystick_float(x_value_float=filtered_coords[0] - 1.0, y_value_float=1.0 - filtered_coords[1])
            gamepad.update()

        if args.debug:
            cv2.imshow('Monitor Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    light_gun.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

