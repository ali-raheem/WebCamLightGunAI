# Ali Raheem 2023
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

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
    parser.add_argument('-v', '--videodevice', type=int, default=0, help='Video device index for left joystick')
    parser.add_argument('-v2', '--videodevice2', type=int, default=1, help='Video device index for right joystick')
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

    cap = cv2.VideoCapture(args.videodevice)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cap2 = cv2.VideoCapture(args.videodevice2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    measurements_x = deque(maxlen=args.filter_len)
    measurements_x.append(0.5)
    measurements_y = deque(maxlen=args.filter_len)
    measurements_y.append(0.5)

    measurements_x2 = deque(maxlen=args.filter_len)
    measurements_x2.append(0.5)
    measurements_y2 = deque(maxlen=args.filter_len)
    measurements_y2.append(0.5)

    initial_state_mean = [measurements_x[0], measurements_y[0]]
    initial_state_covariance = [[1, 0], [0, 1]]
    transition_matrix = [[1, 0], [0, 1]]
    observation_matrix = [[1, 0], [0, 1]]
    observation_covariance = [[1, 0], [0, 1]]
    process_covariance = [[1, 0], [0, 1]]

    kf = KalmanFilter(transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=process_covariance)

    gamepad = vg.VX360Gamepad()
    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if not ret or not ret2:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

        image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        _, scale2 = common.set_resized_input(interpreter, image2.size, lambda size: image2.resize(size, Image.LANCZOS))

        interpreter.invoke()
        objs = detect.get_objects(interpreter, args.threshold, scale)
        interpreter.invoke()
        objs2 = detect.get_objects(interpreter, args.threshold, scale2)

        tv_objs = [obj for obj in objs if labels[obj.id] == args.object]
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        tv_objs2 = [obj for obj in objs2 if labels[obj.id] == args.object]
        frame_center2 = (frame2.shape[1] // 2, frame2.shape[0] // 2)

        if args.debug:
            cv2.drawMarker(frame, frame_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 2) # Mark centre of frame
            cv2.drawMarker(frame2, frame_center2, (255, 0, 0), cv2.MARKER_CROSS, 20, 2) # Mark centre of frame2

        if tv_objs:
            highest_scoring_obj = max(tv_objs, key=lambda tv: tv.score)
            relative_coordinates = get_relative_coordinates(frame_center, highest_scoring_obj)
            measurements_x.append(relative_coordinates[0])
            measurements_y.append(relative_coordinates[1])
            smoothed_coords, _ = kf.smooth(np.column_stack((measurements_x, measurements_y)))
            filtered_coords = smoothed_coords[-1] * 2.0
            if args.debug:
                cv2.rectangle(frame, (highest_scoring_obj.bbox.xmin, highest_scoring_obj.bbox.ymin),
                              (highest_scoring_obj.bbox.xmax, highest_scoring_obj.bbox.ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"({relative_coordinates[0]:.2f}, {relative_coordinates[1]:.2f})",
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            gamepad.left_joystick_float(x_value_float=filtered_coords[0] - 1.0, y_value_float=1.0 - filtered_coords[1])
            gamepad.update()

        if tv_objs2:
            highest_scoring_obj2 = max(tv_objs2, key=lambda tv: tv.score)
            relative_coordinates2 = get_relative_coordinates(frame_center2, highest_scoring_obj2)
            measurements_x2.append(relative_coordinates2[0])
            measurements_y2.append(relative_coordinates2[1])
            smoothed_coords2, _ = kf.smooth(np.column_stack((measurements_x2, measurements_y2)))
            filtered_coords2 = smoothed_coords2[-1] * 2.0
            if args.debug:
                cv2.rectangle(frame2, (highest_scoring_obj2.bbox.xmin, highest_scoring_obj2.bbox.ymin),
                              (highest_scoring_obj2.bbox.xmax, highest_scoring_obj2.bbox.ymax), (0, 255, 0), 2)
                cv2.putText(frame2, f"({relative_coordinates2[0]:.2f}, {relative_coordinates2[1]:.2f})",
                            (10, frame2.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            gamepad.right_joystick_float(x_value_float=filtered_coords2[0] - 1.0, y_value_float=1.0 - filtered_coords2[1])
            gamepad.update()

        if args.debug:
            cv2.imshow('Monitor Detection', frame)
            cv2.imshow('Monitor Detection 2', frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
