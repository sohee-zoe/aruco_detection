import cv2

# 네트워크 설정
SERVER_IP = "192.168.0.155"
CLIENT_IP = "0.0.0.0"
PORT = 5000

# UDP 설정
CHUNK_SIZE = 1400  # MTU 고려
SERVER_SEND_BUFFER = 65536
CLIENT_RECV_BUFFER = 262144

# UDP 카메라 설정
UDP_CAMERA_INDEX = 0
UDP_CALIBRATION_FILE = "camera_params/jetcobot.yaml"

# USB 카메라 설정
USB_CAMERA_INDEX = 2
USB_CALIBRATION_FILE = "camera_params/global.yaml"

# 카메라 공통 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30  # 목표 FPS
CAMERA_BUFFERSIZE = 1
JPEG_QUALITY = 80

DEFAULT_CAMERA_INDEX = 0
DEFAULT_CALIBRATION_FILE = "camera_params/calibration.yaml"

# Aruco 마커 설정
ARUCO_DICT_TYPE = "DICT_6X6_250"
ARUCO_MARKER_LENGTH = 0.06  # ArUco 마커 실제 크기 (미터 단위)
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
