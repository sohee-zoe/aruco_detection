import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import config
from udp_receiver import UdpReceiver
from image_processor import decode_frame, undistort_frame, detect_aruco, display_frame
from calibration_utils import load_calibration_from_yaml


def run_udp_client(use_calibration, calibration_file, aruco_type, marker_length):
    """UDP 스트림을 수신하고 처리하는 클라이언트를 실행합니다."""
    K, D = None, None
    new_K = None  # 왜곡 보정 후 사용할 K 값

    if use_calibration:
        try:
            K, D = load_calibration_from_yaml(calibration_file)
            new_K = K.copy()  # 초기값은 원본 K
            print(f"카메라 캘리브레이션 로드 완료: {calibration_file}")
        except FileNotFoundError:
            print(
                f"[경고] 캘리브레이션 파일({calibration_file})을 찾을 수 없습니다. 캘리브레이션 없이 진행합니다."
            )
            K, D = None, None
        except Exception as e:
            print(f"[오류] 캘리브레이션 파일 로드 실패: {e}")
            K, D = None, None

    try:
        receiver = UdpReceiver(
            config.CLIENT_IP,  # "" 또는 "0.0.0.0" 사용 가능
            config.PORT,
            config.CLIENT_RECV_BUFFER,
        )
    except IOError as e:
        print(f"UDP 수신기 초기화 오류: {e}")
        return

    last_frame = None
    frame_count = 0
    start_time = time.time()
    fps = 0

    print("UDP 클라이언트 시작. 스트림 수신 대기 중...")
    print("종료: 'q', 저장: 's'")

    try:
        while True:
            frame_data = receiver.receive_frame_data()

            processed_frame = None
            if frame_data:
                # 데이터 디코딩
                frame = decode_frame(frame_data)
                if frame is not None:
                    last_frame = frame.copy()  # 성공적으로 디코딩된 마지막 프레임 저장

                    # 왜곡 보정 (캘리브레이션 사용 시)
                    if K is not None and D is not None:
                        processed_frame, temp_K = undistort_frame(last_frame, K, D)
                        # 왜곡 보정 함수가 새 K를 반환하면 업데이트
                        if temp_K is not None:
                            new_K = temp_K
                    else:
                        processed_frame = last_frame
                        new_K = K  # 캘리브레이션 없으면 new_K도 None 또는 원본 K

                    # ArUco 마커 감지 및 정보 표시
                    # 왜곡 보정된 프레임과 그에 맞는 K(new_K) 사용
                    processed_frame, detected_info = detect_aruco(
                        processed_frame, new_K, D, aruco_type, marker_length
                    )
                    # print(detected_info)  # 감지된 정보 출력 (필요시)

                    # FPS 계산 및 표시
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:  # 1초마다 FPS 갱신
                        fps = frame_count / elapsed
                        frame_count = 0
                        start_time = time.time()

                    # FPS 정보 프레임에 추가
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    # 화면 표시 및 사용자 입력 처리
                    result = display_frame(processed_frame, "UDP Stream Client")
                    if result == "quit":
                        break
            else:
                # 데이터 수신 실패 또는 타임아웃 시 마지막 프레임 표시 (선택적)
                if last_frame is not None:
                    # 마지막 프레임에도 FPS 표시 유지
                    cv2.putText(
                        last_frame,
                        f"FPS: {fps:.2f} (Frozen)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                    result = display_frame(last_frame, "UDP Stream Client (Frozen)")
                    if result == "quit":
                        break
                else:
                    # 아직 받은 프레임이 없으면 잠시 대기
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl+C 감지. 클라이언트 종료 중...")
    except Exception as e:
        print(f"\n[오류] 클라이언트 실행 중 예외 발생: {e}")
    finally:
        print("리소스 정리 중...")
        receiver.close()
        cv2.destroyAllWindows()
        print("클라이언트 종료 완료.")


# USB 카메라 처리 함수 (기존 steream_camera2.py 내용 활용)
def run_usb_camera(
    camera_index, use_calibration, calibration_file, aruco_type, marker_length
):
    """USB 카메라 입력을 처리하고 표시합니다."""
    K, D = None, None
    new_K = None
    if use_calibration:
        try:
            K, D = load_calibration_from_yaml(calibration_file)
            new_K = K.copy()
            print(f"카메라 캘리브레이션 로드 완료: {calibration_file}")
        except Exception as e:
            print(f"[경고] 캘리브레이션 로드 실패: {e}")
            K, D = None, None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[오류] USB 카메라 인덱스 {camera_index}를 열 수 없습니다.")
        return

    # 자동 초점 끄기 시도 (선택적)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print(f"USB 카메라 스트리밍 시작 (인덱스: {camera_index}). 종료: 'q', 저장: 's'")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[경고] USB 카메라 프레임 읽기 실패")
                time.sleep(0.1)
                continue

            processed_frame = frame.copy()

            # 왜곡 보정
            if K is not None and D is not None:
                processed_frame, temp_K = undistort_frame(processed_frame, K, D)
                if temp_K is not None:
                    new_K = temp_K
            else:
                new_K = K

            # ArUco 감지
            processed_frame, detected_info = detect_aruco(
                processed_frame, new_K, D, aruco_type, marker_length
            )
            # print(detected_info)

            # 화면 표시
            result = display_frame(processed_frame, "USB Camera Feed")
            if result == "quit":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("USB 카메라 스트림 종료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="카메라 클라이언트 실행 (UDP 또는 USB)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["udp", "usb"],
        default="udp",
        help="영상 입력 소스 선택 (기본값: udp)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        # default=config.CAMERA_INDEX,
        # help=f"USB 카메라 사용 시 인덱스 (기본값: {config.CAMERA_INDEX})",
        help=f"USB 카메라 사용 시 인덱스",
    )
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="카메라 캘리브레이션 적용 여부 (기본값: 적용)",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        # default=config.CALIBRATION_FILE,
        # help=f"카메라 캘리브레이션 파일 경로 (기본값: {config.CALIBRATION_FILE})",
        help=f"카메라 캘리브레이션 파일 경로",
    )
    parser.add_argument(
        "--aruco_type",
        type=str,
        default=config.ARUCO_DICT_TYPE,
        choices=config.ARUCO_DICT.keys(),
        help=f"감지할 ArUco 마커 타입 (기본값: {config.ARUCO_DICT_TYPE})",
    )
    parser.add_argument(
        "--aruco_length",
        type=float,
        default=config.ARUCO_MARKER_LENGTH,
        help=f"ArUco 마커 실제 크기(미터) (기본값: {config.ARUCO_MARKER_LENGTH})",
    )
    args = parser.parse_args()

    camera_index = args.camera_index
    calibration_file = args.calibration_file

    if calibration_file is None:
        if args.source == "udp":
            calibration_file = config.UDP_CALIBRATION_FILE
        elif args.source == "usb":
            calibration_file = config.USB_CALIBRATION_FILE

    if args.source == "usb" and camera_index is None:
        camera_index = config.USB_CAMERA_INDEX

    if args.source == "udp":
        run_udp_client(
            args.calibration, calibration_file, args.aruco_type, args.aruco_length
        )
    elif args.source == "usb":
        run_usb_camera(
            camera_index,
            args.calibration,
            calibration_file,
            args.aruco_type,
            args.aruco_length,
        )
