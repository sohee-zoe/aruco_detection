# run_client.py
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

import config
from udp_receiver import UdpReceiver
from image_processor import decode_frame, undistort_frame, detect_aruco, display_frame
from calibration_utils import load_calibration_from_yaml

def run_udp_client(use_calibration, calibration_file, detect_aruco_flag, aruco_type, marker_length):
    """UDP 스트림을 수신하고 처리하는 클라이언트를 실행합니다."""

    K, D = None, None
    new_K = None # 왜곡 보정 후 사용할 K 값

    if use_calibration:
        try:
            K, D = load_calibration_from_yaml(calibration_file)
            new_K = K.copy() # 초기값은 원본 K
            print(f"카메라 캘리브레이션 로드 완료: {calibration_file}")
        except FileNotFoundError:
            print(f"[경고] 캘리브레이션 파일({calibration_file})을 찾을 수 없습니다. 캘리브레이션 없이 진행합니다.")
            K, D = None, None
        except Exception as e:
            print(f"[오류] 캘리브레이션 파일 로드 실패: {e}")
            K, D = None, None

    try:
        receiver = UdpReceiver(
            config.CLIENT_IP,
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
            detected_info = [] # ArUco 정보 초기화

            if frame_data:
                # 데이터 디코딩
                frame = decode_frame(frame_data)
                if frame is not None:
                    last_frame = frame.copy() # 성공적으로 디코딩된 마지막 프레임 저장

                    # 왜곡 보정 (캘리브레이션 사용 시)
                    if use_calibration and K is not None and D is not None:
                        processed_frame, temp_K = undistort_frame(last_frame, K, D)
                        # 왜곡 보정 함수가 새 K를 반환하면 업데이트
                        if temp_K is not None:
                            new_K = temp_K
                    else:
                        processed_frame = last_frame
                        new_K = K # 캘리브레이션 없으면 new_K도 None 또는 원본 K

                    # --- ArUco 마커 감지 (플래그 확인) ---
                    if detect_aruco_flag:
                        # 왜곡 보정된 프레임과 그에 맞는 K(new_K) 사용
                        processed_frame, detected_info = detect_aruco(
                            processed_frame, new_K, D, aruco_type, marker_length
                        )
                        # if detected_info: # 감지된 정보가 있을 때만 출력
                        #    print(detected_info)

                    # --- ArUco 감지 끝 ---

                    # FPS 계산 및 표시
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0: # 1초마다 FPS 갱신
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
                else: # 프레임 디코딩 실패 시
                    # print("[경고] 프레임 디코딩 실패")
                    time.sleep(0.01) # 짧은 대기

            else: # 데이터 수신 실패 또는 타임아웃 시
                # 타임아웃 시에는 별도 처리 없이 계속 진행하거나, 마지막 프레임 표시 가능
                # if last_frame is not None:
                #     ... (last_frame 표시 로직) ...
                time.sleep(0.01) # CPU 사용량 줄이기 위한 짧은 대기

    except KeyboardInterrupt:
        print("\nCtrl+C 감지. 클라이언트 종료 중...")
    except Exception as e:
        print(f"\n[오류] 클라이언트 실행 중 예외 발생: {e}")
    finally:
        print("리소스 정리 중...")
        receiver.close()
        cv2.destroyAllWindows()
        print("클라이언트 종료 완료.")


def run_usb_camera(
    camera_index, use_calibration, calibration_file, detect_aruco_flag, aruco_type, marker_length
):
    """USB 카메라 입력을 처리하고 표시합니다."""
    K, D = None, None
    new_K = None

    if use_calibration:
        try:
            K, D = load_calibration_from_yaml(calibration_file)
            new_K = K.copy()
            print(f"카메라 캘리브레이션 로드 완료: {calibration_file}")
        except FileNotFoundError:
             print(f"[경고] 캘리브레이션 파일({calibration_file})을 찾을 수 없습니다. 캘리브레이션 없이 진행합니다.")
             K, D = None, None
        except Exception as e:
            print(f"[경고] 캘리브레이션 로드 실패: {e}")
            K, D = None, None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[오류] USB 카메라 인덱스 {camera_index}를 열 수 없습니다.")
        return

    # 카메라 설정 (해상도 등)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
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
            detected_info = [] # ArUco 정보 초기화

            # 왜곡 보정
            if use_calibration and K is not None and D is not None:
                processed_frame, temp_K = undistort_frame(processed_frame, K, D)
                if temp_K is not None:
                    new_K = temp_K
            else:
                 new_K = K # 왜곡 보정 안 할 시 원본 K (또는 None) 사용

            # --- ArUco 감지 (플래그 확인) ---
            if detect_aruco_flag:
                processed_frame, detected_info = detect_aruco(
                    processed_frame, new_K, D, aruco_type, marker_length
                )
                # if detected_info: # 감지된 정보가 있을 때만 출력 (선택적)
                #     print(detected_info)
            # --- ArUco 감지 끝 ---

            # 화면 표시
            result = display_frame(processed_frame, f"USB Camera Feed (Index: {camera_index})")
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
        help=f"USB 카메라 사용 시 인덱스 (지정하지 않으면 config.py의 USB_CAMERA_INDEX 사용)",
    )
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="카메라 캘리브레이션 적용 여부 (기본값: 적용, 비활성화: --no-calibration)",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        help="카메라 캘리브레이션 파일 경로 (지정하지 않으면 소스에 따라 config.py의 기본값 사용)",
    )

    parser.add_argument(
        "--detect_aruco",
        action=argparse.BooleanOptionalAction,
        default=True, # 기본적으로 ArUco 감지 활성화
        help="ArUco 마커 감지 활성화 여부 (기본값: 활성화, 비활성화: --no-detect-aruco)",
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

    if args.source == "usb":
        if camera_index is None:
            camera_index = config.USB_CAMERA_INDEX
            print(f"USB 카메라 인덱스가 지정되지 않아 config.py의 값({camera_index})을 사용합니다.")
        if calibration_file is None:
            calibration_file = config.USB_CALIBRATION_FILE
            print(f"USB 캘리브레이션 파일이 지정되지 않아 config.py의 값({calibration_file})을 사용합니다.")
    elif args.source == "udp":
        if calibration_file is None:
            calibration_file = config.UDP_CALIBRATION_FILE
            print(f"UDP 캘리브레이션 파일이 지정되지 않아 config.py의 값({calibration_file})을 사용합니다.")

    if args.source == "udp":
        run_udp_client(
            args.calibration,
            calibration_file,
            args.detect_aruco,
            args.aruco_type,
            args.aruco_length,
        )
    elif args.source == "usb":
         # USB는 카메라 인덱스 필수
        if camera_index is None:
            print("[오류] USB 소스 선택 시 --camera_index를 지정해야 합니다.")
        else:
            run_usb_camera(
                camera_index,
                args.calibration,
                calibration_file,
                args.detect_aruco,
                args.aruco_type,
                args.aruco_length,
            )

