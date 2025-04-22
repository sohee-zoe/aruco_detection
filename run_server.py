# run_server.py
import time
import config
from camera_handler import CameraHandler
from udp_sender import UdpSender

# import pymycobot # MyCobot 사용 시 주석 해제
# from packaging import version # MyCobot 사용 시 주석 해제

# MyCobot 버전 확인 및 클래스 임포트 (MyCobot 사용 시)
# if version.parse(pymycobot.__version__) >= version.parse("3.6.0"):
#     from pymycobot import MyCobot280 as MyCobot
# else:
#     from pymycobot import MyCobot


def main():
    # MyCobot 초기화 (필요시)
    # try:
    #     mycobot = MyCobot("/dev/ttyJETCOBOT", 1000000)
    #     print("MyCobot 초기화 완료")
    # except Exception as e:
    #     print(f"MyCobot 초기화 실패: {e}")
    #     mycobot = None # MyCobot 없이 진행

    try:
        cam_handler = CameraHandler(
            config.UDP_CAMERA_INDEX,
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
            config.FRAME_RATE,
            config.CAMERA_BUFFERSIZE,
        )
    except IOError as e:
        print(f"카메라 초기화 오류: {e}")
        return  # 카메라 없으면 종료

    sender = UdpSender(
        config.SERVER_IP,  # 여기서는 수신자 IP를 넣어야 함
        config.PORT,
        config.CHUNK_SIZE,
        config.SERVER_SEND_BUFFER,
    )

    last_send_time = time.time()
    target_interval = 1.0 / config.FRAME_RATE if config.FRAME_RATE > 0 else 0

    print(f"UDP 스트리밍 서버 시작. 대상: {config.SERVER_IP}:{config.PORT}")
    print("종료하려면 Ctrl+C를 누르세요.")

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - last_send_time

            # 목표 프레임 레이트 유지 시도
            if target_interval > 0:
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 새 프레임 캡처
            frame = cam_handler.capture_frame()
            if frame is None:
                continue  # 프레임 읽기 실패 시 다음 루프

            # 프레임 전송
            sender.send_frame(frame, config.JPEG_QUALITY)

            # 전송 시간 업데이트
            last_send_time = time.time()

    except KeyboardInterrupt:
        print("\nCtrl+C 감지. 서버 종료 중...")
    except Exception as e:
        print(f"\n[오류] 서버 실행 중 예외 발생: {e}")
    finally:
        print("리소스 정리 중...")
        cam_handler.release_camera()
        sender.close()
        # if mycobot: # MyCobot 사용 시 로봇 연결 해제 등 추가 가능
        #     pass
        print("서버 종료 완료.")


if __name__ == "__main__":
    main()
