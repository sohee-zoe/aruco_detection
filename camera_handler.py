# camera_handler.py
import cv2
import time

class CameraHandler:
    def __init__(self, index, width, height, fps, buffer_size):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"카메라 인덱스 {index}를 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        print(f"카메라 초기화 완료: {width}x{height} @ {fps}FPS, 버퍼 {buffer_size}")

        # MyCobot 관련 코드는 여기서 제외 (run_server.py에서 처리)

    def capture_frame(self):
        """카메라에서 프레임을 캡처하여 반환합니다."""
        ret, frame = self.cap.read()
        if not ret:
            print("카메라에서 프레임을 읽는 데 실패했습니다.")
            time.sleep(0.1) # 잠시 대기
            return None
        return frame

    def release_camera(self):
        """카메라 장치를 해제합니다."""
        if self.cap.isOpened():
            self.cap.release()
            print("카메라 리소스 해제 완료.")

# 테스트용 (직접 실행 시)
if __name__ == '__main__':
    import config
    try:
        cam_handler = CameraHandler(
            config.DEFAULT_CAMERA_INDEX,
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
            config.FRAME_RATE,
            config.CAMERA_BUFFERSIZE
        )
        while True:
            frame = cam_handler.capture_frame()
            if frame is not None:
                cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except IOError as e:
        print(f"카메라 오류: {e}")
    finally:
        if 'cam_handler' in locals():
            cam_handler.release_camera()
        cv2.destroyAllWindows()
