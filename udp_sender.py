import socket
import cv2
import time
import numpy as np

class UdpSender:
    def __init__(self, host_ip, port, chunk_size, buffer_size, reconnect_delay=2):
        self.target_ip = host_ip
        self.port = port
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay # 재연결 시도 간격 (초)
        self.sock = None # 초기에는 None으로 설정
        self.frame_seq = 0
        self._create_socket() # 초기 소켓 생성 시도

    def _create_socket(self):
        """소켓을 생성하고 설정합니다."""
        if self.sock: # 기존 소켓이 있다면 닫기 시도
            try:
                self.sock.close()
            except Exception:
                pass # 이미 닫혔거나 오류 상태일 수 있음

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.buffer_size)
            print(f"UDP Sender 소켓 생성/재생성 완료: 대상 {self.target_ip}:{self.port}, 청크 {self.chunk_size}")
            return True
        except socket.error as e:
            print(f"[오류] UDP 소켓 생성 실패: {e}")
            self.sock = None # 실패 시 None으로 설정
            return False
        except OSError as e: # 버퍼 크기 설정 실패 시
            print(f"[경고] 송신 버퍼 크기 설정 실패 ({self.buffer_size}): {e}. 기본값 사용.")
            # 버퍼 설정 실패해도 소켓 자체는 생성될 수 있으므로 True 반환 가능
            if self.sock:
                print(f"UDP Sender 소켓 생성/재생성 완료 (버퍼 설정 경고): 대상 {self.target_ip}:{self.port}")
                return True
            else: # 소켓 생성 자체가 실패한 경우
                 print(f"[오류] UDP 소켓 생성 실패 (OS 에러): {e}")
                 return False


    def send_frame(self, frame, quality):
        """프레임을 압축하고 청크로 나누어 UDP로 전송합니다. 실패 시 재연결을 시도합니다."""
        if frame is None:
            return False # 프레임 없음

        if not self.sock: # 소켓이 유효하지 않으면 재연결 시도
            print("[정보] 소켓이 유효하지 않아 재연결 시도 중...")
            if not self._create_socket():
                print("[오류] 소켓 재생성 실패. 전송 건너뜀.")
                time.sleep(self.reconnect_delay) # 잠시 후 다시 시도하도록 대기
                return False
            # 소켓 재생성 성공 시 계속 진행

        # 이미지 압축
        ret, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            print("[오류] 이미지 인코딩 실패")
            return False

        img_bytes = img_encoded.tobytes()
        data_len = len(img_bytes)

        # 프레임 시퀀스 번호 증가
        self.frame_seq = (self.frame_seq + 1) % 65536

        try:
            # 프레임 헤더 정보 전송
            header = self.frame_seq.to_bytes(2, byteorder="big") + data_len.to_bytes(4, byteorder="big")
            self.sock.sendto(header, (self.target_ip, self.port))

            # 데이터를 청크로 나누어 전송
            chunk_id = 0
            for i in range(0, data_len, self.chunk_size):
                chunk = img_bytes[i : i + self.chunk_size]
                chunk_header = self.frame_seq.to_bytes(2, byteorder="big") + chunk_id.to_bytes(2, byteorder="big")
                self.sock.sendto(chunk_header + chunk, (self.target_ip, self.port))
                chunk_id += 1
                # time.sleep(0.0001) # 선택적 딜레이

            # 프레임 종료 신호 전송
            end_signal = self.frame_seq.to_bytes(2, byteorder="big") + b"END"
            self.sock.sendto(end_signal, (self.target_ip, self.port))
            return True # 전송 성공

        except socket.error as e:
            print(f"[오류] UDP 전송 중 오류 발생: {e}. 재연결 시도...")
            # 오류 발생 시 소켓을 닫고 재생성 시도
            if not self._create_socket():
                 print("[오류] 소켓 재생성 실패. 다음 프레임에서 재시도.")
                 time.sleep(self.reconnect_delay) # 실패 시 잠시 대기
            # 이번 프레임 전송은 실패로 처리
            return False
        except Exception as e: # 다른 예외 처리
            print(f"[오류] 예기치 않은 오류 발생 (send_frame): {e}")
            # 이 경우에도 소켓 재연결 시도 가능
            if not self._create_socket():
                 print("[오류] 소켓 재생성 실패.")
                 time.sleep(self.reconnect_delay)
            return False

    def close(self):
        """UDP 소켓을 닫습니다."""
        if self.sock:
            try:
                self.sock.close()
                print("UDP Sender 소켓 닫힘.")
            except Exception as e:
                print(f"[경고] 소켓 닫기 중 오류 발생: {e}")
            finally:
                self.sock = None # 소켓 참조 제거
