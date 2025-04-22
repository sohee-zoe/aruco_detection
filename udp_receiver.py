import socket
import time
import numpy as np
from collections import defaultdict
import cv2

class UdpReceiver:
    def __init__(self, host_ip, port, buffer_size, timeout=0.5):
        self.host_ip = host_ip
        self.port = port
        self.buffer_size = buffer_size + 1024 # 헤더 포함 넉넉하게
        self.timeout = timeout
        self.max_buffer_age = 5.0
        self.sock = None # 초기값 None
        self.frame_buffers = defaultdict(lambda: {"data": bytearray(), "expected_size": 0, "chunks": {}, "received_time": 0})
        self._bind_socket() # 소켓 바인딩 시도

    def _bind_socket(self):
        """소켓을 생성하고 바인딩합니다."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size) # 수신 버퍼 설정 먼저 시도
            self.sock.bind((self.host_ip, self.port))
            self.sock.settimeout(self.timeout)
            print(f"UDP Receiver 소켓 생성 및 바인딩 완료: {self.host_ip}:{self.port}, 타임아웃 {self.timeout}초")
            return True
        except socket.error as e:
            print(f"[오류] UDP 소켓 bind 또는 옵션 설정 실패: {e}. 잠시 후 재시도합니다.")
            self.sock = None
            time.sleep(2) # 실패 시 잠시 대기
            return False
        except OSError as e:
             print(f"[경고] 수신 버퍼 크기 설정 실패 ({self.buffer_size}): {e}. 기본값 사용.")
             # 버퍼 설정 실패해도 bind는 성공할 수 있음
             if self.sock:
                 try:
                     self.sock.bind((self.host_ip, self.port))
                     self.sock.settimeout(self.timeout)
                     print(f"UDP Receiver 소켓 생성 및 바인딩 완료 (버퍼 설정 경고): {self.host_ip}:{self.port}")
                     return True
                 except socket.error as bind_e:
                     print(f"[오류] UDP 소켓 bind 실패 (OS 에러 후): {bind_e}")
                     self.sock = None
                     time.sleep(2)
                     return False
             else: # 소켓 생성 자체가 안 된 경우
                 print(f"[오류] UDP 소켓 생성 실패 (OS 에러): {e}")
                 time.sleep(2)
                 return False


    def receive_frame_data(self):
        """UDP 소켓에서 데이터를 수신하고 완전한 프레임 데이터를 재조립하여 반환합니다."""
        if not self.sock: # 소켓이 유효하지 않으면 재바인딩 시도
            print("[정보] 수신 소켓이 유효하지 않아 재바인딩 시도 중...")
            if not self._bind_socket():
                print("[오류] 소켓 재바인딩 실패. 수신 건너뜀.")
                return None # 바인딩 실패 시 데이터 없음

        completed_frame_data = None
        try:
            data, addr = self.sock.recvfrom(self.buffer_size)

            # --- 기존 데이터 처리 로직 (동일) ---
            if len(data) < 4: return None

            frame_seq = int.from_bytes(data[0:2], byteorder="big")
            current_time = time.time()

            if len(data) == 6: # 헤더 패킷
                if frame_seq not in self.frame_buffers:
                    self.frame_buffers[frame_seq]["expected_size"] = int.from_bytes(data[2:6], byteorder="big")
                    self.frame_buffers[frame_seq]["received_time"] = current_time
                    self.frame_buffers[frame_seq]["chunks"] = {}
                    self.frame_buffers[frame_seq]["data"] = bytearray()
                return None

            elif len(data) > 2 and data[2:5] == b"END": # 종료 신호
                if frame_seq in self.frame_buffers:
                    buffer = self.frame_buffers[frame_seq]
                    # 완성 여부 체크 (모든 청크 도착 기반 대신, 데이터 길이 기반)
                    all_chunks = sorted(buffer["chunks"].keys())
                    if all_chunks: # 청크가 하나라도 있다면 조립 시도
                        buffer["data"] = bytearray()
                        for i in all_chunks:
                            buffer["data"].extend(buffer["chunks"][i])

                    if buffer["expected_size"] > 0 and len(buffer["data"]) >= buffer["expected_size"]:
                         # 디코딩 시도하여 데이터 유효성 검증 (선택적)
                         try:
                             temp_img = np.frombuffer(buffer["data"][:buffer["expected_size"]], dtype=np.uint8)
                             cv2.imdecode(temp_img, cv2.IMREAD_COLOR) # 디코딩 가능 여부만 확인
                             completed_frame_data = buffer["data"][:buffer["expected_size"]]
                             # print(f"프레임 {frame_seq} 완성, 크기: {len(completed_frame_data)}")
                         except Exception as decode_error:
                             # print(f"[경고] 프레임 {frame_seq} 데이터 손상 의심 (디코딩 실패): {decode_error}")
                             pass # 손상된 데이터는 무시
                         finally:
                            # 성공/실패 여부와 관계없이 처리된 버퍼는 삭제
                             del self.frame_buffers[frame_seq]
                    else: # 데이터 불완전 시 버퍼 유지 (오래된 버퍼는 정리됨)
                         # print(f"프레임 {frame_seq} 불완전: 기대 {buffer['expected_size']}, 수신 {len(buffer['data'])}")
                         pass

                self._cleanup_old_buffers(current_time)
                return completed_frame_data

            elif len(data) > 4: # 데이터 청크
                if frame_seq in self.frame_buffers:
                    buffer = self.frame_buffers[frame_seq]
                    chunk_id = int.from_bytes(data[2:4], byteorder="big")
                    chunk_data = data[4:]
                    if chunk_id not in buffer["chunks"]:
                        buffer["chunks"][chunk_id] = chunk_data
                        buffer["received_time"] = current_time
                return None

        except socket.timeout:
            self._cleanup_old_buffers(time.time())
            return None # 타임아웃은 정상적인 상황일 수 있음
        except socket.error as e:
            print(f"[오류] UDP 수신 중 소켓 오류 발생: {e}. 소켓 재바인딩 시도...")
            # 소켓 오류 시 재바인딩 시도
            self._bind_socket() # 실패해도 다음 루프에서 다시 시도됨
            return None # 오류 발생 시 데이터 없음
        except Exception as e: # 기타 예외 (데이터 처리 등)
            print(f"[오류] UDP 수신 중 예기치 않은 예외 발생: {e}")
            # 이 경우 소켓 문제는 아닐 수 있으므로 일단 계속 진행
            return None

        return completed_frame_data # 정상 처리 시 None 반환 (종료 신호 처리 시에만 데이터 반환)


    def _check_and_assemble(self, frame_seq):
         # 이 메서드는 종료 신호 처리 로직으로 통합되어 사용하지 않음
         pass

    def _cleanup_old_buffers(self, current_time):
        """(내부 함수) 오래된 프레임 버퍼를 정리합니다."""
        old_seqs = [seq for seq, buf in self.frame_buffers.items() if current_time - buf["received_time"] > self.max_buffer_age]
        for seq in old_seqs:
            # print(f"[정보] 오래된 버퍼 정리: 시퀀스 {seq}, 크기 {len(self.frame_buffers[seq]['data'])}/{self.frame_buffers[seq]['expected_size']}")
            del self.frame_buffers[seq]

    def close(self):
        """UDP 소켓을 닫습니다."""
        if self.sock:
            try:
                self.sock.close()
                print("UDP Receiver 소켓 닫힘.")
            except Exception as e:
                print(f"[경고] 수신 소켓 닫기 중 오류 발생: {e}")
            finally:
                self.sock = None
