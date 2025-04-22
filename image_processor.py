import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import config  # 설정값 사용


def decode_frame(frame_data):
    """수신된 byte 데이터를 이미지 프레임으로 디코딩합니다."""
    if not frame_data:
        return None
    try:
        img_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[오류] 프레임 디코딩 실패: {e}")
        return None


def undistort_frame(frame, K, D):
    """카메라 왜곡을 보정합니다."""
    if K is None or D is None or frame is None:
        return frame, K, D  # 원본 프레임 및 K, D 반환
    try:
        h, w = frame.shape[:2]
        # alpha=0: 유효한 픽셀만, alpha=1: 모든 픽셀 유지 (검은 영역 발생 가능)
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, D, (w, h), alpha=0, newImgSize=(w, h)
        )
        if new_K is None:
            return frame, K, D  # 실패 시 원본 반환

        # undistort 함수 사용
        dst = cv2.undistort(frame, K, D, None, new_K)

        # ROI(Region of Interest)를 사용하여 유효한 영역만 잘라내기 (선택 사항)
        # x, y, w, h = roi
        # if w > 0 and h > 0:
        #    dst = dst[y:y+h, x:x+w]

        # 참고: undistort 결과의 왜곡 계수는 0에 가깝다고 가정할 수 있으나,
        # 정확한 계산은 아니므로 new_D를 0으로 만드는 것은 주의 필요.
        # 여기서는 반환하지 않거나 원본 D를 그대로 반환하는 것이 안전할 수 있음.
        # new_D = np.zeros_like(D) # 이 부분은 사용하지 않음

        return dst, new_K  # 보정된 프레임과 새로운 카메라 매트릭스 반환
    except Exception as e:
        print(f"[오류] 왜곡 보정 실패: {e}")
        return frame, K  # 실패 시 원본 프레임과 K 반환


def _corner_points(corner):
    """ArUco 코너 좌표에서 중심점과 각 꼭지점 좌표를 추출합니다."""
    corner_points = corner[0]
    center = np.mean(corner_points, axis=0).astype(int)
    corner = np.array(corner).reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corner
    return center, topLeft, topRight, bottomRight, bottomLeft


# def _to_pos(tvec):
#     """변환 벡터(tvec)를 (x, y, z) 튜플로 변환합니다."""
#     if tvec is None:
#         return None, None, None
#     return round(tvec[0, 0], 2), round(tvec[1, 0], 2), round(tvec[2, 0], 2) # numpy.float32

# def _to_rot(rvec):
#     """회전 벡터(rvec)를 각도(degree) (rx, ry, rz) 튜플로 변환합니다."""
#     if rvec is None:
#         return None, None, None
#     return (
#         round(np.rad2deg(rvec[0, 0]), 2),
#         round(np.rad2deg(rvec[1, 0]), 2),
#         round(np.rad2deg(rvec[2, 0]), 2),
#     )
# # numpy.float32


def _to_pos(tvec):
    """변환 벡터(tvec)를 파이썬 float (x, y, z) 튜플로 변환합니다."""
    if tvec is None:
        return None, None, None
    # .item()을 사용하여 numpy float을 python float으로 변환
    x = round(tvec[0, 0].item(), 2)
    y = round(tvec[1, 0].item(), 2)
    z = round(tvec[2, 0].item(), 2)
    return x, y, z


def _to_rot(rvec):
    """회전 벡터(rvec)를 파이썬 float 각도(degree) (rx, ry, rz) 튜플로 변환합니다."""
    if rvec is None:
        return None, None, None
    # np.rad2deg 결과에 .item()을 적용하여 numpy float을 python float으로 변환
    rx = round(np.rad2deg(rvec[0, 0]).item(), 2)
    ry = round(np.rad2deg(rvec[1, 0]).item(), 2)
    rz = round(np.rad2deg(rvec[2, 0]).item(), 2)
    return rx, ry, rz


def detect_aruco(
    frame,
    K=None,
    D=None,
    aruco_type_str=config.ARUCO_DICT_TYPE,
    marker_length=config.ARUCO_MARKER_LENGTH,
):
    """프레임에서 ArUco 마커를 감지하고 위치/자세 추정 결과를 그립니다."""
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if aruco_type_str not in config.ARUCO_DICT:
        print(f"[오류] 지원하지 않는 ArUco 타입: {aruco_type_str}")
        return frame

    aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT[aruco_type_str])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    detected_info = []  # 감지된 마커 정보 저장 리스트

    if ids is not None and len(ids) > 0:
        # 감지된 마커 그리기
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 카메라 파라미터가 있으면 위치/자세 추정
        if K is not None and D is not None:
            # 마커 3D 좌표 정의 (중심 기준)
            obj_points = np.array(
                [
                    [-marker_length / 2, marker_length / 2, 0],  # 좌상
                    [marker_length / 2, marker_length / 2, 0],  # 우상
                    [marker_length / 2, -marker_length / 2, 0],  # 우하
                    [-marker_length / 2, -marker_length / 2, 0],  # 좌하
                ],
                dtype=np.float32,
            )

            for i, corner in enumerate(corners):
                img_points = corner[0].astype(np.float32)
                try:
                    # solvePnP로 rvec, tvec 계산
                    # IPPE_SQUARE는 평면 마커에 더 정확할 수 있음
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:
                        # 좌표축 그리기
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, marker_length * 0.5)

                        # 정보 추출 및 텍스트 표시
                        center, topLeft, _, _, _ = _corner_points(corner)
                        x, y, z = _to_pos(tvec)
                        rx, ry, rz = _to_rot(rvec)
                        distance = np.linalg.norm(tvec)

                        marker_id = ids[i][0]
                        # info = {
                        #     "id": marker_id,
                        #     "tvec": (x, y, z),
                        #     "rvec_deg": (rx, ry, rz),
                        #     "distance": distance,
                        # }
                        info = {
                            "id": marker_id.item(),
                            "tvec": (x, y, z),
                            "rvec_deg": (rx, ry, rz),
                            "distance": distance.item(),
                        }
                        detected_info.append(info)
                        print(f"[INFO] {info}")  # 콘솔 출력 대신 반환

                        # id_text = f"ID: {marker_id}"
                        pos_text = f"Pos:({x:.2f},{y:.2f},{z:.2f})m"
                        rot_text = f"Rot:({rx:.1f},{ry:.1f},{rz:.1f})d"

                        # cv2.putText(
                        #     frame,
                        #     id_text,
                        #     (center[0] + 10, center[1] - 20),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5,
                        #     (0, 0, 255),
                        #     2,
                        # )
                        cv2.putText(
                            frame,
                            pos_text,
                            (int(topLeft[0]) - 30, int(topLeft[1]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )
                        cv2.putText(
                            frame,
                            rot_text,
                            (int(topLeft[0]) - 30, int(topLeft[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

                except cv2.error as e:
                    print(f"[오류] ID {ids[i][0]} solvePnP 계산 실패: {e}")
                    continue  # 다음 마커 처리
    return frame, detected_info  # 처리된 프레임과 감지 정보 리스트 반환


def display_frame(frame, window_title="Stream"):
    """프레임을 화면에 표시하고 사용자 입력을 처리합니다."""
    if frame is None or frame.size == 0:
        # print("[경고] 표시할 프레임이 없습니다.")
        return None  # 아무것도 안함

    cv2.imshow(window_title, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        return "quit"
    elif key == ord("s"):
        filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
        save_path = Path("captures")
        save_path.mkdir(parents=True, exist_ok=True)
        full_path = save_path / filename
        try:
            cv2.imwrite(str(full_path), frame)
            print(f"[정보] 이미지 저장됨: {full_path}")
        except Exception as e:
            print(f"[오류] 이미지 저장 실패: {e}")
    return None  # 계속 진행
