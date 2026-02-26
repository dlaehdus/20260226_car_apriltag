"""
전 버전
전기차 번호판인 파란색을 검출하는 필터를 적용함 목표한 전기차 번호판기준 인식률 향상
하지만 해당 전기차번호판을 정확하게 직사각형형태로 가져오고 해당 직사각형을 에어프릴태그의 형식으로 변환하기엔
부정확하게 인식돼는 문제가 있음 즉 인식은 잘되나 에어프릴태그처럼 거리와 각도의 오차가 큼
"""

# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ====================================================================================
# 실제 번호판 크기 설정 (미터 단위)
# ====================================================================================
PLATE_W = 0.520
PLATE_H = 0.110

# ====================================================================================
# 번호판의 3D 모델 좌표 (모서리 점들)
# ====================================================================================
obj_pts = np.float32([
    [-PLATE_W/2, -PLATE_H/2, 0],
    [ PLATE_W/2, -PLATE_H/2, 0],
    [ PLATE_W/2,  PLATE_H/2, 0],
    [-PLATE_W/2,  PLATE_H/2, 0]
])


# ====================================================================================
# 회전 행렬 변환 함수
# ====================================================================================
# 3x3 회전 행렬 $R$을 입력받아 Roll, Pitch, Yaw 각도를 계산하는 함수입니다.
def get_rpy(R):
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
    return roll, pitch, yaw


# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"에러 발생: {e}\n리얼센스 뷰어가 켜져있다면 반드시 끄고 실행하세요.")
    exit()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# ====================================================================================
# AprilTag 수준의 정밀·안정화 파라미터 (이번에 가장 크게 강화)
# ====================================================================================
# 지수 이동 평균(EMA) 필터 값입니다. 
# 92%는 이전 프레임 값을 유지하고 8%만 새 값을 반영하여, 정지 상태에서 수치가 떨리는 현상을 극도로 억제합니다.
# (AprilTag 내부 temporal stability 모방)
SMOOTH_ALPHA = 0.92
# 서브픽셀 정밀도를 위한 탐색 창 크기입니다. 픽셀과 픽셀 사이의 소수점 좌표를 찾습니다.
CORNER_SUBPIX_WIN = (7, 7)
CORNER_SUBPIX_ZERO = (10, 10)
# 이전 프레임의 위치/각도를 저장하여 계산의 연속성을 유지하기 위한 변수입니다.
# 이전 프레임 pose 저장 (useExtrinsicGuess용 + smoothing용)
prev_rvec = None
prev_tvec = None
print("'q'를 누르면 종료됩니다.")



# ====================================================================================
# 이미지 읽기 및 전처리
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # AprilTag의 quad_sigma 효과 모방 (노이즈 제거)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([82, 45, 55])
        upper_blue = np.array([128, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 7))
        closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations=1)

        # ====================================================================================
        # 번호판 후보 검출 및 좌표 정렬
        # ====================================================================================
        # 이진화된 이미지에서 하얀색 덩어리들의 외곽 경계선(윤곽선)을 모두 찾아냅니다.
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1800: continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w < h: w, h = h, w
            aspect = w / h
            if not (4.35 < aspect < 5.15): continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = approx.reshape(4, 2).astype(np.float32)
            rect_pts = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect_pts[0] = pts[np.argmin(s)]
            rect_pts[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect_pts[1] = pts[np.argmin(diff)]
            rect_pts[3] = pts[np.argmax(diff)]
            # AprilTag refine_edges와 동일한 고정밀 subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.0001)
            rect_pts_sub = rect_pts.copy()
            cv2.cornerSubPix(gray, rect_pts_sub, CORNER_SUBPIX_WIN, CORNER_SUBPIX_ZERO, criteria)

            # ====================================================================================
            # PnP 알고리즘을 통한 3D 자세 추정 및 시각화 에어프릴의 원리와 동일
            # ====================================================================================
            if prev_rvec is not None and prev_tvec is not None:
                # 이전 pose를 초기값으로 사용 → 정지 상태에서 거의 변하지 않음
                success, rvec, tvec = cv2.solvePnP(obj_pts, rect_pts_sub, K, dist_coeffs,
                                                   rvec=prev_rvec, tvec=prev_tvec,
                                                   useExtrinsicGuess=True,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                success, rvec, tvec = cv2.solvePnP(obj_pts, rect_pts_sub, K, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                continue
            if prev_rvec is None or prev_tvec is None:
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
            else:
                rvec = SMOOTH_ALPHA * prev_rvec + (1 - SMOOTH_ALPHA) * rvec
                tvec = SMOOTH_ALPHA * prev_tvec + (1 - SMOOTH_ALPHA) * tvec
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = get_rpy(R)
            x, y, z = tvec.flatten()
            cv2.drawFrameAxes(img, K, dist_coeffs, R, tvec, PLATE_W * 0.6)
            thickness = 0.05
            box_3d = np.float32([
                [-PLATE_W/2, -PLATE_H/2, 0], [PLATE_W/2, -PLATE_H/2, 0],
                [PLATE_W/2,  PLATE_H/2, 0], [-PLATE_W/2,  PLATE_H/2, 0],
                [-PLATE_W/2, -PLATE_H/2, -thickness], [PLATE_W/2, -PLATE_H/2, -thickness],
                [PLATE_W/2,  PLATE_H/2, -thickness], [-PLATE_W/2,  PLATE_H/2, -thickness]
            ])
            img_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
            img_pts = np.int32(img_pts).reshape(-1, 2)
            cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), 3)
            for i in range(4):
                cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 3)
            cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 0), 3)
            info = f"EV Plate Z:{z:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
            cv2.putText(img, info, (int(rect_pts_sub[0][0]), int(rect_pts_sub[0][1])-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"EV Plate 검출 Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")
        cv2.imshow('D415 EV Plate 3D (AprilTag-style)', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()