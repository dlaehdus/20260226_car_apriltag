# 전기차 번호판 인식 성공 제일 잘됌
# 아무번호판이나 에어프릴태그처럼 좌표와 각도를 도출해내는 코드

import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ==========================================
# 실제 한국 전기차 번호판 크기 (미터 단위)
# ==========================================
PLATE_W = 0.520
PLATE_H = 0.110

# 번호판 3D 모델 좌표 (중심 기준)
obj_pts = np.float32([
    [-PLATE_W/2, -PLATE_H/2, 0],   # 좌상
    [ PLATE_W/2, -PLATE_H/2, 0],   # 우상
    [ PLATE_W/2,  PLATE_H/2, 0],   # 우하
    [-PLATE_W/2,  PLATE_H/2, 0]    # 좌하
])

def get_rpy(R):
    """AprilTag 코드와 완전히 동일한 RPY 계산"""
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
    return roll, pitch, yaw

# =============== 리얼센스 D415 초기화 (AprilTag 코드와 동일) ===============
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

print("=== 전기차 파란 번호판 검출 시작 (AprilTag 스타일) ===")
print("파란 배경만 사용 → 다른 물체 거의 안 잡힙니다.")
print("'q'를 누르면 종료됩니다.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ==================== 전기차 파란색 번호판 전용 검출 ====================
        # 1. HSV로 파란색만 추출 (한국 전기차 번호판 파란 배경에 최적화)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([85, 50, 60])    # ← 파란색이 약하면 85→75로 낮추세요
        upper_blue = np.array([125, 255, 255]) # ← 너무 많이 잡히면 125→110으로 낮추세요
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 2. 가로로 긴 번호판 형태 만들기 (Morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 8))
        closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations=2)

        # 외곽선 검출
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:                     # 너무 작은 건 무시 (원거리 번호판도 잡히게 여유있음)
                continue

            # 한국 번호판 비율 검사 (4.3 ~ 5.2)
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w < h: w, h = h, w
            aspect = w / h
            if not (4.3 < aspect < 5.2):
                continue

            # 4각형으로 근사
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            # 점 정렬 (solvePnP용: 좌상 → 우상 → 우하 → 좌하)
            pts = approx.reshape(4, 2).astype(np.float32)
            rect_pts = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect_pts[0] = pts[np.argmin(s)]
            rect_pts[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect_pts[1] = pts[np.argmin(diff)]
            rect_pts[3] = pts[np.argmax(diff)]

            # ==================== AprilTag와 완전히 동일한 PnP 계산 ====================
            success, rvec, tvec = cv2.solvePnP(obj_pts, rect_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = get_rpy(R)
            x, y, z = tvec.flatten()

            # 시각화 1: 3D 축 (빨강X 초록Y 파랑Z)
            cv2.drawFrameAxes(img, K, dist_coeffs, R, tvec, PLATE_W * 0.6)

            # 시각화 2: 초록색 3D 상자 (두께 5cm)
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

            # 정보 표시 (AprilTag 코드와 동일 스타일)
            info = f"EV Plate Z:{z:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
            cv2.putText(img, info, (int(rect_pts[0][0]), int(rect_pts[0][1])-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            print(f"EV Plate 검출! Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        # 결과 화면 (AprilTag 코드와 완전히 동일한 한 개 창만)
        cv2.imshow('D415 EV Plate 3D', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()