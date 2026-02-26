import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ==========================================
# 실제 한국 차량 번호판 크기 (미터 단위, 표준 520mm × 110mm)
# ==========================================
PLATE_W = 0.520
PLATE_H = 0.110

# 번호판 3D 모델 좌표 (중심 원점, 카메라 좌표계와 동일하게 x:우, y:하, z:정면)
obj_pts = np.float32([
    [-PLATE_W/2, -PLATE_H/2, 0],   # 좌상
    [ PLATE_W/2, -PLATE_H/2, 0],   # 우상
    [ PLATE_W/2,  PLATE_H/2, 0],   # 우하
    [-PLATE_W/2,  PLATE_H/2, 0]    # 좌하
])

def get_rpy(R):
    """회전 행렬 → Roll, Pitch, Yaw (도) 변환 (AprilTag 코드와 동일)"""
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
    return roll, pitch, yaw

# 리얼센스 D415 초기화 (AprilTag 코드와 완전 동일)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"에러 발생: {e}\n리얼센스 뷰어가 실행 중이라면 반드시 종료 후 다시 실행하세요.")
    exit()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)   # D415 컬러 스트림은 이미 보정됨

print("프로그램 시작... 번호판 감지 중 (검출률 크게 향상됨) | 'q' 키로 종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 개선된 번호판 전처리 (Blackhat + Morphology) ← AprilTag 스타일의 "특정 패턴" 검출 기법
        # ==========================================
        # 1. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 2. Blackhat: 검은 글씨(번호판 문자)를 밝게 강조 → 번호판 영역만 강하게 부각
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))   # 한국 번호판 가로:세로 비율에 최적
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # 3. Closing으로 문자 간 연결
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, sq_kernel)

        # 4. 이진화 (OTSU 자동 임계값)
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 5. 약간 팽창하여 외곽선 연결
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 외곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 큰 영역부터 처리 (속도 + 신뢰도 ↑)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:          # 너무 작은 것 제외 (원거리 번호판도 잡히도록 여유 있게)
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)   # 4각형 근사

            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            # minAreaRect로 정확한 회전된 사각형 aspect ratio 계산 (한국 번호판 ≈ 4.73)
            rot_rect = cv2.minAreaRect(cnt)
            w, h = rot_rect[1]
            if w < h:
                w, h = h, w
            aspect_ratio = w / h
            if not (3.8 < aspect_ratio < 5.6):
                continue

            # 4개 점 정렬 (좌상 → 우상 → 우하 → 좌하) ← solvePnP에 정확히 대응
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]      # 좌상
            rect[2] = pts[np.argmax(s)]      # 우하
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]   # 우상
            rect[3] = pts[np.argmax(diff)]   # 좌하

            # ==========================================
            # AprilTag와 완전 동일한 PnP pose 계산
            # ==========================================
            success, rvec, tvec = cv2.solvePnP(obj_pts, rect, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = get_rpy(R)
            x, y, z = tvec.flatten()

            # 시각화 1: 3D 축 (빨강X, 초록Y, 파랑Z)
            cv2.drawFrameAxes(img, K, dist_coeffs, R, tvec, PLATE_W * 0.6)

            # 시각화 2: 얇은 3D 초록 상자 (두께 5cm)
            thickness = 0.05
            box_3d = np.float32([
                [-PLATE_W/2, -PLATE_H/2, 0], [PLATE_W/2, -PLATE_H/2, 0],
                [PLATE_W/2, PLATE_H/2, 0], [-PLATE_W/2, PLATE_H/2, 0],
                [-PLATE_W/2, -PLATE_H/2, -thickness], [PLATE_W/2, -PLATE_H/2, -thickness],
                [PLATE_W/2, PLATE_H/2, -thickness], [-PLATE_W/2, PLATE_H/2, -thickness]
            ])
            img_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
            img_pts = np.int32(img_pts).reshape(-1, 2)

            cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), 3)
            for i in range(4):
                cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i + 4]), (0, 255, 0), 3)
            cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 0), 3)

            # 화면에 정보 출력 (AprilTag 코드와 동일 스타일)
            info = f"Plate Z:{z:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
            cv2.putText(img, info, (int(rect[0][0]), int(rect[0][1]) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            print(f"번호판 검출! Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        # 결과 화면
        cv2.imshow('D415 Korean License Plate Pose (AprilTag-style)', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()