"""
전 버전
실제로 번호판을 인식하였으나 거리가 멀어지거나 가까워짐에 따라 인식률이 급격하게 변화함
따라서 이번버전은 우리가 타겟하는 전기자동차 번호판인 파란색을 중점적으로 전처리할것임
이번 버전
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
    [-PLATE_W/2, -PLATE_H/2, 0],   # 좌상
    [ PLATE_W/2, -PLATE_H/2, 0],   # 우상
    [ PLATE_W/2,  PLATE_H/2, 0],   # 우하
    [-PLATE_W/2,  PLATE_H/2, 0]    # 좌하
])

# ====================================================================================
# 회전 행렬 변환 함수
# ====================================================================================
# 3x3 회전 행렬 $R$을 입력받아 Roll, Pitch, Yaw 각도를 계산하는 함수입니다.
def get_rpy(R):
    # 삼각함수(atan2)를 이용해 각 축의 회전량을 도(Degree) 단위로 추출합니다.
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
print("전기차 번호판인 파란 배경만 사용")
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
        """
        전 버전에 비해 개선한 부분
        """
        # 일반적인 컬러 이미지(BGR)를 HSV 색 공간으로 변환합니다.
        # BGR 방식은 빛의 밝기에 따라 색상 값이 너무 크게 변해서 색을 구별하기 어렵습니다.
        # 반면 HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 나뉘어 있어 특정 색(파란색)만 골라내기에 훨씬 유리
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 파란색에 해당하는 색상 범위(lower ~ upper)를 설정하고, 그 범위에 해당하는 픽셀만 하얗게(255), 나머지는 검게(0) 만드는 마스크 이미지를 생성
        # 영상 안에서 파란색 번호판 후보들만 남기고 나머지 배경을 모두 지워버리는 필터링 과정입니다.
        lower_blue = np.array([85, 50, 60])
        upper_blue = np.array([125, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 가로 28픽셀, 세로 8픽셀 크기의 직사각형 모양 커널(기준 틀)을 만듭니다.
        # 번호판은 가로로 긴 형태이므로, 가로로 긴 필터를 사용해야 가로 방향의 글자나 끊어진 영역을 연결하기에 가장 적합합니다.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 8))
        
        # Closing(닫기) 연산을 수행합니다. 이는 팽창(Dilate) 후 침식(Erode)을 연속으로 하는 기법입니다.
        # 번호판 내부에 있는 검은색 글자나 작은 구멍들을 하얗게 메워서, 번호판 전체가 하나의 꽉 찬 흰색 직사각형 덩어리가 되도록 만듭니다.
        closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        # 팽창(Dilate) 연산을 2회 반복합니다. 흰색 영역을 사방으로 확장시킵니다.
        # 마스크 연산 중에 얇아지거나 끊어진 외곽선 부분을 두툼하게 보강하여, 나중에 윤곽선(Contour)을 찾을 때 사각형 형태가 훨씬 더 잘 잡히도록 도와줍니다.
        closed = cv2.dilate(closed, None, iterations=2)


        # ====================================================================================
        # 번호판 후보 검출 및 좌표 정렬
        # ====================================================================================
        # 이진화된 이미지에서 하얀색 덩어리들의 외곽 경계선(윤곽선)을 모두 찾아냅니다.
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w < h: w, h = h, w
            aspect = w / h
            if not (4.3 < aspect < 5.2):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
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


            # ====================================================================================
            # PnP 알고리즘을 통한 3D 자세 추정 및 시각화 에어프릴의 원리와 동일
            # ====================================================================================
            success, rvec, tvec = cv2.solvePnP(obj_pts, rect_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                continue
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
            cv2.putText(img, info, (int(rect_pts[0][0]), int(rect_pts[0][1])-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"Plate 검출 Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        cv2.imshow('D415 EV Plate 3D', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
