"""
입체적인 3D 객체(Volumetric Object)로 모델링하는 데 목적이 있습니다. 
이전 버전이 2차원 사각형 위에 좌표축을 세우는 것에 집중했다면, 이번 버전은 번호판의 실제 두께까지 고려한 시각화를 통해 공간 인지 능력을 극대화했습니다.

3D 직육면체 모델링 (Volumetric Box Rendering)가장 눈에 띄는 변화는 get_plate_box_points 함수를 통한 3D 입체 상자의 구현입니다.
이전에는 번호판의 네 꼭짓점만을 다뤘으나, 이번 코드에서는 10cm의 두께(PLATE_DEPTH)를 가진 8개의 꼭짓점을 정의합니다
cv2.projectPoints 함수를 통해 이 8개의 점을 카메라 시점에 맞춰 2D 화면에 투영하고, 
이를 상하좌우 및 기둥 선으로 연결하여 초록색 큐브 형태를 완성합니다. 
이는 화재 상황에서 차량의 전면부가 차지하는 물리적 부피를 직관적으로 파악하게 해줍니다.

SolvePnP를 이용한 에이프릴태그 방식의 좌표 역산이전 버전의 PnP 로직을 계승하되
final_boxes의 양 끝점을 활용해 2D 이미지 좌표와 3D 모델 좌표를 정밀하게 매핑합니다.
이는 에이프릴태그가 패턴의 기하학적 왜곡을 분석해 거리를 재는 원리와 완벽히 일치합니다
이를 통해 얻어낸 이동 벡터(tvec)와 회전 벡터(rvec)는 카메라로부터 번호판까지의 실제 거리와 기울기를
소수점 두 자리까지 정밀하게 산출하는 근거가 됩니다.
 
오일러 각(RPY) 및 3차원 위치 데이터의 정량화시각화에만 그치지 않고, 
cv2.Rodrigues 변환을 통해 얻은 회전 행렬에서 Roll(회전), Pitch(상하), Yaw(좌우) 값을 도(degree) 단위로 
정확히 추출합니다. 이제 터미널 창에는 차량의 정확한 3차원 좌표(x, y, z)와 각도 데이터가 실시간으로 출력됩니다.
이 데이터는 관제 시스템에서 "차량이 어느 방향으로, 얼마나 멀리 있는지"를 디지털 데이터로 저장하고 분석하는 데 
핵심적인 역할을 합니다.
"""

import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ==========================================
# 1. 실제 번호판 크기 및 입체 두께 설정 (미터 단위)
# ==========================================
PLATE_WIDTH = 0.52   # 가로 52cm
PLATE_HEIGHT = 0.11  # 세로 11cm
PLATE_DEPTH = 0.10   # 3D 입체 상자의 두께 (10cm로 설정, 원하면 변경 가능)

# 3D 공간상의 번호판 모델 좌표 (번호판 중심이 0,0,0)
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p1: 좌상
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p2: 우상
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # p3: 좌하
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]  # p4: 우하
], dtype=np.float32)

def get_plate_box_points(w, h, d):
    """번호판 크기를 바탕으로 3D 직육면체의 8개 꼭짓점 좌표를 반환합니다."""
    hw, hh = w / 2, h / 2
    return np.float32([
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0], # 바닥면 (Z=0, 번호판 면)
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d] # 윗면 (Z=-d, 카메라 쪽으로 튀어나오는 방향)
    ])

# 2. 모델 및 리얼센스 초기화
from sahi.models.ultralytics import UltralyticsDetectionModel
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 카메라 내장 파라미터(Intrinsic) 획득
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("프로그램 시작... 'q'를 누르면 종료됩니다.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        img = np.asanyarray(color_frame.get_data())

        results = get_sliced_prediction(
            img, detection_model, slice_height=960, slice_width=960,
            overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0
        )

        detections = []
        for obj in results.object_prediction_list:
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            detections.append({'center': ((x1+x2)/2, (y1+y2)/2), 'p1':(x1,y1), 'p2':(x2,y1), 'p3':(x1,y2), 'p4':(x2,y2)})

        # 근접도 필터링
        final_boxes = []
        for i, det1 in enumerate(detections):
            neighbor_count = sum(1 for j, det2 in enumerate(detections) if i != j and get_distance(det1['center'], det2['center']) < 150)
            if neighbor_count >= 3: final_boxes.append(det1)

        # 3D 포즈 계산 및 시각화
        if len(final_boxes) > 1:
            final_boxes.sort(key=lambda x: x['center'][0])
            l_box, r_box = final_boxes[0], final_boxes[-1]

            # 2D 이미지상의 네 꼭짓점 (에이프릴태그처럼 활용)
            img_points = np.array([l_box['p1'], r_box['p2'], l_box['p3'], r_box['p4']], dtype=np.float32)

            # SolvePnP: 2D 점과 3D 모델 점을 비교하여 회전(rvec)과 이동(tvec) 계산
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)

            if success:
                # 1. 3D 축 그리기 (빨강:X, 초록:Y, 파랑:Z)
                cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 0.2)

                # 2. 3D 초록색 입체 상자 그리기
                box_points = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                img_pts, _ = cv2.projectPoints(box_points, rvec, tvec, K, dist_coeffs)
                img_pts = np.int32(img_pts).reshape(-1, 2)

                # 선 긋기: 바닥면, 기둥, 윗면을 순서대로 연결하여 상자 완성
                cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), 2)
                for i in range(4):
                    cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 2)
                cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 0), 2)

                # 3. 거리 및 각도(Roll, Pitch, Yaw) 계산
                x, y, z = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec) # 회전 벡터를 행렬로 변환
                pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
                yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                roll = math.degrees(math.atan2(-R[1, 0], R[0, 0]))

                # 4. 화면 표시 및 터미널 출력
                # 영상 화면에는 심플하게 거리(Z) 정도만 띄웁니다 (원치 않으시면 지워도 됩니다)
                info_screen = f"Z: {z:.2f}m"
                cv2.putText(img, info_screen, (l_box['p1'][0], l_box['p1'][1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 터미널(Console) 창에 RPY 및 좌표 정보 출력
                print(f"Plate | Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        cv2.imshow("Plate 3D Pose Estimation", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()