"""
인식알고리즘이 실행하기 전에 블루필터를 적용해서 파란색 번호판만 검출할수 있게한 알고리즘 해당 알고리즘

개선할점 : 
번호판의 글자를 인식하다가 만약에 이상한 곳의 번호가 인식이 되면 오인식한곳과 번호판이 이어져서 이상한곳으로 좌표축이 생성됌
번호판이 특정 물체에 대해 가려지면 번호판의 중심좌표를 찍는게아니라 가려진부분은 번호판으로 인식하지 못함 이걸 해결해야함
이 번호검출 모델과 번호판검출 모델을 합치면 정확도가 올라가는지 확인필요
번호판이 가까워지면 검출을 잘 못함
"""



# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math


# ====================================================================================
# 실제 번호판 크기 데이터
# ====================================================================================
PLATE_WIDTH = 0.13   # 가로 52cm
PLATE_HEIGHT = 0.03  # 세로 11cm
PLATE_DEPTH = 0.01   # 3D 입체 상자의 두께 (10cm로 설정, 원하면 변경 가능)
AXIS = 0.01
# 제일 앞이 색상HUE - 파란색의 종류를 결정함 값이 너무 낮으면 초록색 높으면 보라색 최소0 최대 179
# 그 비트가 채도SATURATION - 색의 진함을 결정함 작을수록 흰색을 허용하고 높을수록 흰색을 걸러냄
# 마지막 비트 자리가VALUE 명도 - 색의 밝기를 결정함 작을수록 검정색을 허용하고 높을수록 검정색을 걸러냄
HUE_1=90
HUE_2=129
SATURATION_1=65
SATURATION_2=255
VALUE_1=0
VALUE_2=255
# 3D 공간상의 번호판 모델 좌표 (번호판 중심이 0,0,0)
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p1: 좌상
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p2: 우상
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # p3: 좌하
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]  # p4: 우하
], dtype=np.float32)



# ====================================================================================
# 3D 가상 모델의 뼈대를 만드는 작업
# ====================================================================================
def get_plate_box_points(w, h, d):
    """번호판 크기를 바탕으로 3D 직육면체의 8개 꼭짓점 좌표를 반환합니다."""
    hw, hh = w / 2, h / 2
    return np.float32([
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0], # 바닥면 (Z=0, 번호판 면)
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d] # 윗면 (Z=-d, 카메라 쪽으로 튀어나오는 방향)
    ])

# ====================================================================================
# 객체인식 파일 주소
# ====================================================================================
from sahi.models.ultralytics import UltralyticsDetectionModel
# 모델 경로가 올바른지 확인해주세요.
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")



# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)



# ====================================================================================
# 거리 계산 함수
# ====================================================================================
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("프로그램 시작... 'q'를 누르면 종료됩니다.")




# ====================================================================================
# 메인루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        img_bgr = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([HUE_1, SATURATION_1, VALUE_1]) 
        upper_blue = np.array([HUE_2, SATURATION_2, VALUE_2])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3,3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_blue)
        
        # ====================================================================================
        # 정밀 검출 메커니즘
        # ====================================================================================
        results = get_sliced_prediction(
            img_hsv_blue, # <-- 입력 이미지를 필터링된 이미지로 변경
            detection_model, 
            slice_height=960, slice_width=960,
            overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0
        )

        detections = []
        for obj in results.object_prediction_list:
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            detections.append({'center': ((x1+x2)/2, (y1+y2)/2), 'p1':(x1,y1), 'p2':(x2,y1), 'p3':(x1,y2), 'p4':(x2,y2)})
        
        # ====================================================================================
        # 근접도 필터링 로직
        # ====================================================================================
        final_boxes = []
        for i, det1 in enumerate(detections):
            neighbor_count = sum(1 for j, det2 in enumerate(detections) if i != j and get_distance(det1['center'], det2['center']) < 150)
            if neighbor_count >= 3: final_boxes.append(det1)


        # ====================================================================================
        # 3D 포즈 계산 및 시각화
        # ====================================================================================
        if len(final_boxes) > 1:
            final_boxes.sort(key=lambda x: x['center'][0])
            l_box, r_box = final_boxes[0], final_boxes[-1]
            img_points = np.array([l_box['p1'], r_box['p2'], l_box['p3'], r_box['p4']], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
            if success:
                cv2.drawFrameAxes(img_hsv_blue, K, dist_coeffs, rvec, tvec, AXIS)
                box_points = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                img_pts, _ = cv2.projectPoints(box_points, rvec, tvec, K, dist_coeffs)
                img_pts = np.int32(img_pts).reshape(-1, 2)
                cv2.drawContours(img_hsv_blue, [img_pts[:4]], -1, (0, 255, 0), 2)
                for i in range(4):
                    cv2.line(img_hsv_blue, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 2)
                cv2.drawContours(img_hsv_blue, [img_pts[4:]], -1, (0, 255, 0), 2)
                x_pos, y_pos, z_pos = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec)
                pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
                yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                roll = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
                info_screen = f"Z: {z_pos:.2f}m"
                cv2.putText(img_hsv_blue, info_screen, (l_box['p1'][0], l_box['p1'][1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                print(f"Plate | Pos(x,y,z): {x_pos:.2f},{y_pos:.2f},{z_pos:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")
        cv2.imshow("Plate 3D Pose Estimation (Blue Filtered)", img_hsv_blue)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()