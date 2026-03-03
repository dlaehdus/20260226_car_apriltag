"""
검정글씨와 파란글씨를 동시에 같이 검출하도록해서 글씨문제 해결
"""
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ==========================================
# 1. 실제 번호판 크기 및 그룹화 배수 설정
# ==========================================
PLATE_WIDTH = 0.11    # 한국 표준 번호판 가로 (52cm)
PLATE_HEIGHT = 0.02   # 한국 표준 번호판 세로 (11cm)
PLATE_DEPTH = 0.02    # 3D 입체 상자의 두께 (5cm)
AXIS = 0.02           # 3D 축 길이

GROUP_WIDTH_MULT = 7.0  # 글자 너비의 몇 배까지 옆으로 묶을 것인가
GROUP_HEIGHT_MULT = 1.0  # 글자 높이의 몇 배까지 위아래로 묶을 것인가

# HSV 필터 설정 (전기차 파란색 번호판)
HUE_1, HUE_2 = 90, 130
SATURATION_1, SATURATION_2 = 65, 255
VALUE_1, VALUE_2 = 90, 255

# 검정색 글자 필터 설정 (번호판 문자만 정확히 잡기)
DARK_SAT_MAX = 115      # Saturation 최대 (너무 높이면 다른 색까지 잡힘) 115
DARK_VALUE_MAX = 130     # Value 최대 (너무 높이면 그림자/어두운 배경까지 잡힘) 92

# 3D 모델 좌표
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], 
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], 
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], 
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]  
], dtype=np.float32)

def get_plate_box_points(w, h, d):
    hw, hh = w / 2, h / 2
    return np.float32([
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0],
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d]
    ])

# ==========================================
# 2. 모델 및 리얼센스 초기화
# ==========================================
from sahi.models.ultralytics import UltralyticsDetectionModel
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        img_bgr = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # ==================== [주요 수정] 파란색 + 검정색 글자 마스크 ====================
        # 1. 기존 파란색 배경 마스크
        mask_blue = cv2.inRange(hsv, np.array([HUE_1, SATURATION_1, VALUE_1]), np.array([HUE_2, SATURATION_2, VALUE_2]))
        
        # 2. 검정색 글자 마스크 (번호판 문자만)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, DARK_SAT_MAX, DARK_VALUE_MAX])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 3. 파란색 영역 근처의 검정 글자만 허용 (다른 검은 물체/그림자/타이어 등 완전 제거)
        kernel = np.ones((5,5), np.uint8)
        blue_dilated = cv2.dilate(mask_blue, kernel, iterations=6)   # 파란 영역 약간 확장
        mask_black = cv2.bitwise_and(mask_black, blue_dilated)
        
        # 4. 합치기 + 노이즈 제거 + 글자 연결 강화
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        
        # 최종 입력 이미지 (파란색 배경 + 검정 글자 모두 살아있음)
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_combined)
        # =====================================================================
        
        results = get_sliced_prediction(img_hsv_blue, detection_model, slice_height=960, slice_width=960, verbose=0)

        detections = []
        for obj in results.object_prediction_list:
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            detections.append({
                'center': ((x1+x2)/2, (y1+y2)/2), 
                'p1':(x1,y1), 'p2':(x2,y1), 'p3':(x1,y2), 'p4':(x2,y2),
                'w': (x2-x1), 'h': (y2-y1)
            })

        # ==========================================
        # 3. 그룹화 (설정된 배수 변수 사용)
        # ==========================================
        groups = []
        visited = [False] * len(detections)

        for i in range(len(detections)):
            if visited[i]: continue
            current_group = [detections[i]]
            visited[i] = True
            
            ref_w = detections[i]['w']
            ref_h = detections[i]['h']
            
            for j in range(len(detections)):
                if not visited[j]:
                    dx = abs(detections[i]['center'][0] - detections[j]['center'][0])
                    dy = abs(detections[i]['center'][1] - detections[j]['center'][1])
                    
                    # 상단에서 정의한 배수 변수 적용
                    if dx < (ref_w * GROUP_WIDTH_MULT) and dy < (ref_h * GROUP_HEIGHT_MULT):
                        current_group.append(detections[j])
                        visited[j] = True
            
            if len(current_group) >= 3:
                groups.append(current_group)

        # ==========================================
        # 4. 개별 시각화
        # ==========================================
        for idx, group in enumerate(groups):
            group.sort(key=lambda x: x['center'][0])
            l_box, r_box = group[0], group[-1]
            img_pts_2d = np.array([l_box['p1'], r_box['p2'], l_box['p3'], r_box['p4']], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs)

            if success:
                cv2.drawFrameAxes(img_hsv_blue, K, dist_coeffs, rvec, tvec, AXIS)
                box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                projected_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
                projected_pts = np.int32(projected_pts).reshape(-1, 2)

                cv2.drawContours(img_hsv_blue, [projected_pts[:4]], -1, (0, 255, 0), 2)
                for i in range(4):
                    cv2.line(img_hsv_blue, tuple(projected_pts[i]), tuple(projected_pts[i+4]), (0, 255, 0), 2)
                cv2.drawContours(img_hsv_blue, [projected_pts[4:]], -1, (0, 255, 0), 2)

                z_pos = tvec[2][0]
                cv2.putText(img_hsv_blue, f"ID:{idx} Z:{z_pos:.2f}m", 
                            (l_box['p1'][0], l_box['p1'][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Multi-Plate Detection (Blue + Black Text)", img_hsv_blue)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()