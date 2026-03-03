"""
미치게 잘된 코드 개잘됌 근데 좌표값 출력안함
"""
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ==========================================
# 1. 실제 번호판 크기 및 3D 모델 설정
# ==========================================
# 에이프릴태그처럼 실제 글자/번호판의 물리적 크기가 중요합니다.
PLATE_WIDTH = 0.11    
PLATE_HEIGHT = 0.02   
PLATE_DEPTH = 0.02    
AXIS = 0.05           

GROUP_WIDTH_MULT = 7.0  
GROUP_HEIGHT_MULT = 1.2  

# PnP용 3D 좌표 (평면 기준: 왼쪽상단, 오른쪽상단, 왼쪽하단, 오른쪽하단)
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
        
        # --- [마스킹 로직 유지] ---
        mask_blue = cv2.inRange(hsv, np.array([90, 65, 90]), np.array([130, 255, 255]))
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 115, 130]))
        blue_dilated = cv2.dilate(mask_blue, np.ones((5,5), np.uint8), iterations=6)
        mask_black = cv2.bitwise_and(mask_black, blue_dilated)
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_combined)
        
        results = get_sliced_prediction(img_hsv_blue, detection_model, slice_height=960, slice_width=960, verbose=0)

        detections = []
        for obj in results.object_prediction_list:
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # --- [OBB 좌표 추출 핵심 알고리즘] ---
            pad = 5
            rx1, ry1 = max(0, x1-pad), max(0, y1-pad)
            rx2, ry2 = min(1280, x2+pad), min(720, y2+pad)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            obb_corners = None
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > 20:
                        rect = cv2.minAreaRect(main_cnt)
                        # 에이프릴태그처럼 네 꼭짓점 좌표를 순서대로 정렬
                        box_pts = cv2.boxPoints(rect)
                        box_pts[:, 0] += rx1
                        box_pts[:, 1] += ry1
                        
                        # 좌표 정렬: [좌상, 우상, 좌하, 우하] 순서로 변환 (PnP 매칭용)
                        pts = sorted(box_pts, key=lambda x: x[1]) # y기준 정렬
                        top = sorted(pts[:2], key=lambda x: x[0]) # 상단 2개 중 x기준
                        bottom = sorted(pts[2:], key=lambda x: x[0]) # 하단 2개 중 x기준
                        obb_corners = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

            if obb_corners is not None:
                detections.append({
                    'center': ((x1+x2)/2, (y1+y2)/2),
                    'obb_corners': obb_corners,
                    'w': x2-x1, 'h': y2-y1
                })

        # ==========================================
        # 3. 그룹화 및 OBB 기반 PnP
        # ==========================================
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            group = [detections[i]]
            visited[i], ref_w, ref_h = True, detections[i]['w'], detections[i]['h']
            
            for j in range(i+1, len(detections)):
                if not visited[j]:
                    dx = abs(detections[i]['center'][0] - detections[j]['center'][0])
                    dy = abs(detections[i]['center'][1] - detections[j]['center'][1])
                    if dx < (ref_w * GROUP_WIDTH_MULT) and dy < (ref_h * GROUP_HEIGHT_MULT):
                        group.append(detections[j])
                        visited[j] = True
            
            if len(group) >= 3:
                group.sort(key=lambda x: x['center'][0])
                
                # --- [수정] OBB 좌표를 기준점으로 사용 ---
                # 왼쪽 끝 글자의 OBB 좌상/좌하, 오른쪽 끝 글자의 OBB 우상/우하 좌표 추출
                l_obb = group[0]['obb_corners']
                r_obb = group[-1]['obb_corners']
                
                # AprilTag와 동일한 방식의 2D 대응점 생성
                img_pts_2d = np.array([
                    l_obb[0], # 왼쪽 글자 좌상
                    r_obb[1], # 오른쪽 글자 우상
                    l_obb[2], # 왼쪽 글자 좌하
                    r_obb[3]  # 오른쪽 글자 우하
                ], dtype=np.float32)

                # 개별 글자 OBB 시각화
                for item in group:
                    cv2.drawContours(img_hsv_blue, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    cv2.drawFrameAxes(img_hsv_blue, K, dist_coeffs, rvec, tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)

                    # 3D 박스 그리기
                    cv2.drawContours(img_hsv_blue, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4):
                        cv2.line(img_hsv_blue, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_hsv_blue, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("OBB-based PnP Detection", img_hsv_blue)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()