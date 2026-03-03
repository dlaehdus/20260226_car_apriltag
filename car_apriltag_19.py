"""
찐찐찐 최종코드
"""
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ==========================================
# 1. 설정 및 필터 변수
# ==========================================
PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH = 0.11, 0.02, 0.02
AXIS = 0.05
GROUP_WIDTH_MULT, GROUP_HEIGHT_MULT = 7.0, 1.2

# --- [추가된 필터 변수: 화면엔 안 보임] ---
prev_tvec, prev_rvec = None, None
alpha = 0.15 # 필터 강도 (좌표 튐 방지)

obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], 
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], 
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], 
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]  
], dtype=np.float32)

def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x, y, z = math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
    else:
        x, y, z = math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0
    return np.degrees([x, y, z])

def get_plate_box_points(w, h, d):
    hw, hh = w / 2, h / 2
    return np.float32([[-hw,-hh,0],[hw,-hh,0],[hw,hh,0],[-hw,hh,0],
                       [-hw,-hh,-d],[hw,-hh,-d],[hw,hh,-d],[-hw,hh,-d]])

# ==========================================
# 2. 모델 및 리얼센스 초기화
# ==========================================
from sahi.models.ultralytics import UltralyticsDetectionModel
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")

pipeline = rs.pipeline(); config = rs.config()
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
            pad = 5
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > 20:
                        rect = cv2.minAreaRect(main_cnt)
                        box_pts = cv2.boxPoints(rect)
                        box_pts[:, 0] += rx1; box_pts[:, 1] += ry1
                        pts = sorted(box_pts, key=lambda x: x[1])
                        top = sorted(pts[:2], key=lambda x: x[0]); bottom = sorted(pts[2:], key=lambda x: x[0])
                        detections.append({
                            'center': ((x1+x2)/2, (y1+y2)/2),
                            'obb_corners': np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32),
                            'w': x2-x1, 'h': y2-y1
                        })

        # ==========================================
        # 3. 그룹화 및 PnP (시각화 유지)
        # ==========================================
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            group = [detections[i]]; visited[i] = True
            for j in range(i+1, len(detections)):
                if not visited[j]:
                    dx = abs(detections[i]['center'][0] - detections[j]['center'][0])
                    dy = abs(detections[i]['center'][1] - detections[j]['center'][1])
                    if dx < (detections[i]['w'] * GROUP_WIDTH_MULT) and dy < (detections[i]['h'] * GROUP_HEIGHT_MULT):
                        group.append(detections[j]); visited[j] = True
            
            if len(group) >= 3:
                group.sort(key=lambda x: x['center'][0])
                l_obb, r_obb = group[0]['obb_corners'], group[-1]['obb_corners']
                img_pts_2d = np.array([l_obb[0], r_obb[1], l_obb[2], r_obb[3]], dtype=np.float32)

                # --- [시각화: 개별 글자 OBB - 원본 스타일 유지] ---
                for item in group:
                    cv2.drawContours(img_hsv_blue, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    # --- [필터링 및 데이터 추출: 튀는 값 방지] ---
                    if prev_tvec is not None:
                        # 0.5m 이상 급격하게 튀면 노이즈로 보고 이전 값 사용
                        if np.linalg.norm(tvec - prev_tvec) < 0.5:
                            tvec = alpha * tvec + (1 - alpha) * prev_tvec
                            rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    prev_tvec, prev_rvec = tvec, rvec

                    # 터미널에 좌표값 출력 (화면 방해 안 함)
                    tx, ty, tz = tvec.flatten()
                    roll, pitch, yaw = rotation_vector_to_euler(rvec)
                    print(f"XYZ: {tx:.3f}, {ty:.3f}, {tz:.3f} | RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

                    # --- [시각화: 3D 축 및 박스 - 원본 스타일 유지] ---
                    cv2.drawFrameAxes(img_hsv_blue, K, dist_coeffs, rvec, tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    cv2.drawContours(img_hsv_blue, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4): cv2.line(img_hsv_blue, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_hsv_blue, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("OBB-based PnP Detection", img_hsv_blue)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    pipeline.stop(); cv2.destroyAllWindows()