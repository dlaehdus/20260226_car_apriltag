"""
가장 큰 기술적 차별점은 상·하·좌·우 네 방향의 선분(4-way lines)을 모두 연결하여
기하학적으로 완벽한 닫힌 다각형(Quad Boundary)을 생성한다는 점입니다.
단순히 가로선만 긋는 것이 아니라, 제일 왼쪽 숫자의 수직축(p1-p3)과 제일 오른쪽 숫자의 수직축(p2-p4)을 
연결함으로써 번호판이 카메라를 바라보는 실제 평면의 기울기를 그대로 반영합니다.
"""

# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math
from sahi.models.ultralytics import UltralyticsDetectionModel

# ====================================================================================
# 파일 주소와 GPU관리
# ====================================================================================
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"
)



# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)



# ====================================================================================
# 거리 계산 함수
# ====================================================================================
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



# ====================================================================================
# 메인루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        img = np.asanyarray(color_frame.get_data())
        # ====================================================================================
        # 정밀 검출 메커니즘
        # ====================================================================================
        results = get_sliced_prediction(
            img, detection_model,
            slice_height=960, 
            slice_width=960,
            overlap_height_ratio=0.1, 
            overlap_width_ratio=0.1, 
            verbose=0
        )
        detections = []
        for obj in results.object_prediction_list:
            try:
                bbox = obj.bbox.xyxy
            except AttributeError:
                bbox = obj.bbox.to_xyxy()
            
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # 1:좌상, 2:우상, 3:좌하, 4:우하
            detections.append({
                'center': center, 
                'p1': (x1, y1), 'p2': (x2, y1), 
                'p3': (x1, y2), 'p4': (x2, y2)
            })

        # ====================================================================================
        # 근접도 필터링 로직
        # ====================================================================================
        final_boxes = []
        dist_threshold = 150 
        min_neighbors = 3  

        for i, det1 in enumerate(detections):
            neighbor_count = 0
            for j, det2 in enumerate(detections):
                if i == j: continue
                dist = get_distance(det1['center'], det2['center'])
                if dist < dist_threshold:
                    neighbor_count += 1
            if neighbor_count >= min_neighbors:
                final_boxes.append(det1)

        # ====================================================================================
        # 사각형 그리기
        # ====================================================================================
        if len(final_boxes) > 1:
            # X좌표 기준으로 정렬 (왼쪽 -> 오른쪽)
            final_boxes.sort(key=lambda x: x['center'][0])
            left_box = final_boxes[0]   # 제일 왼쪽 숫자
            right_box = final_boxes[-1] # 제일 오른쪽 숫자
            # [선 1] 상단 가로선 (좌상 -> 우상)
            cv2.line(img, left_box['p1'], right_box['p2'], (0, 255, 0), 2)
            # [선 2] 하단 가로선 (좌하 -> 우하)
            cv2.line(img, left_box['p3'], right_box['p4'], (0, 255, 0), 2)
            # [선 3] 왼쪽 세로선 (좌상 -> 좌하)
            cv2.line(img, left_box['p1'], left_box['p3'], (0, 255, 0), 2)
            # [선 4] 오른쪽 세로선 (우상 -> 우하)
            cv2.line(img, right_box['p2'], right_box['p4'], (0, 255, 0), 2)
        cv2.imshow("License Plate Quad Boundary", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()