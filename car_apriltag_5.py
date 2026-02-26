import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# 1. 모델 클래스 임포트
from sahi.models.ultralytics import UltralyticsDetectionModel

# 2. 모델 및 리얼센스 설정
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# 거리 계산 함수 (두 박스의 중심점 사이의 거리)
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        img = np.asanyarray(color_frame.get_data())

        results = get_sliced_prediction(
            img,
            detection_model,
            slice_height=960,
            slice_width=960,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            verbose=0
        )

        # 박스 정보 추출 및 중심점 계산
        detections = []
        for obj in results.object_prediction_list:
            try:
                bbox = obj.bbox.xyxy
            except AttributeError:
                bbox = obj.bbox.to_xyxy()
            
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append({'bbox': (x1, y1, x2, y2), 'center': center, 'label': obj.category.name, 'score': obj.score.value})

        # 4. 근접도 필터링 로직
        final_boxes = []
        # 번호판 숫자는 보통 가로로 길게 늘어서 있으므로 임계값을 150~200픽셀 정도로 설정
        # (해상도 1280 기준, 숫자 간격에 따라 조절 가능)
        dist_threshold = 150 
        min_neighbors = 3  # 본인 제외 주변에 3개 이상(총 4개 이상) 있어야 인정

        for i, det1 in enumerate(detections):
            neighbor_count = 0
            for j, det2 in enumerate(detections):
                if i == j: continue
                
                dist = get_distance(det1['center'], det2['center'])
                if dist < dist_threshold:
                    neighbor_count += 1
            
            if neighbor_count >= min_neighbors:
                final_boxes.append(det1)

        # 5. 필터링된 결과만 렌더링
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['label']} {det['score']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("RealSense + Cluster Filtering", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()