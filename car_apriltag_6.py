"""
에어프릴태그의 형태를 갖기 위한 코드

특히 이번 버전에서 새롭게 추가된 모서리 연결 로직(Corner Connection)은
필터링된 숫자들을 X좌표 기준으로 정렬한 뒤, 인접한 숫자끼리의 좌상단(p1)과 좌하단(p3) 모서리를 직선으로 연결합니다.
이는 개별적으로 흩어져 보이던 숫자 박스들을 하나의 선형 구조로 결합하여, 
사용자에게 번호판 전체의 흐름과 정렬 상태를 직관적으로 시각화해 줍니다.
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
# 이전에 YOLO11로 직접 학습시킨 최고의 성능을 내는 가중치 파일 경로입니다.
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
# 두 박스의 중심점 사이의 거리
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
            img,
            detection_model,
            slice_height=960,
            slice_width=960,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            verbose=0
        )
        # 필터링 전, 탐지된 모든 객체의 정보를 담아둘 임시 리스트를 만듭니다.
        detections = []
        for obj in results.object_prediction_list:
            try:
                bbox = obj.bbox.xyxy
            except AttributeError:
                bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append({
                'bbox': (x1, y1, x2, y2), 
                'center': center, 
                'label': obj.category.name,
                # 각 모서리 정의 (1: 좌상, 2: 우상, 3: 좌하, 4: 우하)
                'pts': {'p1': (x1, y1), 'p2': (x2, y1), 'p3': (x1, y2), 'p4': (x2, y2)}
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
        # 모서리 연결 로직
        # ====================================================================================
        # 왼쪽 숫지부터 차례대로 연결하기 위해 x좌표 기준으로 정렬
        final_boxes.sort(key=lambda x: x['center'][0])

        if len(final_boxes) > 1:
            for i in range(len(final_boxes) - 1):
                box_curr = final_boxes[i]
                box_next = final_boxes[i+1]

                # 각 숫자의 1번(좌상) 모서리끼리 연결
                cv2.line(img, box_curr['pts']['p1'], box_next['pts']['p1'], (0, 255, 0), 2)
                # 각 숫자의 3번(좌하) 모서리끼리 연결
                cv2.line(img, box_curr['pts']['p3'], box_next['pts']['p3'], (0, 255, 0), 2)

        # ====================================================================================
        # 결과 시각화
        # ====================================================================================
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, det['label'], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Plate Corner Connection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()