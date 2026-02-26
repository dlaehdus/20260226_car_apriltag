
"""
번호판의 기하학적 평면(Plane)을 정의하고 이를 3차원 공간으로 확장하기 위한 구조적 매핑 단계입니다.
이전 버전이 노이즈 필터링에 집중했다면, 이번 버전은 검출된 데이터들 사이의 '관계'를 설정하여 공간 정보를 추출하는 데 목적이 있습니다.
가장 먼저 탐지된 객체들 중 신뢰성이 확보된 후보군을 X축 좌표 기준으로 정렬합니다.
이는 번호판의 시작부터 끝까지 데이터를 순차적으로 처리하기 위함입니다.
정렬된 숫자 박스들 사이에서 인접한 객체끼리 좌상단(p1)은 좌상단끼리, 좌하단(p3)은 좌하단끼리 직선으로 연결하는 로직을 수행합니다.

이러한 모서리 연결 로직(Corner Connection)은 파편화되어 흩어져 있던 숫자 데이터들을 하나의 유기적인 선형 구조로 결합합니다.
이를 통해 단순히 숫자의 위치를 파악하는 것을 넘어, 번호판 전체의 기울기(Slope)와 소멸점(Vanishing Point)을 시각적으로 도출할 수 있습니다.
상단과 하단을 잇는 두 개의 가이드라인은 카메라와 차량 사이의 원 근감을 직관적으로 드러내며, 이는2차원 탐지 결과를 3차원 자세 추정(Pose Estimation)으로 변환하기 위한 결정적인 단서가 됩니다.

결과적으로 이번 버전은 시각적 복잡도를 줄이기 위해 불필요한 라벨 텍스트를 제거하고,
오직 평면의 흐름을 보여주는 선형 가이드에 집중하였습니다.
이러한 데이터 구조화는 화재나 연기로 인해 번호판의 일부가 가려진 극한 상황에서도,
남아있는 숫자들의 연결 흐름을 통해 보이지 않는 영역까지 추론할 수 있는 기하학적 안정성을 시스템에 부여합니다.
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
            detections.append({
                'bbox': (x1, y1, x2, y2), 
                'center': center, 
                # 연결에 필요한 1번, 3번 모서리만 저장
                'pts': {'p1': (x1, y1), 'p3': (x1, y2)}
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
        # x축 정렬 및 선 그리기
        # ====================================================================================
        final_boxes.sort(key=lambda x: x['center'][0])

        if len(final_boxes) > 1:
            for i in range(len(final_boxes) - 1):
                # 숫자 상단(1번) 연결
                cv2.line(img, final_boxes[i]['pts']['p1'], final_boxes[i+1]['pts']['p1'], (0, 255, 0), 2)
                # 숫자 하단(3번) 연결
                cv2.line(img, final_boxes[i]['pts']['p3'], final_boxes[i+1]['pts']['p3'], (0, 255, 0), 2)
        
        
        # ====================================================================================
        # 결과 시각화
        # ====================================================================================
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Plate Guide Lines", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()