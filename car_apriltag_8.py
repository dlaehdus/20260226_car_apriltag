"""
이번 코드는 산발적으로 탐지된 숫자 데이터들로부터 번호판의 전체 영역(Region of Interest)을 하나의 단일 평면으로 통합하는 경계 외곽 추출(Boundary Extraction) 단계입니다. 
이전 버전이 숫자들 사이의 흐름을 파악하는 데 집중했다면, 
이번 버전은 번호판이라는 객체의 시작과 끝을 규정하는 데 목적이 있습니다.

가장 먼저 탐지된 객체군을 X축 기준으로 정렬한 뒤, 
집합의 극단값인 가장 왼쪽 박스(First)와 가장 오른쪽 박스(Last)를 선택합니다. 
이후 개별 숫자를 잇는 방식 대신, 제일 왼쪽 박스의 좌측 지점들(p1, p3)과 
제일 오른쪽 박스의 우측 지점들(p2, p4)을 직접 연결하는 로직을 수행합니다.

이러한 양 끝점 연결 로직(Extreme Points Connection)은 번호판의 부분적인 정보를 통합하여
하나의 거대한 사각형 가이드를 생성합니다. 이는 개별 숫자의 위치에 휘둘리지 않고 번호판 전체의 
기울기와 물리적 점유 면적을 명확히 정의합니다. 상단(좌상→우상)과 하단(좌하→우하)을 가로지르는 
긴 직선은 카메라 렌즈의 왜곡이나 차량의 입체적인 각도를 반영하는 평면의 외곽선이 되며,
이는 향후 번호판 전체를 하나의 이미지로 떼어내는 투영 변환(Perspective Transform)을 위한 가장 완벽한 기초 데이터를 제공합니다.

결과적으로 이번 버전은 파편화된 정보를 하나의 구조적인 바운더리로 묶어냄으로써 시각적 안정성을 극대화하였습니다.
특히 화재 연기나 장애물로 인해 중간 숫자들의 연결성이 끊기더라도, 
검출된 양 끝점만으로 전체 번호판의 윤곽을 복원할 수 있는 공간적 추론 능력을 확보하였습니다. 
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
            slice_height=960, slice_width=960,
            overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0
        )
        detections = []
        for obj in results.object_prediction_list:
            try:
                bbox = obj.bbox.xyxy
            except AttributeError:
                bbox = obj.bbox.to_xyxy()
            
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # 모든 모서리 좌표 저장
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
        # 양 끝점 연결 로직
        # ====================================================================================
        if len(final_boxes) > 1:
            # X좌표 기준으로 정렬 (왼쪽 -> 오른쪽)
            final_boxes.sort(key=lambda x: x['center'][0])
            # 가장 왼쪽 박스와 가장 오른쪽 박스 선택
            left_box = final_boxes[0]
            right_box = final_boxes[-1]
            # 상단 연결: 제일 왼쪽 박스의 p1(좌상) -> 제일 오른쪽 박스의 p2(우상)
            cv2.line(img, left_box['p1'], right_box['p2'], (0, 255, 0), 2)
            # 하단 연결: 제일 왼쪽 박스의 p3(좌하) -> 제일 오른쪽 박스의 p4(우하)
            cv2.line(img, left_box['p3'], right_box['p4'], (0, 255, 0), 2)
            # (옵션) 양옆 세로선까지 그려서 닫힌 사각형을 만들고 싶다면 아래 주석 해제
            # cv2.line(img, left_box['p1'], left_box['p3'], (0, 255, 0), 2)
            # cv2.line(img, right_box['p2'], right_box['p4'], (0, 255, 0), 2)

        # ====================================================================================
        # 결과 시각화
        # ====================================================================================
        for det in final_boxes:
            cv2.rectangle(img, det['p1'], det['p4'], (0, 255, 0), 1)

        cv2.imshow("Plate Boundary Guide", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()