"""
이전 버전에서는 학습된 YOLO11 모델이 단일 물체를 인식하는 성능에만 의존하였습니다.
그러나 화재 현장과 같이 노이즈가 많은 환경에서는 번호판이 아닌 일반 구조물이나 빛 반사 등을 번호판 숫자로 오인하는 '거짓 긍정(False Positive)'이 발생할 가능성이 컸습니다.
이번 버전에서는 차량 번호판이 특정 간격을 두고 여러 숫자가 연속적으로 나열된다는 패턴적 특성에 착안하여, 
개별 탐지 결과가 아닌 집단적 분포를 분석하는 로직을 새롭게 설계하였습니다.

이전 코드가 단순히 탐지된 박스를 화면에 그리는 데 그쳤다면, 
이번 버전에서는 탐지된 각 객체의 좌상단 및 우하단 좌표를 활용해 산술 평균 방식의 중심점(center) 좌표를 생성합니다. 
이를 위해 get_distance라는 사용자 정의 함수를 도입하여 두 객체 사이의 직선 거리를 계산하며, 
이는 단순한 탐지를 넘어 객체 간의 상관관계를 분석하는 기초 데이터로 활용됩니다.

본 코드의 핵심 차별점인 4번 섹션에서는 탐지된 모든 후보군을 대상으로 전수 조사를 수행합니다.
dist_threshold를 150픽셀(1280 해상도 기준)로 설정하여, 특정 객체 주변에 다른 탐지 객체가 밀집해 있는지 확인합니다.
번호판 숫자는 물리적으로 가깝게 배치되어야 한다는 원칙을 적용하여, 본인 주변에 임계값 이내의 이웃이 3개 이상(총 4개 이상의 객체 집합) 존재할 때만 이를 유효한 번호판 정보로 인정합니다.

기존 방식에서는 배경에서 우연히 발견된 숫자 형태의 노이즈가 단독으로 탐지될 경우 이를 걸러낼 방법이 없었습니다. 
그러나 새롭게 도입된 min_neighbors 조건은 단독으로 떨어져 있는 독립된 객체들을 '군집을 형성하지 못한 비정상 데이터'로 간주하여 시각화 대상에서 완전히 제외합니다. 
이는 실제 번호판 영역이 아닌 곳에서 발생하는 산발적인 오검출을 획기적으로 낮추는 여과 장치 역할을 합니다.
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
            overlap_width_ratio=0.2,
            verbose=0
        )
        # 필터링 전, 탐지된 모든 객체의 정보를 담아둘 임시 리스트를 만듭니다.
        detections = []
        # SAHI를 통해 찾은 결과물들을 하나씩 꺼내어 검사합니다.
        for obj in results.object_prediction_list:
            try:
                # 각 객체의 사각형 좌표(왼쪽 위 x, y, 오른쪽 아래 x, y)를 가져옵니다.
                bbox = obj.bbox.xyxy
            except AttributeError:
                bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            # 사각형의 정중앙 좌표를 계산합니다. 이 중심점은 나중에 글자들 사이의 거리를 측정하는 기준이 됩니다.
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # 좌표, 중심점, 라벨, 점수를 딕셔너리 형태로 저장하여 리스트에 추가합니다.
            detections.append({'bbox': (x1, y1, x2, y2), 'center': center, 'label': obj.category.name, 'score': obj.score.value})

        # ====================================================================================
        # 근접도 필터링 로직
        # ====================================================================================
        # 모든 검사를 통과한 진짜 번호판 객체들만 담을 리스트입니다.
        final_boxes = []
        # 두 객체 사이의 거리가 150픽셀 이내여야 '이웃'으로 인정한다는 기준치입니다.
        dist_threshold = 150 
        # 내 주변에 이웃이 최소 3개 이상(나를 포함해 총 4개 이상) 뭉쳐 있어야 진짜 번호판으로 간주하겠다는 조건입니다.
        min_neighbors = 3
        # 첫 번째 객체(det1)를 선택하여 검사를 시작합니다.
        for i, det1 in enumerate(detections):
            neighbor_count = 0
            # 선택한 객체(det1)와 나머지 모든 객체(det2) 사이의 거리를 하나씩 다 계산해 봅니다.
            for j, det2 in enumerate(detections):
                if i == j: continue
                # 두 객체 중심점 사이의 직선 거리를 구합니다.
                dist = get_distance(det1['center'], det2['center'])
                # 거리가 기준보다 가깝다면 이웃 숫자를 하나 올립니다.
                if dist < dist_threshold:
                    neighbor_count += 1
            # 주변에 충분한 숫자들이 모여 있다면, 이 객체는 유효한 번호판 데이터라고 판단하고 최종 리스트에 넣습니다.
            if neighbor_count >= min_neighbors:
                final_boxes.append(det1)

    
        # ====================================================================================
        # 결과 시각화
        # ====================================================================================
        # 필터링을 통과한 '진짜' 객체들만 하나씩 꺼냅니다.
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            # 번호판으로 확정된 객체 주위에 초록색 사각형을 그립니다.
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['label']} {det['score']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("RealSense + Cluster Filtering", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
