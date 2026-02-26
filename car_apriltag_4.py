# 차량 번호판 인식을 제일 잘하는 코드
# 숫자를 제일 잘 인식 영역을 쪼개서 GPU로 연산함
# 다른 숫자도 해당 번호판으로 인식함 따라서 차량 번호판과 같이 연속괴어 인식되는 특정패턴만 검출하도록 맵핑


import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction

# 1. 모델 클래스 직접 임포트 (버전 호환성 해결)
try:
    from sahi.models.ultralytics import UltralyticsDetectionModel
except ImportError:
    from sahi.models.yolov8 import Yolov8DetectionModel as UltralyticsDetectionModel

# 2. 모델 및 리얼센스 설정
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'

# 모델 로드 (RTX 3070 GPU 사용)
detection_model = UltralyticsDetectionModel(
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"
)

# 리얼센스 파이프라인 시작
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)


try:
    while True:
        # 프레임 대기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # 리얼센스 데이터를 넘파이 배열로 변환
        img = np.asanyarray(color_frame.get_data())

        # 3. SAHI Sliced Inference (이미지를 쪼개서 정밀 검출)
        # slice_height/width를 640으로 설정하여 작은 번호판을 크게 인식하게 함
        results = get_sliced_prediction(
            img,
            detection_model,
            slice_height=960,
            slice_width=960,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.2,
            verbose=0 # 터미널 메시지 간소화
        )

        # 4. 결과 렌더링
        for object_prediction in results.object_prediction_list:
            # BoundingBox 좌표 추출 (버전별 에러 방지)
            try:
                bbox = object_prediction.bbox.xyxy # [xmin, ymin, xmax, ymax]
            except AttributeError:
                bbox = object_prediction.bbox.to_xyxy()
            
            label = object_prediction.category.name
            score = object_prediction.score.value
            
            # 정수 좌표 변환
            x1, y1, x2, y2 = map(int, bbox)
            
            # 박스(초록색) 및 텍스트 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 화면 표시
        cv2.imshow("RealSense + SAHI Optimized", img)
        
        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()