"""
이번 버전
기존의 이미지 전처리 방식(Canny Edge, Contour 필터링, HSV 색상 마스크 등)은 하드웨어 리소스를 적게 소모한다는 장점이 있으나,
실제 주차장이나 화재 현장과 같이 변수가 많은 환경에서는 인식률이 급격히 떨어진다는 치명적인 한계가 있었습니다.
특히 번호판에 오염이 있거나 연기로 인해 경계선이 흐려질 경우, 수치 기반의 전통적 필터로는 전기차 화재 시나리오에서 요구되는 고도의 신뢰성을 확보하기 불가능함을 확인하였습니다.
이에 따라 이번 버전에서는 단순한 픽셀 분석을 넘어 물체의 맥락을 이해하는 딥러닝 기반의 YOLO11 모델과 소형 객체 탐지 성능을 극대화하는 SAHI 기법을 도입하여 성능을 전면 고도화하였습니다.

가장 먼저 UltralyticsDetectionModel을 통해 학습된 best.pt 가중치를 로드하는데,
이때 device="cuda:0" 설정을 사용하여 RTX 3070 GPU의 병렬 연산 기능을 풀가동함으로써 수만 개의 파라미터 계산을 실시간으로 처리합니다.
이어지는 핵심 로직인 get_sliced_prediction 함수는 고해상도 전체 영상을 격자 형태로 쪼개서 개별 분석하는 SAHI(Sliced Aided Hyper Inference) 기법을 수행합니다.
이는 원거리의 작은 번호판을 강제로 확대하여 분석하는 효과를 주어,
화염이나 연기에 가려진 극한 상황에서도 번호판의 패턴을 놓치지 않고 찾아내는 압도적인 검출력을 보장합니다.

검출 단계에서는 slice_height와 slice_width를 960픽셀로 설정하여 연산 효율과 정밀도의 균형을 맞추었으며, 
조각 간의 중첩 비율인 overlap 값을 조절하여 경계선에서 번호판이 잘려 인식되지 않는 현상을 방지하였습니다. 
탐지된 결과는 object_prediction_list를 통해 순회하며, 
각 후보군의 bbox.xyxy 좌표를 추출해 정수형으로 변환한 뒤 화면에 초록색 사각형과 확신도 점수를 출력합니다. 
최종적으로 이 시스템은 리얼센스 카메라의 고해상도 데이터와 딥러닝의 강력한 패턴 매핑 능력을 결합함으로써, 
기존 필터 방식으로는 도달할 수 없었던 '화재 상황 속 차량 특정'이라는 목표를 실현 가능한 수준으로 끌어올렸습니다.
"""

# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
# 이미지를 조각내어 추론하고 다시 합쳐주는 SAHI의 핵심 함수를 가져옵니다.
from sahi.predict import get_sliced_prediction
# YOLO 모델을 SAHI 프레임워크 내에서 사용할 수 있게 연결해주는 클래스입니다.
from sahi.models.ultralytics import UltralyticsDetectionModel



# ====================================================================================
# 파일 주소와 GPU관리
# ====================================================================================
# 이전에 YOLO11로 직접 학습시킨 최고의 성능을 내는 가중치 파일 경로입니다.
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
# 학습된 모델을 로드합니다. device="cuda:0"을 통해 RTX 3070 GPU의 병렬 연산 능력을 사용하여 속도를 높입니다.
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
# 고해상도(1280x720) 컬러 영상을 초당 30프레임으로 받도록 설정합니다.
# 해상도가 높을수록 SAHI의 슬라이싱 효과가 극대화됩니다.
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)



# ====================================================================================
# 메인루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # 리얼센스 데이터를 넘파이 배열로 변환
        # 카메라 프레임을 OpenCV에서 처리 가능한 넘파이(Numpy) 배열 형식으로 변환합니다.
        img = np.asanyarray(color_frame.get_data())


        # ====================================================================================
        # 정밀 검출 메커니즘
        # ====================================================================================
        # SAHI Sliced Inference (이미지를 쪼개서 정밀 검출)
        # 전체 이미지를 한 번에 모델에 넣지 않고, 설정된 크기로 잘라서 모델에 여러 번 넣습니다
        results = get_sliced_prediction(
            img,
            detection_model,
            # 이미지를 960x960 크기의 조각으로 나눕니다. 
            # 원본(1280x720)보다 큰 영역을 지정하면 작은 객체가 상대적으로 크게 인식되어 검출력이 대폭 향상됩니다.
            slice_height=960,
            slice_width=960,
            # 조각들 사이에 겹치는 영역(10~20%)을 둡니다. 
            # 번호판이 조각 경계선에 걸려 잘리는 경우를 방지하여 연속성을 유지합니다.
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.2,
            # 터미널에 불필요한 로그 출력을 막아 처리 성능을 조금이라도 더 확보합니다.
            verbose=0
        )


        # ====================================================================================
        # 결과 시각화 및 후처리
        # ====================================================================================
        # 쪼개진 이미지들에서 발견된 모든 번호판 정보를 하나씩 꺼냅니다.
        for object_prediction in results.object_prediction_list:
            try:
                # 번호판의 위치(좌측 상단 x, y, 우측 하단 x, y) 좌표를 추출합니다.
                bbox = object_prediction.bbox.xyxy
            except AttributeError:
                bbox = object_prediction.bbox.to_xyxy()
            # 인식된 클래스 이름(예: EV Plate)과 확신도(Confidence) 점수를 가져옵니다.
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
