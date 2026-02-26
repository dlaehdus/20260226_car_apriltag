
"""
이번 버전은 에이프릴태그(AprilTag)의 정밀 위치 측정 원리를 적용하여 
카메라와 차량 사이의 물리적 거리와 공간적 회전 각도를 계산해내는 3차원 자세 추정 단계입니다.

가장 큰 기술적 차별점은 실제 번호판 규격(52cm x 11cm)에 기반한 수학적 모델링을 수행한다는 점입니다. 
시스템은 미리 정의된 3D 모델 좌표(obj_points)와 이미지 상에서 검출된 2D 꼭짓점 좌표(img_points)를 실시간으로 비교 분석합니다. 
이를 통해 단순히 영상 속의 픽셀 위치를 찾는 것을 넘어, 카메라를 기준으로 한 번호판의 상대적 위치(Translation)와 기울어짐(Rotation)을 수치화합니다.

이러한 SolvePnP 알고리즘의 적용은 번호판이 정면이 아닌 비스듬하거나 수직으로 세워진 각도에서 포착되더라도 카메라로부터의
정확한 거리(Depth)를 미터(m) 단위로 역산해냅니다. 
이는 단순한 시각화를 넘어, 번호판 중심에 X, Y, Z축으로 구성된 3D 좌표축(Frame Axes)을 투영함으로써 
시스템이 객체의 입체적인 자세를 완벽히 인지하고 있음을 보여줍니다.

또한, 로드리게스(Rodrigues) 변환 및 삼각함수 연산을 통해 번호판의 
Yaw(좌우 회전), Pitch(상하 기울기), Roll(회전) 값을 도(degree) 단위 각도로 산출합니다.
이는 화재 현장과 같은 복잡한 환경에서 관제자가 차량이 어떤 방향으로 진입해 있는지,
카메라와의 실제 거리가 어느 정도인지 즉각적으로 파악할 수 있는 정밀한 물리 데이터를 제공합니다.

결과적으로 이번 버전은 파편화된 2차원 이미지 데이터를 입체적인 3D 공간 좌표로 완벽히 치환하였습니다.
이는 전기차 화재 시 소방 로봇이나 무인 소화 설비가 화재 차량의 정확한 위치와 각도를 계산하여 
정밀한 방수를 수행할 수 있도록 돕는 시스템의 핵심적인 기술적 도약입니다.
"""


# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ====================================================================================
# 실제 번호판 크기 데이터
# ====================================================================================
PLATE_WIDTH = 0.52   # 가로 52cm
PLATE_HEIGHT = 0.11  # 세로 11cm
# 3D 공간상의 번호판 모델 좌표 (번호판 중심이 0,0,0)
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p1: 좌상
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # p2: 우상
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # p3: 좌하
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]  # p4: 우하
], dtype=np.float32)


# ====================================================================================
# 객체인식 파일 주소
# ====================================================================================
from sahi.models.ultralytics import UltralyticsDetectionModel
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")



# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)



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
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            detections.append({'center': ((x1+x2)/2, (y1+y2)/2), 'p1':(x1,y1), 'p2':(x2,y1), 'p3':(x1,y2), 'p4':(x2,y2)})

        # ====================================================================================
        # 근접도 필터링 로직
        # ====================================================================================
        final_boxes = []
        for i, det1 in enumerate(detections):
            neighbor_count = sum(1 for j, det2 in enumerate(detections) if i != j and get_distance(det1['center'], det2['center']) < 150)
            if neighbor_count >= 3: final_boxes.append(det1)



        # ====================================================================================
        # 3D 포즈 계산 및 시각화
        # ====================================================================================
        if len(final_boxes) > 1:
            final_boxes.sort(key=lambda x: x['center'][0])
            l_box, r_box = final_boxes[0], final_boxes[-1]
            # 2D 이미지상의 네 꼭짓점 (에이프릴태그처럼 활용)
            img_points = np.array([l_box['p1'], r_box['p2'], l_box['p3'], r_box['p4']], dtype=np.float32)
            # SolvePnP: 2D 점과 3D 모델 점을 비교하여 회전(rvec)과 이동(tvec) 계산
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
            if success:
                # 3D 축 그리기 (빨강:X, 초록:Y, 파랑:Z)
                cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, 0.2)
                # 거리 및 각도 계산
                x, y, z = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec) # 회전 벡터를 행렬로 변환
                pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
                yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                roll = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
                # 3. 외곽선 및 정보 표시
                cv2.line(img, l_box['p1'], r_box['p2'], (0, 255, 0), 2)
                cv2.line(img, l_box['p3'], r_box['p4'], (0, 255, 0), 2)
                cv2.line(img, l_box['p1'], l_box['p3'], (0, 255, 0), 2)
                cv2.line(img, r_box['p2'], r_box['p4'], (0, 255, 0), 2)
                info = f"Dist:{z:.2f}m Yaw:{yaw:.1f} P:{pitch:.1f}"
                cv2.putText(img, info, (l_box['p1'][0], l_box['p1'][1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Plate 3D Pose Estimation", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()