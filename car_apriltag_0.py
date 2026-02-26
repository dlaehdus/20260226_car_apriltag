"""
realsenseD415 카메라에 비친 번호판의 2D이미지 분석을 통해 실제 세계의 3D위치와 각도를 찾아내는 프로그램
인식률이 낮아 번호판을 찾지 못하는 문제가 발생함 데이터를 학습시켜서 Yolo와 결합해서 인식률을 올려야할것같음
"""


# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import pyrealsense2 as rs
import numpy as np
import cv2
import math


# ====================================================================================
# 실제 번호판 크기 설정 (미터 단위)
# ====================================================================================
PLATE_W = 0.52  
PLATE_H = 0.11

# ====================================================================================
# 번호판의 3D 모델 좌표 (모서리 점들)
# ====================================================================================
obj_pts = np.float32([
    [-PLATE_W/2, -PLATE_H/2, 0], # 좌상
    [ PLATE_W/2, -PLATE_H/2, 0], # 우상
    [ PLATE_W/2,  PLATE_H/2, 0], # 우하
    [-PLATE_W/2,  PLATE_H/2, 0]  # 좌하
])

# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
# 카메라 데이터가 흐르는 통로(파이프라인)를 생성합니다.
pipeline = rs.pipeline()
# 카메라 설정 객체를 만듭니다.
config = rs.config()
# 컬러 영상을 1280x720 해상도, BGR 형식, 초당 30프레임으로 설정합니다
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 설정을 적용하여 카메라 작동을 시작합니다.
profile = pipeline.start(config)
# 카메라 렌즈의 고유 파라미터(왜곡, 초점거리 등)를 가져옵니다.
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# 카메라 내부 행렬(Intrinsic Matrix)을 구성합니다. 3D를 2D로 투영할 때 핵심이 됩니다.
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
# 렌즈 왜곡 계수입니다. 여기서는 왜곡이 없다고 가정(0)하고 설정했습니다.
dist_coeffs = np.zeros(5)

# ====================================================================================
# 회전 행렬 변환 함수
# ====================================================================================
# 3x3 회전 행렬 $R$을 입력받아 Roll, Pitch, Yaw 각도를 계산하는 함수입니다.
def get_rpy(R):
    # 삼각함수(atan2)를 이용해 각 축의 회전량을 도(Degree) 단위로 추출합니다.
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
    return roll, pitch, yaw


# ====================================================================================
# 이미지 읽기 및 전처리
# ====================================================================================
try:
    while True:
        # 카메라로부터 새로운 프레임 세트를 기다립니다.
        frames = pipeline.wait_for_frames()
        # 그중 컬러 이미지 프레임만 추출합니다.
        color_frame = frames.get_color_frame()
        # 프레임을 제대로 못 가져왔다면 이번 루프는 건너뜁니다.
        if not color_frame: continue
        # 리얼센스 데이터를 넘파이 배열(OpenCV 형식)로 변환합니다.
        img = np.asanyarray(color_frame.get_data())
        # 연산 속도를 높이기 위해 흑백 이미지로 바꿉니다.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 노이즈를 제거하기 위해 이미지를 살짝 흐리게 만듭니다.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 이미지에서 경계선(엣지)만 하얗게 추출합니다.
        edged = cv2.Canny(blur, 50, 200)

        # ====================================================================================
        # 번호판 후보 검출 및 좌표 정렬
        # ====================================================================================

        # 하얀 경계선들을 연결해 폐곡선(윤곽선) 덩어리들을 찾습니다.
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 찾아낸 모든 윤곽선을 하나씩 검사합니다.
        for cnt in contours:
            # 윤곽선의 전체 둘레 길이를 계산합니다.
            peri = cv2.arcLength(cnt, True)
            # 복잡한 선을 직선 위주의 다각형으로 단순화합니다.
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # 꼭짓점이 4개(사각형)이고 면적이 일정 크기 이상인 것만 골라냅니다.
            if len(approx) == 4 and cv2.contourArea(cnt) > 2000:
                # 점 정렬 (좌상, 우상, 우하, 좌하) 찾은 네 점의 좌표를 4x2 배열로 정리합니다.
                pts = approx.reshape(4, 2).astype(np.float32)
                # 정렬된 좌표를 담을 빈 배열을 만듭니다.
                rect = np.zeros((4, 2), dtype="float32")
                # 좌표의 합과 차를 계산해 위치(좌상, 우상 등)를 판별합니다.
                s = pts.sum(axis=1)
                # 점들을 [좌측상단, 우측상단, 우측하단, 좌측하단] 순서로 확정하여 저장합니다.
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # ====================================================================================
                # PnP 알고리즘을 통한 3D 자세 추정 및 시각화 에어프릴의 원리와 동일
                # ====================================================================================
                # 가장 핵심 줄입니다. 실제 크기(obj_pts)와 화면 속 크기(rect)를 비교해 카메라와의 상대적 위치(tvec)와 회전(rvec)을 계산합니다.
                success, rvec, tvec = cv2.solvePnP(obj_pts, rect, K, dist_coeffs)
                # 계산에 성공했을 경우에만 다음 시각화를 진행합니다.
                if success:
                    # 회전 벡터를 3x3 행렬 형식으로 변환합니다.
                    R, _ = cv2.Rodrigues(rvec)
                    # 앞서 만든 함수로 회전 각도를 구합니다.
                    roll, pitch, yaw = get_rpy(R)
                    # 번호판 중심에 빨강, 초록, 파랑색의 3D 좌표축을 그립니다.
                    cv2.drawFrameAxes(img, K, dist_coeffs, R, tvec, 0.2)
                    # 번호판 뒤로 5cm 두께를 가진 가상의 3D 박스 꼭짓점 8개를 정의합니다.
                    box_pts_3d = np.float32([[-PLATE_W/2, -PLATE_H/2, 0], [PLATE_W/2, -PLATE_H/2, 0], 
                                            [PLATE_W/2, PLATE_H/2, 0], [-PLATE_W/2, PLATE_H/2, 0],
                                            [-PLATE_W/2, -PLATE_H/2, -0.05], [PLATE_W/2, -PLATE_H/2, -0.05], 
                                            [PLATE_W/2, PLATE_H/2, -0.05], [-PLATE_W/2, PLATE_H/2, -0.05]])
                    # 3D 박스 좌표를 현재 카메라 각도에 맞춰 2D 화면 좌표로 변환합니다.
                    img_pts_2d, _ = cv2.projectPoints(box_pts_3d, rvec, tvec, K, dist_coeffs)
                    img_pts_2d = np.int32(img_pts_2d).reshape(-1, 2)
                    # 변환된 2D 좌표들을 선으로 연결해 화면에 입체적인 초록색 박스를 그립니다.
                    cv2.drawContours(img, [img_pts_2d[:4]], -1, (0, 255, 0), 2)
                    for i in range(4): cv2.line(img, tuple(img_pts_2d[i]), tuple(img_pts_2d[i+4]), (0, 255, 0), 2)
                    cv2.drawContours(img, [img_pts_2d[4:]], -1, (0, 255, 0), 2)
                    # tvec의 Z축 값이 바로 카메라와 번호판 사이의 거리(미터)입니다.
                    dist = tvec[2][0]
                    cv2.putText(img, f"Dist:{dist:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}", 
                                (int(rect[0][0]), int(rect[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # 최종 결과물이 그려진 이미지를 화면에 보여줍니다.
        cv2.imshow('RealSense Plate Detection', img)
        # 'q' 키를 누르면 루프를 빠져나옵니다.
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()