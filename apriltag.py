# 리얼센스 D415 카메라로부터 받은 영상에서 에어프릴태그를 찾아내고 수학적인 계산을 통해 3차원 거리와 회전각을 시각화하는 코드

import pyrealsense2 as rs           # 리얼센스 카메라 제어 SDK
import numpy as np                  # 수치 계산 및 행렬 처리
import cv2                          # 이미지 처리 및 화면 출력 (OpenCV)
import math                         # 삼각함수 등 수학 계산
from dt_apriltags import Detector   # 에이프릴태그 검출 엔진

# ==========================================
# 1. 실제 출력한 태그 크기로 수정
# ==========================================
TAG_SIZE = 0.03                      # 미터단위

def get_cube_points(tag_size):      # 3D 큐브의 꼭짓점을 정의하는 함수 (태그를 바닥으로 하는 정육면체)
    s = tag_size / 2
    return np.float32([
        [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],                                 # 바닥면 4개 점
        [-s, -s, -tag_size], [s, -s, -tag_size], [s, s, -tag_size], [-s, s, -tag_size]  # 윗면 4개 점
    ])

# 리얼센스 D415 카메라 초기화
pipeline = rs.pipeline()                                                                # 카메라 데이터 흐름(통로) 생성
config = rs.config()                                                                    # 카메라 설정 객체 생성
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)                    # 컬러 영상(1280x720, 30fps) 활성화

try:
    profile = pipeline.start(config)        # 설정한 내용으로 스트리밍 시작
except RuntimeError as e:
    print(f"에러 발생: {e}\n리얼센스 뷰어가 켜져있다면 반드시 끄고 실행하세요.")
    exit()

# 카메라 내장 파라미터(Intrinsic) 자동 획득
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)

# 3. 에이프릴태그 검출기 설정 (에러 방지를 위한 파라미터 튜닝)
at_detector = Detector(
    families='tag36h11',            # 인식할 태그 종류 (가장 일반적인 형태)
    nthreads=4,                     # 처리에 사용할 CPU 코어 수
    quad_decimate=2.0,              # 이미지를 1/2로 줄여서 분석 (속도 향상 및 노이즈 제거)
    quad_sigma=0.8,                 # 가우시안 블러로 노이즈 제거
    refine_edges=1,                 # 태그 테두리를 더 정밀하게 찾음
    decode_sharpening=0.25          # 인식 성능 향상을 위한 선명도 조절
)

print("프로그램 시작... 'q'를 누르면 종료됩니다.")

try:
    while True:
        # 프레임 받기
        frames = pipeline.wait_for_frames()             # 카메라로부터 새로운 프레임이 올 때까지 대기
        color_frame = frames.get_color_frame()          # 컬러 이미지 데이터 추출
        if not color_frame:
            continue

        # 이미지 변환
        img = np.asanyarray(color_frame.get_data())     # 데이터를 넘파이 배열(이미지)로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 인식 속도를 높이기 위해 흑백으로 변환

        # 실제 태그를 찾는 핵심 코드. estimate_tag_pose=True가 좌표와 각도를 계산하라는 옵션입니다.
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE)

        for tag in tags:
            R = tag.pose_R  # 3x3 회전 행렬 (태그가 어느 방향으로 기울었는지)
            t = tag.pose_t  # 3x1 이동 벡터 (카메라로부터 얼마나 떨어졌는지 x, y, z)

            # OpenCV 시각화 함수들에 필요한 카메라 매트릭스(K) 구성
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            dist_coeffs = np.zeros(5) # D415의 컬러 영상은 이미 보정되어 나오므로 0으로 둡니다.

            # 태그 중심에 빨강(X), 초록(Y), 파랑(Z) 화살표를 그립니다.
            cv2.drawFrameAxes(img, K, dist_coeffs, R, t, TAG_SIZE * 0.8)

            # 3D 초록색 상자 그리기
            rvec, _ = cv2.Rodrigues(R)                          # 3x3 행렬인 R을 OpenCV가 쓰는 회전 벡터 형식으로 변환
            cube_points = get_cube_points(TAG_SIZE)             # 아까 정의한 8개 3D 점들 생성
            img_pts, _ = cv2.projectPoints(cube_points, rvec, t, K, dist_coeffs)    # 3D 점들을 2D 화면 좌표로 투영
            img_pts = np.int32(img_pts).reshape(-1, 2)

            # # 선 긋기: 바닥면, 기둥, 윗면을 순서대로 연결하여 상자 완성 (초록색)
            cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), 2)
            # 수직 기둥 4개
            for i in range(4):
                cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 2)
            # 윗면 테두리
            cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 0), 2)

            # --- 좌표 및 롤, 피치, 요 계산 ---
            # 복잡한 회전 행렬 R을 우리가 이해하기 쉬운 도(degree) 단위 각도로 변환합니다.
            x, y, z = t.flatten()
            pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
            yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
            roll = math.degrees(math.atan2(-R[1, 0], R[0, 0]))

            # 화면 상단에 ID, 거리, 각도를 텍스트로 표시
            info = f"ID:{tag.tag_id} Z:{z:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
            cv2.putText(img, info, (int(tag.center[0]), int(tag.center[1])-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            print(f"ID:{tag.tag_id} | Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        # 결과 화면 보기
        cv2.imshow('D415 AprilTag 3D', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 종료 시 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()