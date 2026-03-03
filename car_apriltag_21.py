"""
에어프릴태그 위치추종 테스트
"""
# ====================================================================================
# 필요 라이브러리
# ====================================================================================


# Open cv 이미지 처리를 담당함 예를들어 필터링, 마스킹, 그리기, 컴퓨터 비전 알고리즘Pnp의 라이브러리
import cv2
# Realsense 카메라 라이브러리 RGB 및 Depth 스트림을 가져오고 카메라의 렌저 정보 즉 내부 파라미터를 제어
import pyrealsense2 as rs
# 행렬 연산용, 모든 좌표 데이터를 배열 형태로 처리하기 위해 사용
import numpy as np
# SAHI(Slicing Aided Hyper Inference). 이미지를 쪼개서 검출하여 아주 작은 번호판 글자도 놓치지 않게 해줍니다.
from sahi.predict import get_sliced_prediction
# 삼각함수 연산용, 회전행렬을 우리가 아는 각도 단위로 변환할때 사용
import math


# ====================================================================================
# 실제 번호판 크기 및 3D 모델 설정
# ====================================================================================


# 실제 번호판의 가로 m단위
PLATE_WIDTH = 0.11
# 실제 번호판의 세로 m단위
PLATE_HEIGHT = 0.02

# 카메라의 이미지상에 그릴 초록상자의 두께 m단위, 나중에 위치각도 계산에 들어가지 않음
PLATE_DEPTH = 0.02

# 화면에 그릴 3D 그림 좌표축 길이 설정 m단위
AXIS = 0.05           


# 글자들을 하나의 번호판으로 묶을 때 사용할 거리 가중치
# 글자의 가로 길이에 7을 곱한 범위 안에 다른글자가 있으면 같은 번호판으로 인식함
GROUP_WIDTH_MULT = 7.0  
# 세로는 1.2배로 좁게 잡아 줄바꿈이 된 다른 물체와 섞이지 않게함
GROUP_HEIGHT_MULT = 1.2  

# ====================================================================================
# 지수 이동 평균 (Exponential Moving Average)
# ====================================================================================
# 박스가 파르르 떨리는 지터(Jitter) 현상을 방지하여 훨씬 부드럽게 보이게 합니다

# 이전 프레임의 위치/회전 값을 저장 (필터용)
prev_tvec, prev_rvec = None, None
# 현재 값을 80%, 이전 값을 20% 섞어서 부드럽게 만듦 (지터 방지)  , 반응 속도 (0.1~1.0, 높을수록 빠름)
alpha = 0.8 
# 좌표가 0.8m 이상 갑자기 튀면 무시함 (오류 방지)
JUMP_THRESHOLD = 0.8



# ====================================================================================
# 3D 모델 좌표계의 정의
# ====================================================================================
# PnP 알고리즘이 기준점으로 삼을 실제 세계의 3D 좌표 (중심이 0,0,0인 평면)
# 번호판의 정중앙을 0,0,0
# X 좌표: 왼쪽은 -절반, 오른쪽은 +절반으로 배치합니다.
# Y 좌표: 위쪽은 -절반, 아래쪽은 +절반으로 배치합니다.
# (컴퓨터 그래픽스 좌표계는 아래로 갈수록 Y가 커지기 때문입니다.)
# Z 좌표: 번호판 표면이므로 모두 0입니다.
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 좌상단
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 우상단
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0],   # 좌하단
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]    # 우하단
], dtype=np.float32)
# 나중에 카메라가 번호판의 네 모서리(2D)를 찾으면, 이 obj_points($3D$)와 1:1로 매칭
# 이 물체가 지금 오른쪽으로 15도 돌아가 있구나라는 것을 수학적으로 풀어냄





# ====================================================================================
# 회전 벡터 데이터를 3축 각도 Roll, Pitch, Yaw로 변화
# ====================================================================================
# PnP 알고리즘 결과값인 3차원 회전 벡터(rvec)를 3x3 회전 행렬(R)로 펼쳐줍니다.
def rotation_vector_to_euler(rvec):
    
    # 역할: PnP 알고리즘 결과값인 3차원 회전 벡터(rvec)를 3x3 회전 행렬(R)로 펼쳐줍니다.
    # 이유: rvec 자체는 숫자가 3개뿐이라 어느 방향으로 몇 도 회전했는지 바로 알기 어렵습니다. 이를 격자 형태의 행렬로 변환해야 각 축별 회전량을 계산할 수 있습니다.
    # PnP (Perspective-n-Point) 알고리즘 - 실제 번호판이 나로부터 몇미터 어느방향으로 몇도 틀어져 있는지 계산하는 기술
    # 필요한 정보 : 1. 실제 번호판의 크기, 2. 사진속 번호판의 4개 모서리 좌표, 3. 카메라 렌즈의 특징
    # Rodrigues (로드리게스) 변환
    # PnP 연산을 하고 나면 컴퓨터는 회전값을 rvec (숫자 3개)이라는 암호 형태로 줍니다.
    # 하지만 이 숫자 3개만 봐서는 사람이 "어디로 몇 도 돌았네?"라고 알 수가 없어요.
    # 그래서 cv2.Rodrigues 함수를 써서 이 암호를 3x3 크기의 표(행렬)로 쫙 펼쳐주는 겁니다.
    # 표로 정리해야 비로소 각 축(X, Y, Z)별로 얼마나 돌아갔는지 계산할 수 있는 상태가 됩니다.
    # https://searching-fundamental.tistory.com/73
    R, _ = cv2.Rodrigues(rvec)
    
    # 역할: "짐벌락(Gimbal Lock)"이라는 수학적 오류 상황인지 체크하기 위한 준비 단계입니다.
    # 의미: 회전 행렬의 특정 성분들을 이용해 코사인(cos) 값을 유도하는 과정입니다.
    # 우리는 각도를 측정할 때 보통 Roll(좌우 갸우뚱), Pitch(위아래 끄덕), Yaw(양옆 도리도리) 세 축을 씁니다.
    # 문제 발생: 예를 들어, 스마트폰을 위로 90도 딱 세웠다고 해봅시다(Pitch 90도).
    # 이때 스마트폰을 좌우로 흔드는 동작과 뱅글뱅글 돌리는 동작이 구분이 안 가고 축 두 개가 하나처럼 겹쳐버리는 순간이 옵니다.
    # 컴퓨터가 어 지금 이게 Roll이야, Yaw야? 하나가 없어졌어!"라며 혼란에 빠져서 값이 미친 듯이 튀거나 계산이 안 되는 현상을 짐벌락이라고 합니다.
    # https://m.blog.naver.com/ycpiglet/222941386101
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)

    # (일반적인 상황)
    # 역할: 아크탄젠트(atan2) 함수를 이용해 행렬 속에 숨어 있는 x, y, z축의 회전량을 뽑아냅니다.
    # x: Roll (좌우 기울기)
    # y: Pitch (위아래 기울기)
    # z: Yaw (좌우 회전/도리도리)
    # 1e-6의 의미: sy가 거의 0에 가깝지 않다면(즉, 수직으로 완전히 꺾인 특수 상황이 아니라면) 정상적으로 계산하라는 뜻입니다.
    if sy > 1e-6:
        x, y, z = math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
    
    # (특이 상황/짐벌락 상황)
    # 역할: 번호판이 카메라와 완전히 수직이 되거나 평행이 되어 각도 계산이 꼬일 때(나누기 0 오류 등),
    # 다른 공식을 써서 에러를 방지하는 안전장치입니다.
    else:
        x, y, z = math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0

    # 컴퓨터가 계산한 라디안(Radian) 단위를 우리가 익숙한 도(Degree, 0~360도) 단위로 바꿔서 반환합니다.
    return np.degrees([x, y, z])



# ====================================================================================
# 컴퓨터 속에 가상의 3D 번호판 상자 설계도를 그리는 함수
# ====================================================================================

def get_plate_box_points(w, h, d):

    # hw (Half Width): 가로의 절반
    # hh (Half Height): 세로의 절반
    hw, hh = w / 2, h / 2
    return np.float32([
        # 실제 번호판이 딱 붙어 있는 평면
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0],
        # 두께만큼 띄워서 윗면 4개에 점을 찍음
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d]
    ])

# ====================================================================================
# 모델 및 리얼센스 초기화
# ====================================================================================
# YOLOv8 같은 최신 AI 모델을 SAHI라는 기술(이미지를 쪼개서 정밀하게 보는 기술)로 실행할 수 있게 도와주는 도구를 가져옵니다.
# https://docs.ultralytics.com/ko/guides/sahi-tiled-inference/
from sahi.models.ultralytics import UltralyticsDetectionModel
# 미리 학습시켜둔 (.pt 파일)이 어디에 있는지 경로를 알려줍니다.
# 이 파일이 있어야 번호판 글자를 알아볼 수 있습니다.
model_path = '/home/limdoyeon/realsense_apriltag/runs/detect/EV_Plate_Master_v/weights/best.pt'
# AI 모델을 실제로 메모리에 올립니다.
# confidence_threshold=0.3: AI가 "이거 번호판 글자야!"라고 확신하는 확률이 30%만 넘어도 인정해주겠다는 뜻입니다.
# device="cuda:0": CPU 대신 훨씬 빠른 NVIDIA 그래픽카드(GPU)를 써서 계산하라는 명령입니다.
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=0.3, device="cuda:0")
# 카메라와 컴퓨터 사이의 데이터 통로(파이프라인)를 만듭니다.
pipeline = rs.pipeline()
# 카메라를 어떻게 켤지 '설정지'를 꺼냅니다.
config = rs.config()
# 구체적인 촬영 옵션을 정합니다.
# 1280, 720: HD 화질 해상도로 찍어라.
# bgr8: 색상 정보를 일반적인 이미지 형식으로 받아라.
# 30: 1초에 30장(30 FPS) 찍어라.
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 설정한 대로 카메라의 전원을 켜고 영상 데이터를 받기 시작합니다.
profile = pipeline.start(config)
# 방금 켠 카메라 렌즈의 고유 정보(인트린직, Intrinsics)를 읽어옵니다.
# 렌즈가 얼마나 휘었는지, 초점 거리는 얼마인지 등을 알아내는 겁니다.
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# 방금 켠 카메라 렌즈의 고유 정보(인트린직, Intrinsics)를 읽어옵니다.
# 렌즈가 얼마나 휘었는지, 초점 거리는 얼마인지 등을 알아내는 겁니다.
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
# 렌즈 특유의 볼록하게 휘어 보이는 현상(왜곡)을 보정하기 위한 숫자들입니다.
dist_coeffs = np.zeros(5)






# ====================================================================================
# 메인루프
# ====================================================================================
try:
    # 무한 반복 (프로그램 종료 전까지 계속 실행)
    while True:


        # ====================================================================================
        # 카메라 영상 받아오기 및 전처리
        # ====================================================================================
        # 카메라로부터 영상 데이터 뭉치를 기다렸다가 받음
        frames = pipeline.wait_for_frames()
        # 그 중에서 RGB(컬러) 이미지만 추출
        color_frame = frames.get_color_frame()
        # 영상이 제대로 안 들어왔으면 이번 루프는 건너뜀
        if not color_frame: continue
        # 카메라 원본 데이터를 처리하기 쉬운 숫자 배열로 변환
        img_bgr = np.asanyarray(color_frame.get_data())
        # 색상 인식이 쉬운 HSV 색공간으로 변환
        # https://m.blog.naver.com/ycpiglet/222229696956
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        

        # ====================================================================================
        # 마스킹 로직, 색깔 마스크맵
        # ====================================================================================
        # 파란색 범위에 해당하는 부분만 흰색으로 표시하는 마스크(지도) 생성
        mask_blue = cv2.inRange(hsv, np.array([90, 65, 90]), np.array([130, 255, 255]))
        # 검은색 범위에 해당하는 부분만 흰색으로 표시하는 마스크 생성
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 115, 130]))
        # 파란색 영역을 6번 팽창(두껍게)시켜서 글자 주변 영역을 확보
        blue_dilated = cv2.dilate(mask_blue, np.ones((5,5), np.uint8), iterations=6)
        # 파란색 주변에 있는 검은색만 글자로 인정 (배경의 다른 검은색 무시)
        mask_black = cv2.bitwise_and(mask_black, blue_dilated)
        # 파란색 마스크와 검은색 마스크를 합침
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        # 원본 이미지에서 마스크가 흰색인 부분(번호판 후보)만 추출하고 나머지는 검게 지움
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_combined)
        
        # AI 모델(YOLO)을 써서 번호판 글자들을 찾음. 이미지를 960x960 크기로 쪼개서 정밀 검사함 (SAHI)
        results = get_sliced_prediction(img_hsv_blue, detection_model, slice_height=960, slice_width=960, verbose=0)



        # ====================================================================================
        # 찾은 글자의 정밀 좌표 추출
        # ====================================================================================
        # 이번 한 프레임(찰나의 순간) 동안 카메라에 포착된 모든 글자들의 정밀한 위치 정보를 모아두기 위해 빈 리스트(바구니)를 준비합니다
        detections = []
        # 앞서 예측한 명단에서 글자 후보를 하나씩 꺼내어 정밀 검사를 시작합니다
        for obj in results.object_prediction_list:
            # AI가 글자를 감싸고 있다고 판단한 사각형의 네 군데 좌표(왼쪽 끝, 위 끝, 오른쪽 끝, 아래 끝)를 가져옵니다
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            # AI는 좌표를 소수점 단위로 아주 정밀하게 주지만, 실제 이미지의 픽셀은 소수점이 없으므로 계산하기 좋게 정수(Integer)로 변환
            x1, y1, x2, y2 = map(int, bbox)



            # ====================================================================================
            # 글자 주변 도려내기 (ROI 설정)
            # ====================================================================================
            # 글자 테두리를 너무 바짝 깎으면 글자의 외곽선이 잘릴 수 있으므로, 상하좌우로 5픽셀씩 여유 공간을 더 줍니다.
            pad = 5
            # 전체 화면(1280x720) 밖으로 나가지 않도록 범위를 조절하면서, 글자 하나가 들어있는 작은 사각형 영역의 좌표를 최종 결정
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            # 원본 이미지에서 방금 계산한 좌표만큼만 칼로 도려내듯 잘라내어 '작은 글자 조각 이미지'를 만듭니다. 이제 이 작은 조각만 집중적으로 분석합니다.
            roi = img_bgr[ry1:ry2, rx1:rx2]
            

            # 혹시라도 잘라낸 이미지가 텅 비어있지는 않은지 확인하여 오류를 방지합니다.
            if roi.size > 0:
                # 색상 정보는 무시하고 오직 글자의 '모양'에만 집중하기 위해 흑백 이미지로 바꿉니다.
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # '오츠(Otsu)' 알고리즘을 써서 배경은 완벽한 검은색으로, 글자 형태는 완벽한 흰색으로 바꿉니다. 글자가 비스듬하게 있든 빛이 번졌든 상관없이 글자 형태만 선명하게 남기는 작업입니다.
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # 흰색으로 변한 글자 덩어리의 '바깥쪽 테두리(외곽선)' 라인을 수학적 점들로 모두 찾아냅니다.
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                
                
                # ====================================================================================
                # 기울어진 사각형(OBB) 계산
                # ====================================================================================
                if contours:
                    # 글자 조각 안에 먼지 같은 작은 점들이 섞여 있을 수 있습니다.
                    # 그중 가장 큰 덩어리가 진짜 글자일 것이므로 면적이 가장 큰 테두리를 선택합니다.
                    main_cnt = max(contours, key=cv2.contourArea)
                    # 선택한 덩어리가 너무 작다면(20픽셀 미만) 글자가 아닐 확률이 높으므로 무시하고 넘어갑니다.
                    if cv2.contourArea(main_cnt) > 20:
                        # 테두리를 감싸는 사각형을 만들되, 똑바로 서 있는 사각형이 아니라 
                        # 글자의 기울기에 맞춰 가장 꽉 끼게 설계된 최소 면적 사각형을 계산
                        rect = cv2.minAreaRect(main_cnt)
                        # 기울어진 사각형의 네 꼭짓점 좌표를 얻어냅니다.
                        box_pts = cv2.boxPoints(rect)
                        # 작은 조각 이미지 안에서 찾은 좌표이므로, 다시 전체 화면(1280x720) 기준의 위치로 좌표값을 더해줍니다.
                        box_pts[:, 0] += rx1; box_pts[:, 1] += ry1
                        
                        # 좌표 정렬: [좌상, 우상, 좌하, 우하]
                        # 네 꼭짓점이 어떤 게 위인지 아래인지 모르기 때문에, 높이($y$) 값을 기준으로 순서대로 나열합니다.
                        pts = sorted(box_pts, key=lambda x: x[1])
                        # 위쪽에 있는 두 점 중 왼쪽($x$가 작은 것)을 좌상단, 오른쪽을 우상단으로 정합니다.
                        top = sorted(pts[:2], key=lambda x: x[0])
                        # 아래쪽에 있는 두 점 중 왼쪽을 좌하단, 오른쪽을 우하단으로 정합니다.
                        bottom = sorted(pts[2:], key=lambda x: x[0])
                        # 수학 계산(PnP)에 바로 쓸 수 있게 [좌상, 우상, 좌하, 우하] 순서로 예쁘게 포장합니다.
                        obb_corners = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)
                        # 글자의 중심 위치, 정밀한 네 꼭짓점, 원래 크기 등을 한데 묶어 맨 처음 만든 바구니에 저장합니다.
                        detections.append({
                            'center': ((x1+x2)/2, (y1+y2)/2),
                            'obb_corners': obb_corners,
                            'w': x2-x1, 'h': y2-y1
                        })

        # ====================================================================================
        # PnP계산
        # ====================================================================================
        

        # 화면에 글자가 여러 개 있을 때, 어떤 글자들이 같은 번호판에 붙어 있는지 판단하는 과정입니다.
        # 찾은 글자 개수만큼 '확인 완료' 체크리스트를 만듭니다. 이미 번호판으로 묶인 글자를 또 계산하지 않기 위해서입니다.
        visited = [False] * len(detections)
        # 첫 번째 글자부터 하나씩 기준점으로 잡고 검사를 시작합니다.
        for i in range(len(detections)):
            # 이미 다른 번호판 그룹에 들어간 글자라면 그냥 지나칩니다.
            if visited[i]: continue
            # 새로운 번호판 그룹 바구니를 만들고, 기준이 되는 글자를 먼저 넣은 뒤 '확인 완료' 표시를 합니다.
            group = [detections[i]]; visited[i] = True
            # 나머지 다른 글자들을 하나하나 대조해 봅니다.
            for j in range(i+1, len(detections)):
                if not visited[j]:
                    # 두 글자 사이의 가로 거리를 잽니다.
                    dx = abs(detections[i]['center'][0] - detections[j]['center'][0])
                    # 두 글자 사이의 세로 거리를 잽니다.
                    dy = abs(detections[i]['center'][1] - detections[j]['center'][1])
                    # 가로 거리가 글자 폭의 몇배 이내이고, 세로 거리가 글자 높이의 몇배 이내라면 같은 번호판이구나라고 판단합니다.
                    if dx < (detections[i]['w'] * GROUP_WIDTH_MULT) and dy < (detections[i]['h'] * GROUP_HEIGHT_MULT):
                        # 한 팀으로 판명된 글자를 그룹 바구니에 추가하고 '확인 완료' 표시를 합니다.
                        group.append(detections[j]); visited[j] = True
            
            # 글자가 최소 3개는 모여야 "이건 진짜 번호판이다"라고 인정하고 계산에 들어갑니다.
            if len(group) >= 3:
                # 번호판 글자들을 왼쪽부터 오른쪽 순서대로 예쁘게 나열합니다.
                group.sort(key=lambda x: x['center'][0])
                # 화면에 개별 글자들의 테두리를 얇은 초록색 선으로 그려서 잘 찾았는지 보여줍니다.
                for item in group:
                    cv2.drawContours(img_hsv_blue, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)
                # 번호판 전체의 네 모서리 좌표를 결정합니다.
                # 맨 왼쪽 글자의 좌상단, 맨 오른쪽 글자의 우상단, 맨 왼쪽 글자의 좌하단, 맨 오른쪽 글자의 우하단을 뽑아 2D 이미지상의 번호판 사각형을 완성합니다.
                img_pts_2d = np.array([
                    group[0]['obb_corners'][0], group[-1]['obb_corners'][1],
                    group[0]['obb_corners'][2], group[-1]['obb_corners'][3]
                ], dtype=np.float32)
                

                # 우리가 미리 정한 실제 번호판 크기(obj_points)와 방금 찾은 사진 속 좌표(img_pts_2d)를 비교해서, 카메라로부터 얼마나 떨어져 있는지(tvec)와 어느 방향으로 꺾여 있는지(rvec)를 찾아냅니다.
                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    # 이전 프레임에서 계산했던 위치값이 있다면 비교를 시작합니다.
                    if prev_tvec is not None:
                        # 방금 계산한 위치와 조금 전 위치 사이의 거리를 계산합니다.
                        diff = np.linalg.norm(tvec - prev_tvec)
                        # 이 거리가 80cm 이내라면 정상적인 움직임으로 보고 필터를 적용합니다.
                        if diff < JUMP_THRESHOLD:
                            # 새로운 값과 옛날 값을 8:2 비율로 섞어서 좌표가 파르르 떨리는 것을 막고 부드럽게 움직이게 만듭니다.
                            tvec = alpha * tvec + (1 - alpha) * prev_tvec
                            # 다음 프레임에서 또 써먹기 위해 현재 값을 저장해둡니다.
                            rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    
                    prev_tvec, prev_rvec = tvec, rvec

                    # 복잡한 행렬 데이터를 우리가 읽을 수 있는 m단위 거리와 도(degree) 단위 각도로 변환합니다.
                    tx, ty, tz = tvec.flatten()
                    roll, pitch, yaw = rotation_vector_to_euler(rvec)
                    # 최종 위치와 각도값을 터미널 창에 실시간으로 출력합니다.
                    print(f"Pos: {tx:.3f}, {ty:.3f}, {tz:.3f} | Rot: {roll:.1f}, {pitch:.1f}, {yaw:.1f}")

                    # 번호판 정중앙에 빨강(X), 초록(Y), 파랑(Z) 축을 그려서 어느 쪽이 정면인지 보여줍니다.
                    cv2.drawFrameAxes(img_hsv_blue, K, dist_coeffs, rvec, tvec, AXIS)
                    # 입체 상자를 만들기 위한 3D 설계도를 가져옵니다.
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    # 계산된 위치와 각도를 바탕으로, 3D 상자가 카메라 화면(2D) 어디에 그려져야 할지 위치를 찍습니다.
                    projected_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    # 찍힌 점들을 선으로 연결하여 화면에 입체적인 초록색 상자를 완성합니다.
                    cv2.drawContours(img_hsv_blue, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4):
                        cv2.line(img_hsv_blue, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_hsv_blue, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("OBB-based PnP Detection", img_hsv_blue)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()