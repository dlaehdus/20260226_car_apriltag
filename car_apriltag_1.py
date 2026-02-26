"""
전 버전
인식률이 낮아 번호판을 찾지 못하는 문제가 발생함 조금더 이미지 전처리 기법을 견고하게 설정할것임
이번 버전
90번줄부터 전처리 기법을 견고하게 하여 실제로 번호판을 인식하였으나 거리가 멀어지거나 가까워짐에 따라
인식률이 급격하게 변화함
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
PLATE_W = 0.520
PLATE_H = 0.110

# ====================================================================================
# 번호판의 3D 모델 좌표 (모서리 점들)
# ====================================================================================
obj_pts = np.float32([
    [-PLATE_W/2, -PLATE_H/2, 0],   # 좌상
    [ PLATE_W/2, -PLATE_H/2, 0],   # 우상
    [ PLATE_W/2,  PLATE_H/2, 0],   # 우하
    [-PLATE_W/2,  PLATE_H/2, 0]    # 좌하
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
try:
    # 설정을 적용하여 카메라 작동을 시작합니다.
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"에러 발생: {e}\n리얼센스 뷰어가 실행 중이라면 반드시 종료 후 다시 실행하세요.")
    exit()
# 카메라 렌즈의 고유 파라미터(왜곡, 초점거리 등)를 가져옵니다.
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# 카메라 내부 행렬(Intrinsic Matrix)을 구성합니다. 3D를 2D로 투영할 때 핵심이 됩니다.
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float32)
# 렌즈 왜곡 계수입니다. 여기서는 왜곡이 없다고 가정(0)하고 설정했습니다.
dist_coeffs = np.zeros(5)
print("q 키로 종료")


# ====================================================================================
# 회전 행렬 변환 함수
# ====================================================================================
# 3x3 회전 행렬 $R$을 입력받아 Roll, Pitch, Yaw 각도를 계산하는 함수입니다.
def get_rpy(R):
    # 삼각함수(atan2)를 이용해 각 축의 회전량을 도(Degree) 단위로 추출합니다.
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
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
        if not color_frame:
            continue
        # 리얼센스 데이터를 넘파이 배열(OpenCV 형식)로 변환합니다.
        img = np.asanyarray(color_frame.get_data())
        # 연산 속도를 높이기 위해 흑백 이미지로 바꿉니다.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """
        전 버전에 비해 개선한 부분
        """
        # 이미지를 일정한 구역(8x8)으로 나누어 각 구역별로 대비를 높이는 기법입니다.
        # 전체 화면의 밝기를 한꺼번에 조절하면 너무 밝거나 어두운 곳의 정보가 날아가는데
        # CLAHE는 구역별로 처리하므로 어두운 곳에 숨은 번호판의 형태를 더 뚜렷하게 살려줍니다.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 원본 이미지에서 닫힘(Closing) 연산 결과를 뺀 것으로, 배경보다 어두운 부분(번호판의 글자)만 추출합니다.
        # 한국 번호판은 밝은 바탕에 검은 글씨이므로, 이 연산을 수행하면 배경은 사라지고 검은색 글자들만 하얗게 둥둥 뜬 상태가 됩니다.
        # (13x5 커널은 긴 번호판 비율에 맞춘 것입니다.)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # 팽창(Dilate) 후 침식(Erode)을 수행하여 객체 내부의 작은 구멍을 메우는 연산입니다
        # 개별 글자들 사이에 미세하게 끊어진 부분들을 연결하여 나중에 하나의 큰 사각형 덩어리로 인식될 수 있도록 돕습니다.
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, sq_kernel)

        # 흑백 이미지를 완전한 검은색(0)과 하얀색(255)으로 나눕니다
        # OTSU 알고리즘은 사용자가 수치를 직접 입력하지 않아도 이미지의 명암 분포를 분석해 가장 적절한 경계값(Threshold)을 자동으로 찾아줍니다.
        # 결과적으로 강조된 글자 부분만 하얗게 남습니다.
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 하얀색 영역을 사방으로 확장시키는 연산입니다.
        # 흩어져 있던 글자 알갱이들을 더 크게 불려서 서로 맞붙게 만듭니다.
        # 이렇게 하면 개별 글자가 아닌 하나의 커다란 '번호판 덩어리'가 되어, 나중에 findContours 함수로 사각형 영역을 찾기가 훨씬 수월해집니다.
        thresh = cv2.dilate(thresh, None, iterations=2)


        # ====================================================================================
        # 번호판 후보 검출 및 좌표 정렬
        # ====================================================================================
        # 이진화된 이미지에서 하얀색 덩어리들의 외곽 경계선(윤곽선)을 모두 찾아냅니다.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 찾아낸 수많은 윤곽선 중 면적이 큰 순서대로 20개만 남기고 나머지는 버립니다.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        # 찾아낸 모든 윤곽선을 하나씩 검사합니다.
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 남은 20개 중에서도 면적이 1200픽셀보다 작은 것은 무시합니다. (너무 멀리 있거나 무의미한 조각들 제외)
            if area < 1200:
                continue
            # 윤곽선의 전체 둘레 길이를 계산합니다.
            peri = cv2.arcLength(cnt, True)
            # 복잡한 선을 직선 위주의 다각형으로 단순화합니다.
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            # 꼭짓점이 4개(사각형)이고 면적이 일정 크기 이상인 것만 골라냅니다.
            if len(approx) != 4:
                continue
            # 오목한 부분이 없는 볼록한(Convex) 사각형인지 확인합니다.
            if not cv2.isContourConvex(approx):
                continue
            # 윤곽선을 감싸는 최소 크기의 회전된 사각형을 구하고, 가로와 세로 길이를 추출합니다.
            rot_rect = cv2.minAreaRect(cnt)
            w, h = rot_rect[1]
            if w < h:
                w, h = h, w
            aspect_ratio = w / h
            # 가로 대비 세로 비율이 한국 표준 번호판(대략 4.7)과 유사한지 확인합니다.
            # 이 과정을 통해 정사각형에 가까운 표지판이나 길쭉한 기둥 같은 오검출 대상을 걸러냅니다.
            if not (3.8 < aspect_ratio < 5.6):
                continue
            # 근사화된 4개 꼭짓점을 2차원 좌표 배열로 변환하고, 정렬된 점을 담을 빈 그릇(rect)을 준비합니다.
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = np.zeros((4, 2), dtype="float32")
            # $(x, y)$ 좌표의 합이 가장 작은 점은 왼쪽 위에 위치하고($0+0$), 가장 큰 점은 오른쪽 아래에 위치한다는 원리를 이용합니다.
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]      # 좌상
            rect[2] = pts[np.argmax(s)]      # 우하
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]   # 우상
            rect[3] = pts[np.argmax(diff)]   # 좌하

            # ====================================================================================
            # PnP 알고리즘을 통한 3D 자세 추정 및 시각화 에어프릴의 원리와 동일
            # ====================================================================================
            # 가장 핵심 줄입니다. 실제 크기(obj_pts)와 화면 속 크기(rect)를 비교해 카메라와의 상대적 위치(tvec)와 회전(rvec)을 계산합니다.
            success, rvec, tvec = cv2.solvePnP(obj_pts, rect, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            # 수학적 계산이 실패했을 경우(값이 발산하거나 해를 찾지 못할 때) 오류를 방지하기 위해 다음 단계로 넘어가지 않고 이번 루프를 건너뜁니다.
            if not success:
                continue
            # 로드리게스(Rodrigues) 변환을 사용하여, 3x1 크기의 회전 벡터(rvec)를 계산하기 용이한 3x3 회전 행렬(R) 형식으로 바꿉니다.
            R, _ = cv2.Rodrigues(rvec)
            # 행렬 형태의 회전 값을 사람이 이해하기 쉬운 도(degree) 단위 각도로 변환하고, 이동 벡터에서 실제 거리 값($x, y, z$)을 평평하게 펴서 추출합니다.
            roll, pitch, yaw = get_rpy(R)
            x, y, z = tvec.flatten()
            # 번호판의 정중앙에 3D 좌표축(X-빨강, Y-초록, Z-파랑)을 화살표로 그려넣습니다
            cv2.drawFrameAxes(img, K, dist_coeffs, R, tvec, PLATE_W * 0.6)
            # 번호판 면을 기준으로 뒤쪽으로 5cm(thickness)만큼 두께가 있는 입체 박스의 3차원 꼭짓점 8개를 가상으로 설정
            thickness = 0.05
            box_3d = np.float32([
                [-PLATE_W/2, -PLATE_H/2, 0], [PLATE_W/2, -PLATE_H/2, 0],
                [PLATE_W/2, PLATE_H/2, 0], [-PLATE_W/2, PLATE_H/2, 0],
                [-PLATE_W/2, -PLATE_H/2, -thickness], [PLATE_W/2, -PLATE_H/2, -thickness],
                [PLATE_W/2, PLATE_H/2, -thickness], [-PLATE_W/2, PLATE_H/2, -thickness]
            ])
            # 위에서 만든 가상의 3D 점들을 현재 카메라 시점(각도와 거리)에 맞춰서 2D 이미지 평면의 어디에 찍혀야 하는지 계산(투영, Projection)합니다.
            img_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
            img_pts = np.int32(img_pts).reshape(-1, 2)
            # 투영된 점들을 서로 연결하여 번호판을 감싸는 초록색 입체 박스를 화면에 그려넣습니다.
            cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), 3)
            for i in range(4):
                cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i + 4]), (0, 255, 0), 3)
            cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 0), 3)
            # 번호판의 왼쪽 상단 꼭짓점 위쪽에 실시간 거리($Z$)와 각도($R, P, Y$) 수치를 노란색 글자로 출력합니다.
            info = f"Plate Z:{z:.2f}m R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
            cv2.putText(img, info, (int(rect[0][0]), int(rect[0][1]) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            print(f"번호판 검출 Pos(x,y,z): {x:.2f},{y:.2f},{z:.2f} | RPY: {roll:.1f},{pitch:.1f},{yaw:.1f}")

        # 결과 화면
        cv2.imshow('D415 Korean License Plate Pose (AprilTag-style)', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
