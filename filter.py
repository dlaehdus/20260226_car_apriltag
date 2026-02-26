import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 리얼센스 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 깊이(depth) 스트림도 필요하면 추가 가능
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 스트리밍 시작
print("리얼센스 스트리밍 시작...")
pipeline.start(config)

# 각 필터 결과를 담을 리스트
processed_frames = []
# 필터 이름 리스트
filter_names = []

try:
    while True:
        # 프레임 대기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame() # 깊이 프레임 사용 시

        if not color_frame: # or not depth_frame:
            continue

        # 넘파이 배열로 변환 (원본)
        img_bgr = np.asanyarray(color_frame.get_data())

        # --- 필터 적용 ---
        processed_frames.clear() # 이전 프레임 결과 초기화
        filter_names.clear()

        # 1. 원본
        processed_frames.append(img_bgr.copy())
        filter_names.append("Original")

        # 2. 흑백 필터 (Grayscale)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        processed_frames.append(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)) # 3채널로 변환하여 합치기 용이하게
        filter_names.append("Grayscale")

        # 3. Canny 엣지 검출 (Edge Detection)
        img_canny = cv2.Canny(img_gray, 100, 200) # (원본 흑백, 낮은 임계값, 높은 임계값)
        processed_frames.append(cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR))
        filter_names.append("Edge Detection")

        # 4. HSV 색상 강조 (Blue Color Mask - 전기차용)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 60, 50])  # 채도를 60으로 올려서 흰색 혼입 방지
        upper_blue = np.array([125, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        # 노이즈 제거 (Morphology Open): 작은 흰색 점(노이즈)들을 지워버림
        kernel = np.ones((3,3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_blue)
        img_hsv_blue = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_blue)
        processed_frames.append(img_hsv_blue)
        filter_names.append("Blue Mask (Electric)")
        
        # 5. 블러 (Gaussian Blur - 노이즈 제거)
        img_blur = cv2.GaussianBlur(img_bgr, (5, 5), 0) # (커널 크기, 시그마)
        processed_frames.append(img_blur)
        filter_names.append("Gaussian Blur")

        # 6. 이진화 (Binary Threshold - 흑백 이미지의 대비를 극대화)
        _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # (임계값, 최대값, 타입)
        processed_frames.append(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR))
        filter_names.append("Binary Threshold")

        # 7. 모폴로지 연산 (Dilation - 객체 확장)
        kernel = np.ones((5,5), np.uint8) # 커널 정의
        img_dilation = cv2.dilate(img_thresh, kernel, iterations=1) # 이진화 이미지에 적용
        processed_frames.append(cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR))
        filter_names.append("Dilation")
        
        # 8. 색상 역전 (Invert Color - 명암 반전)
        img_invert = cv2.bitwise_not(img_bgr)
        processed_frames.append(img_invert)
        filter_names.append("Invert Color")

        # --- 화면 병합 (2x4 격자) ---
        # 한 줄에 4개씩, 총 2줄
        
        # 첫 번째 줄
        top_row_images = processed_frames[0:4]
        # 두 번째 줄
        bottom_row_images = processed_frames[4:8]

        # 모든 이미지는 같은 크기여야 합니다 (640x480)
        # 텍스트 추가
        for i, img in enumerate(top_row_images):
            cv2.putText(img, filter_names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, img in enumerate(bottom_row_images):
            cv2.putText(img, filter_names[i+4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 가로로 합치기
        top_combined = np.hstack(top_row_images)
        bottom_combined = np.hstack(bottom_row_images)
        
        # 세로로 합치기
        combined_view = np.vstack((top_combined, bottom_combined))

        # --- 결과 표시 ---
        # 너무 크면 창 크기 조정 (선택 사항)
        # display_img = cv2.resize(combined_view, (1280, 720)) # 1280x720 픽셀 창 크기
        
        cv2.imshow('RealSense Multi-Filter View (Press Q to quit)', combined_view)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 스트리밍 중지 및 창 닫기
    print("스트리밍 중지 및 리소스 해제...")
    pipeline.stop()
    cv2.destroyAllWindows()