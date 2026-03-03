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

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

# 창 미리 생성
cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plate Detection", 1280, 720)

print("프로그램 시작... 글자 각도(OBB) 검출 모드 활성 (q: 종료)")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
            
        img = np.asanyarray(color_frame.get_data())
        orig_h, orig_w = img.shape[:2]

        # 1. SAHI 검출 (원본 + 0.5배)
        detections = []
        SCALES = [1.0, 0.5]          
        for scale in SCALES:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            proc_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
            
            results = get_sliced_prediction(
                proc_img, 
                detection_model, 
                slice_height=min(960, proc_img.shape[0]), 
                slice_width=min(960, proc_img.shape[1]), 
                verbose=0
            )

            scale_factor = 1.0 / scale
            for obj in results.object_prediction_list:
                b = obj.bbox
                x1, y1, x2, y2 = map(int, [b.minx*scale_factor, b.miny*scale_factor, b.maxx*scale_factor, b.maxy*scale_factor])
                detections.append({
                    'bbox': (x1, y1, x2, y2), 
                    'center': ((x1+x2)/2, (y1+y2)/2), 
                    'label': obj.category.name, 
                    'score': obj.score.value
                })

        # 2. 중복 제거 (NMS)
        detections.sort(key=lambda x: x['score'], reverse=True)
        unique_dets = []
        for det in detections:
            if not any(calculate_iou(det['bbox'], u['bbox']) > 0.4 for u in unique_dets):
                unique_dets.append(det)

        # 3. 근접 필터링
        final_boxes = [det for i, det in enumerate(unique_dets) if sum(1 for j, d2 in enumerate(unique_dets) if i!=j and get_distance(det['center'], d2['center']) < 150) >= 3]

        # 4. 각도 정밀 추출 및 회전 박스 시각화
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            
            # ROI 여유공간(Padding) 설정: 글자 각도 계산을 위해 주변을 조금 더 포함
            pad_w = int((x2 - x1) * 0.2)
            pad_h = int((y2 - y1) * 0.2)
            rx1, ry1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            rx2, ry2 = min(orig_w, x2 + pad_w), min(orig_h, y2 + pad_h)
            roi = img[ry1:ry2, rx1:rx2]

            if roi.size > 0:
                # 전처리: 그레이스케일 -> 가우시안 블러 -> 이진화
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 모폴로지 연산: 글자 조각들을 하나로 합침
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # ROI 내에서 가장 큰 덩어리(글자)를 선택
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # 노이즈가 아닌 유효한 크기일 때만 회전 사각형 계산
                    if cv2.contourArea(main_contour) > 50:
                        rect = cv2.minAreaRect(main_contour) # 중심, 크기, 각도 추출
                        box_points = cv2.boxPoints(rect)
                        box_points = np.int64(box_points)
                        
                        # 좌표 원본 복원
                        box_points[:, 0] += rx1
                        box_points[:, 1] += ry1
                        
                        # 회전된 초록색 사각형 그리기
                        cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 표시
            cv2.putText(img, f"{det['label']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Plate Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("프로그램이 안전하게 종료되었습니다.")