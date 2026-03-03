"""
이 코드는 원본영상을 2배줄여서 인식하도록 한 코드로 가까히서도 잘 인식되게 함 각글자들이 잘 인식
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
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ====================================================================================
# IoU 계산 함수
# ====================================================================================
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ====================================================================================
# 창 미리 생성 (화면이 안 뜨는 문제 해결 핵심!)
# ====================================================================================
cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)   # 창을 먼저 만들고 크기 조절 가능하게 설정
cv2.resizeWindow("Plate Detection", 1280, 720)          # 창 크기 고정

print("프로그램 시작... 카메라 연결 확인 중... (q로 종료)")

# ====================================================================================
# 메인루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("카메라 프레임 없음 - 재시도")
            continue
            
        img = np.asanyarray(color_frame.get_data())
        print(f"프레임 수신 완료 - 크기: {img.shape}")   # 디버깅용

        # ====================================================================================
        # 단순 2가지 스케일 (원본 + 5배 축소)
        # ====================================================================================
        detections = []
        SCALES = [1.0, 0.5]          
        orig_h, orig_w = img.shape[:2]

        for scale in SCALES:
            if scale == 1.0:
                proc_img = img.copy()
            else:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                proc_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # SAHI slice 크기를 현재 이미지 크기에 맞게 동적 조정 (작은 이미지에서 에러 방지)
            slice_h = min(960, proc_img.shape[0])
            slice_w = min(960, proc_img.shape[1])

            try:
                results = get_sliced_prediction(
                    proc_img,
                    detection_model,
                    slice_height=slice_h,
                    slice_width=slice_w,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.2,
                    verbose=0
                )
            except Exception as e:
                print(f"SAHI 오류 (scale={scale}): {e}")
                continue

            scale_factor = 1.0 / scale
            for obj in results.object_prediction_list:
                try:
                    bbox = obj.bbox.xyxy
                except:
                    bbox = obj.bbox.to_xyxy()
                x1, y1, x2, y2 = map(float, bbox)

                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)

                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'label': obj.category.name,
                    'score': obj.score.value
                })

        print(f"총 검출 수: {len(detections)}개")   # 디버깅용

        # ====================================================================================
        # IoU 기반 중복 제거
        # ====================================================================================
        detections.sort(key=lambda x: x['score'], reverse=True)
        unique_dets = []
        for det in detections:
            is_unique = True
            for u in unique_dets:
                if calculate_iou(det['bbox'], u['bbox']) > 0.50:
                    is_unique = False
                    break
            if is_unique:
                unique_dets.append(det)
        detections = unique_dets

        # ====================================================================================
        # 근접도 필터링
        # ====================================================================================
        final_boxes = []
        dist_threshold = 150 
        min_neighbors = 3  

        for i, det1 in enumerate(detections):
            neighbor_count = 0
            for j, det2 in enumerate(detections):
                if i == j: continue
                dist = get_distance(det1['center'], det2['center'])
                if dist < dist_threshold:
                    neighbor_count += 1
            if neighbor_count >= min_neighbors:
                final_boxes.append(det1)

        # ====================================================================================
        # 결과 시각화
        # ====================================================================================
        for det in final_boxes:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['label']} {det['score']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Plate Detection", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("프로그램 종료")