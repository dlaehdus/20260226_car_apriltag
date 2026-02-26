from ultralytics import YOLO

model = YOLO('yolo11s.pt') 

yaml_path = "/home/limdoyeon/realsense_apriltag/src/LicensePlate/data.yaml"

if __name__ == '__main__':
    model.train(
        data=yaml_path,
        epochs=1000,
        imgsz=1280,           
        # --- [메모리 해결 핵심 변경점] ---
        batch=4,              # 8에서 4로 줄였습니다. (메모리 사용량 절반 감소)
        amp=True,             # 자동 혼합 정밀도 사용 (메모리 절약 및 속도 향상)
        workers=4,            # 데이터 로딩 스레드 수를 줄여 시스템 부하 감소
        
        # --- [정밀도 및 증강 유지] ---
        device=0,
        degrees=45.0,         
        translate=0.3,        
        scale=0.9,            
        shear=20.0,           
        perspective=0.001,    
        mosaic=1.0,           
        mixup=0.5,            
        copy_paste=0.5,       
        hsv_h=0.02,           
        hsv_s=0.9,            
        hsv_v=0.6,            
        optimizer='SGD',      
        lr0=0.01,             
        lrf=0.001,            
        close_mosaic=50,      
        multi_scale=False,    # 메모리 부족 시 True보다 False가 안전합니다.
        
        name='EV_Plate_Master_v'
    )