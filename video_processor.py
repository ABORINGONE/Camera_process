# video_processor.py (调试版本)
import time
import cv2
import requests
import numpy as np
from ultralytics import YOLO
from config import VIDEO_STREAM_URL

# --- YOLOv8 模型配置 ---
print("[视频线程] 正在加载 YOLOv8n 模型...")
try:
    model = YOLO('yolov8n.pt') 
    print("[视频线程] 模型加载成功。")
except Exception as e:
    print(f"[视频线程] 致命错误：YOLOv8模型加载失败: {e}")
    print("请确保 'ultralytics' 库已成功安装 (pip install ultralytics)。")
    model = None

# --- 调试设置 ---
# 暂时将目标类别设置为 None，让模型显示所有识别到的物体。
# 当您确认模型工作正常后，可以再把它改回 "bottle"。
TARGET_CLASS_NAME = None 
# 降低置信度阈值，让模型更容易报告检测结果。
CONFIDENCE_THRESHOLD = 0.25

def run_video_processing(shared_state, lock):
    """
    在一个独立线程中运行，负责连接视频流，使用YOLOv8模型检测物体，并更新共享的坐标。
    """
    if model is None:
        print("[视频线程] 由于模型未能加载，线程无法启动。")
        with lock:
            shared_state['running'] = False
        return

    print("[视频线程] 正在连接视频流...")
    try:
        stream = requests.get(VIDEO_STREAM_URL, stream=True, timeout=10)
        if stream.status_code != 200:
            print(f"[视频线程] 错误：无法连接到视频流，状态码: {stream.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"[视频线程] 错误：连接视频流失败: {e}")
        with lock:
            shared_state['running'] = False
        return

    print("[视频线程] 视频流连接成功。")
    bytes_data = b''

    while shared_state.get('running', True):
        try:
            for chunk in stream.iter_content(chunk_size=1024):
                if not shared_state.get('running', True):
                    break

                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    if jpg:
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        
                        # 1. 使用YOLOv8进行预测，并传入我们设置的较低置信度
                        results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False) 

                        found_object = False
                        best_target = None 
                        
                        # 从结果中获取第一个（也是唯一一个）结果对象
                        r = results[0]
                        boxes = r.boxes
                        
                        # --- 新增：在终端打印检测到的物体数量 ---
                        print(f"\r[视频线程] 当前帧检测到 {len(boxes)} 个物体。", end="")

                        # 2. 解析结果
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]

                            # 检查是否是我们想要的目标 (现在会匹配所有物体，因为 TARGET_CLASS_NAME is None)
                            if TARGET_CLASS_NAME is None or class_name == TARGET_CLASS_NAME:
                                if best_target is None or conf > best_target['conf']:
                                    best_target = {
                                        'box': box,
                                        'conf': conf,
                                        'class_name': class_name
                                    }
                        
                        # 3. 如果找到了目标，处理最佳结果
                        if best_target is not None:
                            found_object = True
                            x1, y1, x2, y2 = map(int, best_target['box'].xyxy[0])
                            cX = x1 + (x2 - x1) // 2
                            cY = y1 + (y2 - y1) // 2

                            with lock:
                                shared_state['center_coordinates'] = (cX, cY)

                            # 在图像上绘制所有检测到的框，而不仅仅是最佳目标
                            for box in boxes:
                                x1_draw, y1_draw, x2_draw, y2_draw = map(int, box.xyxy[0])
                                cls_draw = int(box.cls[0])
                                label = f"{model.names[cls_draw]}: {float(box.conf[0]):.2f}"
                                cv2.rectangle(img, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 2)
                                cv2.putText(img, label, (x1_draw, y1_draw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # 用一个不同的颜色标记被选为目标的物体中心
                            cv2.circle(img, (cX, cY), 7, (255, 0, 0), -1)

                        if not found_object:
                            with lock:
                                shared_state['center_coordinates'] = None

                        cv2.imshow('Video Feed', img)
                        if cv2.waitKey(1) == 27:
                            with lock:
                                shared_state['running'] = False
                            break

            if cv2.getWindowProperty('Video Feed', cv2.WND_PROP_VISIBLE) < 1:
                print("\n[视频线程] 视频窗口已关闭，正在停止程序...")
                with lock:
                    shared_state['running'] = False
                break
        except Exception as e:
            print(f"\n[视频线程] 处理视频帧时发生错误: {e}")
            time.sleep(1)

    print("\n[视频线程] 正在关闭...")
    cv2.destroyAllWindows()