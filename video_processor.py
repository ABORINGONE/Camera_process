# video_processor.py
import time
import cv2
import requests
import numpy as np
from ultralytics import YOLO
from config import VIDEO_STREAM_URL

# --- 新增：YOLOv8 模型配置 ---

# 加载 YOLOv8n 模型。'n' 代表 nano，是最小最快的版本。
# 第一次运行时，模型文件 yolov8n.pt 会被自动下载。
# 您也可以换成 'yolov8s.pt' (small) 来获取更高的精度，但速度会稍慢。
print("[视频线程] 正在加载 YOLOv8n 模型...")
try:
    model = YOLO('yolov8n.pt') 
    print("[视频线程] 模型加载成功。")
except Exception as e:
    print(f"[视频线程] 致命错误：YOLOv8模型加载失败: {e}")
    print("请确保 'ultralytics' 库已成功安装 (pip install ultralytics)。")
    model = None

# --- 您可以指定只追踪特定的物体 ---
# YOLOv8 使用 COCO 数据集，其中 'bottle' 的类别索引是 39。
# 您可以在这里指定想要追踪的类别名称。
# 如果设置为 None，则会追踪检测到的置信度最高的物体。
TARGET_CLASS_NAME = "bottle" 

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
                        
                        # 1. 使用YOLOv8进行预测
                        # stream=True 是一种优化的处理模式
                        results = model(img, stream=True, verbose=False) 

                        found_object = False
                        best_target = None # 用来存储找到的最佳目标信息

                        # 2. 解析结果
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                # 获取类别和置信度
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = model.names[cls]

                                # 检查是否是我们想要的目标
                                if TARGET_CLASS_NAME is None or class_name == TARGET_CLASS_NAME:
                                    # 如果是，就记录下来，并寻找置信度最高的那个
                                    if best_target is None or conf > best_target['conf']:
                                        best_target = {
                                            'box': box,
                                            'conf': conf,
                                            'class_name': class_name
                                        }
                        
                        # 3. 如果找到了目标，处理最佳结果
                        if best_target is not None:
                            found_object = True
                            
                            # 获取边界框坐标
                            x1, y1, x2, y2 = best_target['box'].xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 计算中心点
                            cX = x1 + (x2 - x1) // 2
                            cY = y1 + (y2 - y1) // 2

                            with lock:
                                shared_state['center_coordinates'] = (cX, cY)

                            # 在图像上绘制方框和标签
                            label = f"{best_target['class_name']}: {best_target['conf']:.2f}"
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
                print("[视频线程] 视频窗口已关闭，正在停止程序...")
                with lock:
                    shared_state['running'] = False
                break

        except Exception as e:
            print(f"[视频线程] 处理视频帧时发生错误: {e}")
            time.sleep(1)

    print("[视频线程] 正在关闭...")
    cv2.destroyAllWindows()