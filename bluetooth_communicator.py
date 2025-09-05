# bluetooth_communicator.py

import serial
import time
import struct
from config import SERIAL_PORT, BAUD_RATE

# 定义一个缓冲区阈值，当待发送字节超过这个数时，我们就暂停写入
# 这个值可以根据实际情况调整，例如设置为数据包长度的几倍
BUFFER_THRESHOLD = 21  # 7字节/包 * 3

def run_bluetooth_communication(shared_state, lock):
    """
    在一个独立线程中运行，负责连接蓝牙串口，并周期性地发送更新后的指令。
    此版本为终极优化版，可防止因缓冲区满导致的卡顿和超时。
    """
    print("[蓝牙线程] 线程已启动。")
    ser = None
    last_moving = None
    last_firing = None

    while shared_state.get('running', True):
        try:
            if ser is None or not ser.is_open:
                print(f"[蓝牙线程] 正在尝试连接到串口 {SERIAL_PORT}...")
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, write_timeout=0.5)
                print(f"[蓝牙线程] 串口 {SERIAL_PORT} 连接成功。")
                time.sleep(2)

            with lock:
                current_firing = shared_state.get('firing')
                current_moving = shared_state.get('moving')
            
            # 1. 仅在状态改变时才准备发送
            if current_moving != last_moving or current_firing != last_firing:
                
                # 2. 在发送前，检查输出缓冲区是否拥堵
                if ser.out_waiting < BUFFER_THRESHOLD:
                    data_packet = struct.pack('<BBHHB', 
                                             0xF0, 0x00,
                                             current_moving[0], current_moving[1],
                                             int(current_firing))
                    ser.write(data_packet)
                    # print(f"[蓝牙线程] 发送指令: {data_packet.hex(' ')} (缓冲区: {ser.out_waiting}字节)")

                    # 更新状态
                    last_moving = current_moving
                    last_firing = current_firing
                else:

                    print(f"[蓝牙线程] 警告：蓝牙输出缓冲区拥堵 ({ser.out_waiting}字节)，跳过本次发送。")

            # 无论是否发送，都保持固定的循环频率
            time.sleep(0.1)

        except serial.SerialException:
            if ser and ser.is_open:
                ser.close()
            ser = None
            print("[蓝牙线程] 串口连接丢失，将在5秒后重试...")
            for _ in range(50): 
                if not shared_state.get('running', True): break
                time.sleep(0.1)
        except Exception as e:
            print(f"[蓝牙线程] 发生未知错误: {e}")
            time.sleep(5)

    if ser and ser.is_open:
        ser.close()
    print("[蓝牙线程] 正在关闭...")