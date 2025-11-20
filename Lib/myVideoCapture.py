import cv2
import queue
import threading
import time


# 自定义无缓存读视频类
class my_VideoCapture:
    """Customized VideoCapture, always read latest frame """

    def __init__(self, camera_id):
        # "camera_id" is a int type id or string name
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_GSTREAMER)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False  # to gracefully close sub-thread
        th = threading.Thread(target=self._reader)
        th.daemon = True  # 设置工作线程为后台运行
        th.start()

    def get_cap(self):
        return self.cap

    # 实时读帧，只保存最后一帧
    def _reader(self):
        while not self.stop_threads:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, -1)  # 垂直+水平翻转
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def terminate(self):
        self.stop_threads = True
        self.cap.release()


if __name__ == "__main__":
    # 测试自定义VideoCapture类
    cap = VideoCapture(0)
    while True:
        frame = cap.read()
        time.sleep(0.05)  # 模拟耗时操作，单位：秒
        cv2.imshow("frame", frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break