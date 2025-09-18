import cv2
import face_recognition
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

class VideoThread(QThread):
    # 此信号将用于将处理后的帧发送到主线程
    frame_ready = pyqtSignal(QImage)
    
    def __init__(self, known_face_encodings, known_face_names, parent=None):
        super().__init__(parent)
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.is_running = True
        self.cap = None

    def run(self):
        """视频线程的主循环。"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头。")
            return

        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 对帧进行人脸识别处理
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                
                # 检查是否与已知人脸匹配
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                # 将人脸位置缩放回原始尺寸
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # 在帧上绘制矩形和标签
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # 将帧转换为 QImage 并发送
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.frame_ready.emit(qt_image)
            
        if self.cap:
            self.cap.release()

    def stop(self):
        """优雅地停止视频线程。"""
        self.is_running = False

class ImageProcessingThread(QThread):
    # 信号用于向主线程报告结果
    processing_finished = pyqtSignal(list, str) # 编码列表，结果消息
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        """处理图像并发送结果。"""
        try:
            known_image = face_recognition.load_image_file(self.file_path)
            encodings = face_recognition.face_encodings(known_image)
            
            if len(encodings) > 0:
                self.processing_finished.emit(encodings, "图像加载成功。")
            else:
                self.processing_finished.emit([], "未找到人脸，请选择另一张图像。")
        except Exception as e:
            self.processing_finished.emit([], f"加载失败：{e}")
