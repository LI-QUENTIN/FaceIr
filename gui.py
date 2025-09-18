import os
import json
import numpy as np
import threading
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame,
                             QGraphicsDropShadowEffect, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont

from threads import VideoThread, ImageProcessingThread
from data_manager import load_known_faces, save_known_faces
import face_recognition

class FaceRecognitionApp(QMainWindow):
    # 用于保存和加载人脸数据的文件
    KNOWN_FACES_FILE = "known_faces.json"

    def __init__(self):
        super().__init__()
        
        # 初始化窗口状态并最大化显示
        self.setWindowTitle("人脸识别系统")
        self.showMaximized()
        
        # 现代风格的 UI 样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
                color: #f7f7f7;
            }
            QLabel#mainLabel {
                border-radius: 15px;
                background-color: #16213e;
                color: #53e7f2;
            }
            QFrame#controlFrame, QFrame#sidebarFrame {
                background-color: #16213e;
                border-radius: 15px;
            }
            QPushButton {
                background-color: #53e7f2;
                color: #1a1a2e;
                border: 2px solid #45c1ca;
                border-radius: 15px;
                padding: 12px 25px;
                font-size: 20px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45c1ca;
                border: 2px solid #3699a2;
            }
            QPushButton:pressed {
                background-color: #3699a2;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
                border: none;
            }
            QLabel#infoLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e94560;
            }
            QLabel#sidebarTitle {
                font-size: 24px;
                font-weight: bold;
                padding-bottom: 10px;
                border-bottom: 2px solid #53e7f2;
                color: #e94560;
            }
            .sidebar-item-label {
                color: #f7f7f7;
                font-size: 16px;
            }
            .delete-btn {
                background-color: #ff6b6b;
                border: 2px solid #ff4f4f;
                padding: 5px 10px;
                font-size: 14px;
                border-radius: 10px;
                color: white;
            }
            .delete-btn:hover {
                background-color: #ff4f4f;
                border: 2px solid #e74c3c;
            }
        """)

        # 主窗口部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 将主布局改为 QHBoxLayout 以容纳侧边栏
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # --- 左侧边栏 ---
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setFixedWidth(300)
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame)
        self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
        
        self.sidebar_title = QLabel("已加载照片")
        self.sidebar_title.setObjectName("sidebarTitle")
        self.sidebar_layout.addWidget(self.sidebar_title)
        
        self.face_list_layout = QVBoxLayout()
        self.face_list_layout.setAlignment(Qt.AlignTop)
        self.sidebar_layout.addLayout(self.face_list_layout)
        
        self.sidebar_layout.addStretch() # 添加伸展器，让列表顶部对齐
        
        self.main_layout.addWidget(self.sidebar_frame)

        # --- 右侧主内容区域 ---
        self.main_content_widget = QWidget()
        self.main_content_layout = QVBoxLayout(self.main_content_widget)

        # 视频和状态显示
        self.camera_label = QLabel("摄像头已关闭")
        self.camera_label.setObjectName("mainLabel")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.camera_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ecf0f1;")
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.camera_label.setGraphicsEffect(shadow)
        
        self.main_content_layout.addWidget(self.camera_label, 3)

        # 控制框架
        self.control_frame = QFrame()
        self.control_frame.setObjectName("controlFrame")
        self.control_frame.setFrameShape(QFrame.StyledPanel)
        self.control_frame.setFrameShadow(QFrame.Raised)
        self.control_layout = QVBoxLayout(self.control_frame)
        self.control_layout.setContentsMargins(20, 20, 20, 20)
        
        self.info_label = QLabel("请上传照片以开始人脸识别。")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(self.info_label)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传照片")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setToolTip("上传一张带有人脸的照片进行识别。")
        
        self.toggle_cam_btn = QPushButton("打开摄像头")
        self.toggle_cam_btn.clicked.connect(self.toggle_camera)
        self.toggle_cam_btn.setToolTip("打开或关闭网络摄像头流。")
        
        self.button_layout.addWidget(self.upload_btn)
        self.button_layout.addWidget(self.toggle_cam_btn)
        self.control_layout.addLayout(self.button_layout)
        
        self.main_content_layout.addWidget(self.control_frame, 1)

        self.main_layout.addWidget(self.main_content_widget, 1)

        # 状态变量
        self.known_face_encodings = []
        self.known_face_names = []
        self.video_thread = None
        self.image_thread = None
        self.is_camera_on = False

        # 启动时加载现有人脸数据
        self.load_known_faces()

    def load_known_faces(self):
        """从 JSON 文件加载人脸编码和名称。"""
        self.known_face_encodings, self.known_face_names, message = load_known_faces(self.KNOWN_FACES_FILE)
        
        self.info_label.setText(message)
        print(message)
        self.populate_sidebar() # 加载后更新侧边栏

    def save_known_faces(self):
        """将人脸编码和名称保存到 JSON 文件。"""
        save_known_faces(self.known_face_encodings, self.known_face_names, self.KNOWN_FACES_FILE)
        
    def toggle_camera(self):
        """打开和关闭摄像头。"""
        if self.is_camera_on:
            # 关闭摄像头
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
            self.is_camera_on = False
            self.camera_label.setPixmap(QPixmap())
            self.camera_label.setText("摄像头已关闭")
            self.toggle_cam_btn.setText("打开摄像头")
            print("摄像头已停止。")
        else:
            # 打开摄像头
            self.camera_label.setText("正在启动摄像头...")
            self.toggle_cam_btn.setText("关闭摄像头")
            self.video_thread = VideoThread(self.known_face_encodings, self.known_face_names)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.start()
            self.is_camera_on = True
            print("摄像头已启动。")
        
    def update_frame(self, image):
        """接收来自线程的 QImage 并更新 QLabel。"""
        if self.camera_label.text() != "":
             self.camera_label.setText("") 
        self.camera_label.setPixmap(QPixmap.fromImage(image).scaled(self.camera_label.size(), Qt.KeepAspectRatio))
        
    def upload_image(self):
        """打开文件对话框以上传图像，并启动一个新线程来处理它。"""
        # 检查摄像头是否打开，如果是则先关闭它
        if self.is_camera_on:
            self.toggle_camera()
            
        file_path, _ = QFileDialog.getOpenFileName(self, "选择一张带有人脸的照片", "", "Image Files (*.jpg *.jpeg *.png)")
        if not file_path:
            self.info_label.setText("未选择文件，请重试。")
            return

        self.info_label.setText("正在处理图像...")
        self.upload_btn.setEnabled(False)
        
        # 启动新的图像处理线程
        self.image_thread = ImageProcessingThread(file_path)
        self.image_thread.processing_finished.connect(self.handle_image_result)
        self.image_thread.start()

    def handle_image_result(self, encodings, message):
        """处理来自图像处理线程的结果。"""
        self.upload_btn.setEnabled(True)

        if not encodings:
            self.info_label.setText(message)
            print(message)
            return

        new_encoding = encodings[0]

        is_duplicate = False
        if self.known_face_encodings:
            matches = [face_recognition.compare_faces([e], new_encoding) for e in self.known_face_encodings]
            if any(True in match for match in matches):
                is_duplicate = True
        
        if is_duplicate:
            self.info_label.setText("此人脸已存在，无需重复添加。")
            print("发现重复人脸，未添加到数据库。")
        else:
            new_face_id = len(self.known_face_encodings) + 1
            new_face_name = f"Known ({new_face_id})"
            
            self.known_face_encodings.append(new_encoding)
            self.known_face_names.append(new_face_name)

            self.save_known_faces()
            
            self.info_label.setText(f"{message} 人脸 {new_face_id} 已保存。")
            print(f"新的人脸编码已生成并保存为 {new_face_name}。")
            self.populate_sidebar() # 添加新脸后更新侧边栏
    
    def populate_sidebar(self):
        """动态创建并填充侧边栏中的人脸列表。"""
        # 清除现有列表
        while self.face_list_layout.count():
            item = self.face_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 遍历已知人脸并添加
        for i, name in enumerate(self.known_face_names):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            
            name_label = QLabel(name)
            name_label.setStyleSheet("padding: 5px; color: #f7f7f7;")
            item_layout.addWidget(name_label)
            
            delete_btn = QPushButton("删除")
            delete_btn.setObjectName("delete-btn")
            delete_btn.setFixedSize(60, 30)
            # 使用 lambda 表达式将索引传递给删除函数
            delete_btn.clicked.connect(lambda _, index=i: self.delete_face(index))
            item_layout.addWidget(delete_btn)
            
            self.face_list_layout.addWidget(item_widget)
            
    def delete_face(self, index):
        """根据索引删除人脸数据。"""
        if 0 <= index < len(self.known_face_names):
            name_to_delete = self.known_face_names[index]
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            self.save_known_faces()
            self.populate_sidebar()
            self.info_label.setText(f"已删除人脸: {name_to_delete}")
            print(f"已删除人脸: {name_to_delete}")
        
    def closeEvent(self, event):
        """处理窗口关闭事件，确保线程停止。"""
        print("正在关闭应用程序...")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait() 
        if self.image_thread and self.image_thread.isRunning():
            self.image_thread.quit()
            self.image_thread.wait()
        event.accept()
