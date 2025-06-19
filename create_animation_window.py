import os
import sys
import time

import cv2
from useful_classes import (TrapezoidWidget, VideoDropLabel, ProgressOverlay, VideoPlayerPopup, AnimationWorker,
                            resource_path)
from PyQt5.QtCore import Qt, QCoreApplication, QThread
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPalette, QBrush, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QMainWindow, QGraphicsDropShadowEffect, QComboBox, QProgressBar, QMessageBox, QInputDialog
)


class CreateAnimationWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.metr_model_path = None
        self.select_frame = None
        self.metr_model_combobox = None
        self.frame_combobox = None
        self.worker = None
        self.thread = None
        self.model_window = None
        self.res_window = None

        self.progress_bar = None
        self.json_path = None
        self.toggle_mode_btn = None
        self.player_window = None
        self.video_label = None
        self.choose_video_btn = None
        self.model_label = None
        self.add_model_btn = None
        self.video_path = None
        self.video_player_window = None
        self.model_combobox = None
        self.model_path = None
        self.accept_json_mode = False

        self.setWindowTitle("Генератор анимаций")
        self.initUI()
        self.progress_overlay = ProgressOverlay(self)

    def open_res_window(self):
        self.main_window.get_res_win().load_animations()
        self.main_window.slide_to(1)

    def open_model_window(self):
        self.main_window.slide_to(2)

    def initUI(self):
        central_widget = QWidget()
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0.0, QColor("#6344BC"))
        gradient.setColorAt(0.5, QColor("#C4B5E0"))
        gradient.setColorAt(1.0, QColor("#6344BC"))

        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        central_widget.setAutoFillBackground(True)
        central_widget.setPalette(palette)

        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)

        nav_bar = self.create_nav_bar()
        root_layout.addWidget(nav_bar)

        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content_layout = self.create_main_content()
        root_layout.addLayout(content_layout)
        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        central_widget.setLayout(root_layout)

        self.setStyleSheet(""" 
            QLabel {
                color: #C4B5E0;
                font-size: 14px;
            }

            QPushButton {
                background-color: #2c2c2c;
                color: white;
                font-size: 14px;
                border: 1px solid #555555;
                border-radius: 10px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #6344BC;
            }

            QPushButton:pressed {
                background-color: #D1B42A;
            }

            QLabel#videoLabel, QLabel#modelLabel {
                border: 2px solid #777;
                border-radius: 16px;
                background-color: #2b2b2b;
                padding: 10px;
            }

            QLabel#videoLabel:hover, QLabel#modelLabel:hover {
                background-color: #333333;
                border: 2px solid #aaa;
            }

            QWidget#navBar QPushButton {
                background-color: #202020;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 6px 12px;
            }

            QWidget#navBar QPushButton:hover {
                background-color: #6344BC;
            }

            QWidget#navBar QPushButton:pressed {
                background-color: #D1B42A;
            }

            QComboBox {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 10px;
                padding: 8px 12px;
            }

            QComboBox:hover {
                background-color: #6344BC;
            }

            QComboBox::drop-down {
                width: 30px;
                border-left: 1px solid #444;
            }

            QComboBox QAbstractItemView {
                background-color: #2c2c2c;
                color: #ffffff;
                selection-background-color: #6344BC;
                selection-color: #ffffff;
                border: 1px solid #222;
                padding: 4px;
            }

            QProgressBar {
                border: 2px solid #555;
                border-radius: 10px;
                text-align: center;
                color: white;
                background-color: #6344BC;
            }

            QProgressBar::chunk {
                background-color: #D1B42A;
                width: 20px;
            }

            QInputDialog {
                background-color: #C4B5E0;
            }

            QInputDialog QLabel {
                color: black;
                font-size: 14px;
            }

            QMessageBox {
                background-color: #C4B5E0;
            }

            QMessageBox QLabel {
                color: black;
                font-size: 14px;
            }
        """)

        arrow_icon_path = resource_path("Source/Images/icons8-стрелка-вниз-64_2.png").replace("\\", "/")

        self.model_combobox.setStyleSheet(f"""
            QComboBox::down-arrow {{
                image: url("{arrow_icon_path}");
                width: 20px;
                height: 20px;
            }}
        """)

        self.frame_combobox.setStyleSheet(f"""
                    QComboBox::down-arrow {{
                        image: url("{arrow_icon_path}");
                        width: 20px;
                        height: 20px;
                    }}
                """)

        self.metr_model_combobox.setStyleSheet(f"""
                    QComboBox::down-arrow {{
                        image: url("{arrow_icon_path}");
                        width: 20px;
                        height: 20px;
                    }}
                """)

    def create_nav_bar(self):
        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        nav_widget = TrapezoidWidget()
        nav_widget.setFixedWidth(720)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(20, 12, 20, 12)
        nav_layout.setSpacing(20)
        nav_layout.setAlignment(Qt.AlignCenter)

        archive_btn = QPushButton("🗂 Архив анимаций")
        create_btn = QPushButton("✨ Создать анимацию")
        upload_btn = QPushButton("➕ Загрузить модель")

        cur_window_btn = create_btn

        for btn in [archive_btn, create_btn, upload_btn]:
            btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            btn.setMinimumHeight(10)
            btn.setMinimumWidth(200)
            if btn == cur_window_btn:
                btn.setStyleSheet("""
                            QPushButton {
                                background-color: #2c2c2c;
                                color: white;
                                font-size: 14px;
                                border: 2px solid #D1B42A;
                                border-radius: 10px;
                                padding: 8px 16px;
                            }
                        """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2c2c2c;
                        color: white;
                        font-size: 14px;
                        border: 1px solid #555555;
                        border-radius: 10px;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #6344BC;
                    }

                    QPushButton:pressed {
                        background-color: #D1B42A;
                    }
                """)

        archive_btn.clicked.connect(self.open_res_window)
        upload_btn.clicked.connect(self.open_model_window)

        nav_layout.addWidget(archive_btn)
        nav_layout.addWidget(create_btn)
        nav_layout.addWidget(upload_btn)

        nav_widget.setLayout(nav_layout)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 150))
        nav_widget.setGraphicsEffect(shadow)

        container_layout.addWidget(nav_widget)
        container.setLayout(container_layout)

        return container

    def create_main_content(self):
        layout = QHBoxLayout()

        # Блок загрузки видео
        video_frame = QVBoxLayout()
        self.video_label = VideoDropLabel(self)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(500, 500)
        self.video_label.setAlignment(Qt.AlignCenter)

        default_pixmap = QPixmap(resource_path("Source/Images/no_video3.png"))
        if not default_pixmap.isNull():
            self.video_label.setPixmap(default_pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        else:
            self.video_label.setText("Видео не выбрано")

        self.video_label.mousePressEvent = self.open_video_player

        video_shadow = QGraphicsDropShadowEffect()
        video_shadow.setBlurRadius(25)
        video_shadow.setOffset(0, 0)
        video_shadow.setColor(QColor(0, 0, 0, 180))
        self.video_label.setGraphicsEffect(video_shadow)

        self.choose_video_btn = QPushButton("Выбрать видео")
        self.choose_video_btn.clicked.connect(self.select_video)
        video_container = QVBoxLayout()
        video_header = QHBoxLayout()

        self.toggle_mode_btn = QPushButton("Режим: Видео")
        self.toggle_mode_btn.setFixedWidth(140)
        self.toggle_mode_btn.clicked.connect(self.toggle_input_mode)

        video_header.addStretch()
        video_header.addWidget(self.toggle_mode_btn)

        video_container.addLayout(video_header)
        video_container.addWidget(self.video_label)

        video_frame.addLayout(video_container)
        video_frame.addWidget(self.choose_video_btn)

        # Блок выбора 3D модели
        model_frame = QVBoxLayout()
        self.model_label = QLabel("Нет 3D моделей")
        self.model_label.setObjectName("modelLabel")
        self.model_label.setFixedSize(500, 500)
        self.model_label.setAlignment(Qt.AlignCenter)

        model_shadow = QGraphicsDropShadowEffect()
        model_shadow.setBlurRadius(25)
        model_shadow.setOffset(0, 0)
        model_shadow.setColor(QColor(80, 80, 80, 150))
        self.model_label.setGraphicsEffect(model_shadow)

        self.model_combobox = QComboBox()
        self.model_combobox.currentIndexChanged.connect(self.update_model_preview)
        self.load_models()

        model_frame.addWidget(self.model_label)
        model_frame.addWidget(self.model_combobox)

        generate_frame = QVBoxLayout()

        # Кнопка запуска процесса создания анимации
        btn_wrapper_layout = QHBoxLayout()

        btn_wrapper_layout.addStretch(1)
        generate_btn = QPushButton("Создать анимацию")
        generate_btn.setFixedSize(200, 60)
        generate_btn.clicked.connect(self.generate_animation)
        btn_wrapper_layout.addWidget(generate_btn)
        btn_wrapper_layout.addStretch(1)

        combo_box_frame = QHBoxLayout()

        self.frame_combobox = QComboBox()
        self.frame_combobox.currentIndexChanged.connect(self.update_select_frame)
        self.load_frames()

        self.metr_model_combobox = QComboBox()
        self.metr_model_combobox.currentIndexChanged.connect(self.update_metr_model_preview)
        self.load_metr_models()

        combo_box_frame.addWidget(self.frame_combobox)
        combo_box_frame.addWidget(self.metr_model_combobox)

        generate_frame.addLayout(btn_wrapper_layout)
        generate_frame.addLayout(combo_box_frame)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(video_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(generate_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(model_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        return layout

    def toggle_input_mode(self):
        self.accept_json_mode = not self.accept_json_mode

        if self.accept_json_mode:
            self.video_path = None
            self.toggle_mode_btn.setText("Режим: JSON")
            self.choose_video_btn.setText("Выбрать JSON файл")
            self.video_label.setText("JSON не выбран")
        else:
            self.json_path = None
            self.toggle_mode_btn.setText("Режим: Видео")
            self.choose_video_btn.setText("Выбрать видео")
            default_pixmap = QPixmap(resource_path("Source/Images/no_video3.png"))
            if not default_pixmap.isNull():
                self.video_label.setPixmap(default_pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
            else:
                self.video_label.setText("Видео не выбрано")

    def handle_video_drop(self, file_path):
        if not os.path.exists(file_path):
            return

        if self.accept_json_mode:
            if file_path.endswith(".json"):
                self.video_label.setText(f"JSON файл загружен:\n{os.path.basename(file_path)}")
            else:
                self.video_label.setText("Неверный формат файла (только .json)")
        else:
            if file_path.lower().endswith((".mp4", ".avi", ".mov")):
                self.video_path = file_path
                self.display_first_frame(file_path)
            else:
                self.video_label.setText("Неверный формат файла (только видео)")

    def select_video(self):
        if self.accept_json_mode:
            file, _ = QFileDialog.getOpenFileName(self, "Выберите JSON", "", "JSON Files (*.json)")
            if file:
                self.handle_video_drop(file)
                self.json_path = file
        else:
            file, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi *.mov)")
            if file:
                self.video_label.setText("Загрузка видео...")
                self.display_first_frame(file)
                self.video_path = file

    def display_first_frame(self, video_path):
        # Используем OpenCV для извлечения первого кадра
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Преобразуем кадр в формат, который может быть использован в PyQt
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame_rgb.shape
                byte_array = frame_rgb.tobytes()

                # Создаем QPixmap
                qimage = QImage(byte_array, width, height, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap(qimage).scaled(self.video_label.size(), Qt.KeepAspectRatio))
            else:
                self.video_label.setText("Не загрузить видео")
        else:
            self.video_label.setText("Ошибка загрузки видео")
        cap.release()

    def open_video_player(self, event):
        if not self.video_path:
            return

        if self.video_player_window is not None and self.video_player_window.isVisible():
            self.video_player_window.raise_()
            self.video_player_window.activateWindow()
            return

        if self.video_player_window is not None:
            self.video_player_window.close()

        self.video_player_window = VideoPlayerPopup(self.video_path, parent=self)
        self.video_player_window.show()

    def load_models(self):
        models_dir = resource_path("Source/rigged_models")
        if not os.path.exists(models_dir):
            return

        model_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.fbx')]
        self.model_combobox.clear()
        self.model_combobox.addItems(model_files)

        if model_files:
            self.update_model_preview(0)
        else:
            self.model_label.setText("Нет доступных моделей")

    def update_model_preview(self, index):
        model_name = self.model_combobox.itemText(index)
        if not model_name:
            return

        self.model_path = resource_path(f"Source/rigged_models/{model_name}")
        preview_path = resource_path("Source/rigged_models/" + os.path.splitext(model_name)[0] + ".png")
        if not os.path.exists(preview_path):
            preview_path = resource_path("Source/rigged_models/" + os.path.splitext(model_name)[0] + ".jpg")

        if os.path.exists(preview_path):
            pixmap = QPixmap(preview_path)
            self.model_label.setPixmap(pixmap.scaled(self.model_label.size(), Qt.KeepAspectRatio))
        else:
            self.model_label.setText(f"Модель: {model_name}\n(Нет изображения)")

    def load_frames(self):
        frames = ['24 fps', '30 fps', '48 fps', '60 fps', '120 fps']
        self.frame_combobox.clear()
        self.frame_combobox.addItems(frames)

        self.update_select_frame(0)

    def update_select_frame(self, index):
        self.select_frame = self.frame_combobox.itemText(index).strip(' fps')

    def load_metr_models(self):
        models_dir = resource_path("metrabs_models")
        if not os.path.exists(models_dir):
            return

        model_files = [f for f in os.listdir(models_dir)]
        self.metr_model_combobox.clear()
        self.metr_model_combobox.addItems(model_files)

        if model_files:
            self.update_metr_model_preview(0)

    def update_metr_model_preview(self, index):
        model_name = self.metr_model_combobox.itemText(index)
        if not model_name:
            return

        self.metr_model_path = resource_path(f"metrabs_models/{model_name}")

    def update_progress(self, value, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        QCoreApplication.processEvents()

    def generate_animation(self):
        if ((self.json_path is not None or (self.video_path is not None and self.metr_model_path is not None)) and
                self.model_path is not None):
            self.start_skeleton_thread()

    def start_skeleton_thread(self):
        file_path = self.json_path if self.accept_json_mode else self.video_path
        model_path = self.model_path
        frame_rate = self.select_frame
        from_json = self.accept_json_mode
        metr_model_path = self.metr_model_path

        self.thread = QThread()
        self.worker = AnimationWorker(file_path, model_path, from_json, frame_rate, metr_model_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_overlay.update_progress)
        self.worker.finished.connect(self.on_animation_created)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.progress_overlay.show_overlay("Создание анимации...")
        self.thread.start()

    def on_animation_created(self, fbx_path):
        self.progress_overlay.hide_overlay()

        if fbx_path:
            name = fbx_path.split('/')[-1]
            new_name, ok = QInputDialog.getText(self, "Название файла", "Задайте имя файла (с расширением .fbx):",
                                                text=name)
            if ok and new_name:
                new_path = resource_path(f"animated_models/{new_name}")
                new_video_path = new_path.strip('.fbx') + '.mp4'
                if os.path.exists(new_path):
                    QMessageBox.warning(self, "Ошибка", "Файл с таким именем уже существует.")
                    return
                try:
                    os.rename(fbx_path, new_path)
                    if self.video_path is not None and os.path.exists(self.video_path):
                        os.rename(self.video_path, new_video_path)
                    self.video_path = new_video_path
                except Exception as e:
                    QMessageBox.warning(self, "Ошибка", str(e))
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось создать анимацию. Попробуйте другой файл.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CreateAnimationWindow()
    window.show()
    sys.exit(app.exec_())
