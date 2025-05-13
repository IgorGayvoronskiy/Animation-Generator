import os
import sys
import time

import cv2
from useful_classes import TrapezoidWidget, VideoDropLabel, ProgressOverlay, VideoPlayerPopup
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPalette, QBrush, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QMainWindow, QGraphicsDropShadowEffect, QComboBox, QProgressBar
)

import animation_generator
import model_download_window
import results_window


class CreateAnimationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.showMaximized()

    def open_res_window(self):
        self.res_window = results_window.ResultsWindow()
        self.res_window.show()
        self.close()

    def open_model_window(self):
        self.model_window = model_download_window.ModelDownloadWindow()
        self.model_window.show()
        self.close()

    def initUI(self):
        # Центральный виджет с градиентом
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

        # Основной вертикальный layout
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)

        nav_bar = self.create_nav_bar()
        root_layout.addWidget(nav_bar)

        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content_layout = self.create_main_content()
        root_layout.addLayout(content_layout)
        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        central_widget.setLayout(root_layout)

        # Стили
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
                background-color: #2c2c2c;      /* фон выбранного элемента */
                color: #ffffff;                 /* цвет текста выбранного элемента */
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
            
            QComboBox::down-arrow {
                image: url(Source/Images/icons8-стрелка-вниз-64_2.png);
                width: 20px;
                height: 20px;
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

        archive_btn = QPushButton("🗂 Результаты")
        create_btn = QPushButton("✨ Создать анимацию")
        upload_btn = QPushButton("➕ Загрузить модель")

        cur_window_btn = create_btn

        # Сделать кнопки одинаковой ширины
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

        # Эффект тени
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

        default_pixmap = QPixmap("Source/Images/no_video3.png")
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

        # Кнопка запуска процесса создания анимации
        generate_btn = QPushButton("Создать анимацию")
        generate_btn.setFixedSize(200, 60)
        generate_btn.clicked.connect(self.generate_animation)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(video_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(generate_btn)
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
            default_pixmap = QPixmap("Source/Images/no_video3.png")
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
                # Можно здесь добавить парсинг и отображение чего-то из JSON
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

        # Открываем новое окно с плеером
        self.video_player_window = VideoPlayerPopup(self.video_path, parent=self)
        self.video_player_window.show()

    def load_models(self):
        models_dir = "Source/rigged_models"
        if not os.path.exists(models_dir):
            return

        model_files = [f for f in os.listdir(models_dir) if f.lower().endswith(('.glb', '.obj', '.fbx'))] # возможно нужно будет оставить только fbx
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

        self.model_path = f"Source/rigged_models/{model_name}"
        preview_path = os.path.join("Source/rigged_models", os.path.splitext(model_name)[0] + ".png")
        if not os.path.exists(preview_path):
            preview_path = os.path.join("Source/rigged_models", os.path.splitext(model_name)[0] + ".jpg")

        if os.path.exists(preview_path):
            pixmap = QPixmap(preview_path)
            self.model_label.setPixmap(pixmap.scaled(self.model_label.size(), Qt.KeepAspectRatio))
        else:
            self.model_label.setText(f"Модель: {model_name}\n(Нет изображения)")

    def update_progress(self, value, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        QCoreApplication.processEvents()

    def generate_animation(self):
        if self.json_path is not None or self.video_path is not None:
            self.progress_overlay.show_overlay()
            self.progress_overlay.update_message("Создание анимации...")

            animation_generator.create_animation(
                key_points_data_path=self.json_path if self.accept_json_mode else self.video_path,
                model_path=self.model_path,
                from_json=self.accept_json_mode,
                progress_callback=self.progress_overlay.update_progress
            )

            self.progress_overlay.hide_overlay()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CreateAnimationWindow()
    window.show()
    sys.exit(app.exec_())
