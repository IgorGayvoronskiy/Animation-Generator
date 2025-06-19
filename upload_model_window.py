import os
import shutil

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QLinearGradient, QColor, QPalette, QBrush, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QPushButton, QWidget, QVBoxLayout,
    QFileDialog, QApplication, QSpacerItem, QSizePolicy, QHBoxLayout,
    QGraphicsDropShadowEffect, QMainWindow
)
import sys

from useful_classes import ImageDropLabel, FbxDropLabel, resource_path


class UploadWindow(QMainWindow):
    model_uploaded = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.choose_model_btn = None
        self.model_label = None
        self.choose_image_btn = None
        self.image_label = None
        self.setWindowIcon(QIcon(resource_path("Source/Images/AM2.ico")))
        self.setWindowTitle("Загрузка модели и изображения")
        self.initUI()

        self.image_path = None
        self.model_path = None

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

        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content_layout = self.create_main_content()
        root_layout.addLayout(content_layout)
        root_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        central_widget.setLayout(root_layout)

        # Стили
        self.setStyleSheet("""
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

            GifButton:hover {
                background-color: #2c2c2c;
            }

            QPushButton:pressed {
                background-color: #D1B42A;
            }

            QLabel#imageLabel, QLabel#modelLabel{
                color: #C4B5E0;
                font-size: 14px;
                border: 2px solid #777;
                border-radius: 16px;
                background-color: #2b2b2b;
                padding: 10px;
            }
        """)

    def create_main_content(self):
        layout = QHBoxLayout()

        # Блок загрузки видео
        image_frame = QVBoxLayout()
        self.image_label = ImageDropLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setFixedSize(500, 500)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label_reset()

        image_shadow = QGraphicsDropShadowEffect()
        image_shadow.setBlurRadius(25)
        image_shadow.setOffset(0, 0)
        image_shadow.setColor(QColor(0, 0, 0, 180))
        self.image_label.setGraphicsEffect(image_shadow)

        self.choose_image_btn = QPushButton("Выбрать изображение")
        self.choose_image_btn.clicked.connect(self.select_json_img_file)

        image_frame.addWidget(self.image_label)
        image_frame.addWidget(self.choose_image_btn)

        # Блок загрузки модели
        model_frame = QVBoxLayout()
        self.model_label = FbxDropLabel(self)
        self.model_label.setObjectName("modelLabel")
        self.model_label.setText('Модель не выбрана')
        self.model_label.setFixedSize(500, 500)
        self.model_label.setAlignment(Qt.AlignCenter)

        self.image_label_reset()

        model_shadow = QGraphicsDropShadowEffect()
        model_shadow.setBlurRadius(25)
        model_shadow.setOffset(0, 0)
        model_shadow.setColor(QColor(0, 0, 0, 180))
        self.model_label.setGraphicsEffect(model_shadow)

        self.choose_model_btn = QPushButton("Выбрать модель")
        self.choose_model_btn.clicked.connect(self.select_fbx_file)

        model_frame.addWidget(self.model_label)
        model_frame.addWidget(self.choose_model_btn)

        # Кнопка сохранения
        save_btn = QPushButton("Сохранить")
        save_btn.setFixedSize(140, 60)
        save_btn.clicked.connect(self.save_model)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(image_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(save_btn)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(model_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        return layout

    def image_label_reset(self):
        default_pixmap = QPixmap(resource_path("Source/Images/no_image.jpg"))
        if not default_pixmap.isNull():
            self.image_label.setPixmap(default_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        else:
            self.image_label.setText("Изображение не выбрано")

    def handle_img_file_drop(self, file_path):
        if not os.path.exists(file_path):
            return

        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            self.image_path = file_path
            default_pixmap = QPixmap(file_path)
            if not default_pixmap.isNull():
                self.image_label.setPixmap(default_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            else:
                self.image_label.setText("Изображение не выбрано")
        else:
            self.image_label.setText("Неверный формат файла (только изображение)")

    def select_json_img_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Video Files (*.png *.jpg *.jpeg)")
        if file:
            self.image_label.setText("Загрузка изображения...")
            default_pixmap = QPixmap(file)
            if not default_pixmap.isNull():
                self.image_label.setPixmap(default_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            else:
                self.image_label.setText("Ошибка загрузки изображения")
            self.image_path = file

    def handle_fbx_file_drop(self, file_path):
        if not os.path.exists(file_path):
            return

        if file_path.lower().endswith('.fbx'):
            name = file_path.split('/')[-1].split('.')[0]
            self.model_path = file_path
            self.model_label.setText(f"Модель: {name}")
        else:
            self.model_label.setText("Неверный формат файла (только изображение)")

    def select_fbx_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "", "Fbx Files (*.fbx)")
        if file:
            self.handle_fbx_file_drop(file)

    def save_model(self):
        if self.model_path is not None:
            model_name = self.model_path.split('/')[-1]

            new_model_path = resource_path(f'Source/rigged_models/{model_name}')
            new_image_path = new_model_path.split('.')[0] + '.png'

            shutil.copy2(self.model_path, new_model_path)

            if self.image_path is not None:
                shutil.copy2(self.image_path, new_image_path)

            self.model_uploaded.emit()
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UploadWindow()
    window.show()
    sys.exit(app.exec_())
