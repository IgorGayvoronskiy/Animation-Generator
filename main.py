import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QFileDialog, QMainWindow, QFrame,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор анимаций")
        self.initUI()

        self.showMaximized()  # Разворачиваем окно

    def initUI(self):
        central_widget = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Верхняя навигация
        nav_bar = self.create_nav_bar()
        root_layout.addLayout(nav_bar)

        # Центральное содержимое с вертикальным центрированием
        center_spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        center_spacer_bottom = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        content_layout = self.create_main_content()

        root_layout.addItem(center_spacer_top)
        root_layout.addLayout(content_layout)
        root_layout.addItem(center_spacer_bottom)

        central_widget.setLayout(root_layout)
        self.setCentralWidget(central_widget)

    def create_nav_bar(self):
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        archive_btn = QPushButton("Архив")
        create_btn = QPushButton("Создать анимацию")
        upload_btn = QPushButton("Загрузить модель")

        nav_layout.addWidget(archive_btn)
        nav_layout.addWidget(create_btn)
        nav_layout.addWidget(upload_btn)

        return nav_layout

    def create_main_content(self):
        layout = QHBoxLayout()

        # Видео блок
        video_frame = QVBoxLayout()
        self.video_label = QLabel("Выберите видео")
        self.video_label.setFixedSize(300, 300)
        self.video_label.setStyleSheet("background-color: gray; color: white;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.mousePressEvent = self.open_video_player

        self.choose_video_btn = QPushButton("Выбрать видео")
        self.choose_video_btn.clicked.connect(self.select_video)

        video_frame.addWidget(self.video_label)
        video_frame.addWidget(self.choose_video_btn)

        # Кнопка по центру
        generate_btn = QPushButton("Создать анимацию")
        generate_btn.setFixedSize(150, 40)
        generate_btn.clicked.connect(self.generate_animation)

        # Модель блок
        model_frame = QVBoxLayout()
        self.model_label = QLabel("Нет 3D моделей")
        self.model_label.setFixedSize(300, 300)
        self.model_label.setStyleSheet("background-color: lightgray;")
        self.model_label.setAlignment(Qt.AlignCenter)

        self.add_model_btn = QPushButton("Добавить модель")
        model_frame.addWidget(self.model_label)
        model_frame.addWidget(self.add_model_btn)

        # Симметричное размещение
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(video_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(generate_btn)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(model_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        return layout

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi *.mov)")
        if file:
            self.video_label.setText("Видео выбрано")

    def open_video_player(self, event):
        print("Открыть мини-плеер (заглушка)")

    def generate_animation(self):
        print("Генерация анимации...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())