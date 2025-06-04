import os
import sys

import cv2
from useful_classes import TrapezoidWidget, VideoPlayerPopup, GifButton
from PyQt5.QtCore import Qt, QCoreApplication, QEvent, QTimer
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPalette, QBrush, QImage, QFont, QMovie
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QMainWindow, QGraphicsDropShadowEffect, QComboBox, QProgressBar, QListWidget, QMessageBox, QInputDialog
)

import create_animation_window
import model_download_window


class ResultsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.save_button = None
        self.rename_button = None
        self.delete_button = None
        self.video_path = None
        self.animation_title = None
        self.video_player_window = None
        self.list_title = None
        self.animation_label = None
        self.animation_list = None
        self.create_window = None
        self.model_window = None

        self.setWindowTitle("Результаты")
        self.initUI()
        self.showMaximized()

    def open_model_window(self):
        self.model_window = model_download_window.ModelDownloadWindow()
        self.model_window.show()
        self.hide()

    def open_create_window(self):
        self.create_window = create_animation_window.CreateAnimationWindow()
        self.create_window.show()
        self.hide()

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

            QLabel#animationLabel{
                color: #C4B5E0;
                font-size: 14px;
                border: 2px solid #777;
                border-radius: 16px;
                background-color: #2b2b2b;
                padding: 10px;
            }

            QLabel#animationLabel:hover {
                background-color: #333333;
                border: 2px solid #aaa;
            }
            
            QLabel#listTitle, QLabel#animationTitle{
                color: #black;
                font-size: 14px;
                background-color: #6344BC;
            }
            QMessageBox {
                color: #black;
                background-color: #C4B5E0;
            }
            QInputDialog {
                color: #black;
                background-color: #C4B5E0;
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

        archive_btn = QPushButton("🗂 Архив анимаций")
        create_btn = QPushButton("✨ Создать анимацию")
        upload_btn = QPushButton("➕ Загрузить модель")

        cur_window_btn = archive_btn

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

        upload_btn.clicked.connect(self.open_model_window)
        create_btn.clicked.connect(self.open_create_window)

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
        self.animation_title = QLabel("Видео источник")
        self.animation_title.setObjectName('animationTitle')
        self.animation_title.setFixedHeight(40)
        self.animation_title.setFixedWidth(600)
        self.animation_title.setAlignment(Qt.AlignCenter)
        self.animation_title.setContentsMargins(0, 0, 0, 0)

        self.animation_label = QLabel("Предпросмотр анимации")
        self.animation_label.setObjectName('animationLabel')
        self.animation_label.setFixedSize(600, 400)
        self.animation_label.setAlignment(Qt.AlignCenter)

        self.animation_label.mousePressEvent = self.open_video_player

        animation_shadow = QGraphicsDropShadowEffect()
        animation_shadow.setBlurRadius(25)
        animation_shadow.setOffset(0, 0)
        animation_shadow.setColor(QColor(80, 80, 80, 150))
        self.animation_label.setGraphicsEffect(animation_shadow)

        preview_layout = QVBoxLayout()
        preview_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        preview_layout.setSpacing(0)
        preview_layout.addWidget(self.animation_title)
        preview_layout.addWidget(self.animation_label)

        # Заголовок списка
        self.list_title = QLabel("Список анимаций")
        self.list_title.setObjectName('listTitle')
        self.list_title.setFixedHeight(40)
        self.list_title.setFixedWidth(600)
        self.list_title.setAlignment(Qt.AlignCenter)
        self.list_title.setFont(QFont("Arial", 12, QFont.Bold))
        self.list_title.setContentsMargins(0, 0, 0, 0)

        # Список анимаций
        self.animation_list = QListWidget()
        self.animation_list.setFixedWidth(600)
        self.animation_list.setMinimumHeight(300)
        self.animation_list.setStyleSheet("""
                    QListWidget {
                        background-color: #2c2c2c;
                        border: 1px solid #ccc;
                        padding: 0px;
                        font-size: 14px;
                        color: white;
                    }
                    QListWidget::item {
                        padding: 8px;
                        border: 1px solid #555555;
                    }
                    QListWidget::item:hover {
                        background-color: #6344BC;
                        color: black;
                    }
                    QListWidget::item:selected {
                        background-color: #D1B42A;
                        color: black;
                    }
                    QScrollBar:vertical {
                        background-color: #2c2c2c;
                        width: 15px;
                        margin: 0px;
                        border-radius: 5px;
                    }
                    QScrollBar::handle:vertical {
                        background: #6344BC;
                        min-height: 25px;
                        border-radius: 5px;
                    }
                    QScrollBar::handle:vertical:hover {
                        background: #6344BC;
                    }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        height: 0px;
                    }
                """)

        self.animation_list.setContentsMargins(0, 0, 0, 0)
        self.animation_list.currentRowChanged.connect(self.update_animation_preview)

        self.delete_button = GifButton("Source/icons/delete.gif")
        self.rename_button = GifButton("Source/icons/rename.gif")
        self.save_button = GifButton("Source/icons/save.gif")
        self.delete_button.setFixedSize(60, 60)
        self.rename_button.setFixedSize(60, 60)
        self.save_button.setFixedSize(60, 60)

        self.delete_button.clicked.connect(self.delete_animation)
        self.rename_button.clicked.connect(self.rename_animation)
        self.save_button.clicked.connect(self.save_animation)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.rename_button)
        button_layout.addWidget(self.delete_button)
        button_layout.setSpacing(10)

        list_layout = QVBoxLayout()
        list_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        list_layout.setSpacing(0)
        list_layout.addWidget(self.list_title)
        list_layout.addWidget(self.animation_list)

        list_and_button_layout = QVBoxLayout()
        list_and_button_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        list_and_button_layout.addLayout(list_layout)
        list_and_button_layout.setSpacing(10)
        list_and_button_layout.addLayout(button_layout)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        main_layout.addLayout(preview_layout)
        main_layout.addSpacing(25)
        main_layout.addLayout(list_and_button_layout)

        self.setLayout(main_layout)
        self.load_animations()

        return main_layout

    def delete_animation(self):
        index = self.animation_list.currentRow()
        if index < 0:
            return

        item = self.animation_list.item(index)
        name = item.text().split(' — ')[0]
        path = f"animated_models/{name}"
        video_path = path.strip('.fbx') + '.mp4'

        reply = QMessageBox().question(
            self,
            "Удаление",
            f"Удалить анимацию {name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                os.remove(path)
                if os.path.exists(video_path):
                    os.remove(video_path)
                self.video_path = None
                self.animation_list.takeItem(index)
                self.play_delete_animation()
                self.animation_label.setText("Анимация удалена")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

    def play_delete_animation(self):
        gif_path = "Source/icons/anim_deleted.gif"
        if not os.path.exists(gif_path):
            self.animation_label.setText("Удалено")  # fallback
            return

        movie = QMovie(gif_path)
        movie.setCacheMode(QMovie.CacheAll)
        movie.setScaledSize(self.animation_label.size() * 0.5)

        self.animation_label.setMovie(movie)
        movie.start()

        # Отслеживаем текущий кадр
        def on_frame_changed(frame_number):
            if frame_number == movie.frameCount() - 1:
                movie.stop()

        movie.frameChanged.connect(on_frame_changed)

    def rename_animation(self):
        index = self.animation_list.currentRow()
        if index < 0:
            return

        item = self.animation_list.item(index)
        old_name = item.text().split(' — ')[0]
        anim_time = item.text().split(' — ')[1]
        old_path = f"animated_models/{old_name}"

        old_video_name = old_name.strip('.fbx') + '.mp4'
        old_video_path = f"animated_models/{old_video_name}"

        new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя файла (с расширением .fbx):",
                                            text=old_name)
        if ok and new_name:
            new_path = f"animated_models/{new_name}"
            new_video_path = new_path.strip('.fbx') + '.mp4'
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Ошибка", "Файл с таким именем уже существует.")
                return
            try:
                os.rename(old_path, new_path)
                if os.path.exists(old_video_path):
                    os.rename(old_video_path, new_video_path)
                self.video_path = new_video_path
                item.setText(f'{new_name} — {anim_time}')
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

    def save_animation(self):
        index = self.animation_list.currentRow()
        if index < 0:
            return

        item = self.animation_list.item(index)
        name = item.text().split(' — ')[0]
        source_path = f"animated_models/{name}"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как", name, "FBX файлы (*.fbx)"
        )
        if save_path:
            try:
                import shutil
                shutil.copy2(source_path, save_path)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

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
                self.animation_label.setPixmap(QPixmap(qimage).scaled(self.animation_label.size(), Qt.KeepAspectRatio))
            else:
                self.animation_label.setText("Не загрузить видео")
        else:
            self.animation_label.setText("Ошибка загрузки видео")
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

    def load_animations(self):
        animations_dir = "animated_models"
        if not os.path.exists(animations_dir):
            return

        animation_files = [
            f for f in os.listdir(animations_dir)
            if f.lower().endswith('.fbx')  # можно расширить по необходимости
        ]

        self.animation_list.clear()
        for f in animation_files:
            duration = self.get_fake_duration(f)  # пока фиктивная
            self.animation_list.addItem(f"{f} — {duration}")

        if animation_files:
            self.update_animation_preview(0)
        else:
            self.animation_label.setText("Нет доступных анимаций")

    def get_fake_duration(self, filename):
        # TODO: Заменить на реальное чтение длительности из .fbx
        return "3.2 сек"

    def update_animation_preview(self, index):
        if index < 0:
            return

        animation_name = self.animation_list.item(index).text().split(' — ')[0]
        base_name = os.path.splitext(animation_name)[0]

        self.video_path = f"animated_models/{base_name}.mp4"

        if self.video_path is not None and os.path.exists(self.video_path):
            self.display_first_frame(self.video_path)
        else:
            self.video_path = None
            self.animation_label.setText(f"Анимация: {animation_name}\n(Нет изображения)")
            return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResultsWindow()
    window.show()
    sys.exit(app.exec_())
