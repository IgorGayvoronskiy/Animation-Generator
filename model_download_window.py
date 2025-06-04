import os
import sys

import upload_model_window
from multiprocessing import Process
from useful_classes import TrapezoidWidget, JsonImageDropLabel, ProgressOverlay, GifButton, SkeletonWorker
from PyQt5.QtCore import Qt, QCoreApplication, QThread
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPalette, QBrush, QImage, QFont, QMovie
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QMainWindow, QGraphicsDropShadowEffect, QComboBox, QProgressBar, QMessageBox, QListWidget, QInputDialog
)

import create_animation_window
import results_window


class ModelDownloadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.thread = None
        self.upload_window = None
        self.download_button = None
        self.rename_button = None
        self.delete_button = None
        self.model_list = None
        self.list_title = None
        self.model_label = None
        self.model_title = None
        self.progress_bar = None
        self.json_path = None
        self.image_path = None
        self.accept_json_mode = False
        self.skeleton_create_btn = None
        self.toggle_mode_btn = None
        self.choose_image_btn = None
        self.rigging_label = None
        self.create_window = None
        self.res_window = None
        self.setWindowTitle("Загрузка Моделей")
        self.initUI()
        self.progress_overlay = ProgressOverlay(self)
        self.showMaximized()

    def open_res_window(self):
        self.res_window = results_window.ResultsWindow()
        self.res_window.show()
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

            QLabel#imageLabel, QLabel#modelLabel{
                color: #C4B5E0;
                font-size: 14px;
                border: 2px solid #777;
                border-radius: 16px;
                background-color: #2b2b2b;
                padding: 10px;
            }
            
            QLabel#listTitle, QLabel#modelTitle{
                color: black
                font-size: 14px;
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

        cur_window_btn = upload_btn

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
        layout = QHBoxLayout()

        # Блок загрузки видео
        image_frame = QVBoxLayout()
        self.rigging_label = JsonImageDropLabel(self)
        self.rigging_label.setObjectName("imageLabel")
        self.rigging_label.setFixedSize(500, 500)
        self.rigging_label.setAlignment(Qt.AlignCenter)

        self.rig_label_reset()

        rig_shadow = QGraphicsDropShadowEffect()
        rig_shadow.setBlurRadius(25)
        rig_shadow.setOffset(0, 0)
        rig_shadow.setColor(QColor(0, 0, 0, 180))
        self.rigging_label.setGraphicsEffect(rig_shadow)

        self.choose_image_btn = QPushButton("Выбрать изображение")
        self.skeleton_create_btn = QPushButton("Создать скелет")
        self.choose_image_btn.clicked.connect(self.select_json_img_file)
        self.skeleton_create_btn.clicked.connect(self.create_sceleton)

        rig_container = QVBoxLayout()
        rig_header = QHBoxLayout()
        buttons_header = QHBoxLayout()

        buttons_header.addWidget(self.choose_image_btn)
        buttons_header.addWidget(self.skeleton_create_btn)

        self.toggle_mode_btn = QPushButton("Режим: Изображение")
        self.toggle_mode_btn.setFixedWidth(200)
        self.toggle_mode_btn.clicked.connect(self.toggle_input_mode)

        rig_header.addStretch()
        rig_header.addWidget(self.toggle_mode_btn)

        rig_container.addLayout(rig_header)
        rig_container.addWidget(self.rigging_label)

        image_frame.addLayout(rig_container)
        image_frame.addLayout(buttons_header)

        # Блок загрузки 3D модели
        self.model_title = QLabel("Изображение модели")
        self.model_title.setObjectName('modelTitle')
        self.model_title.setFixedHeight(40)
        self.model_title.setFixedWidth(600)
        self.model_title.setAlignment(Qt.AlignCenter)
        self.model_title.setContentsMargins(0, 0, 0, 0)

        self.model_label = QLabel("Изображение модели")
        self.model_label.setObjectName('modelLabel')
        self.model_label.setFixedSize(600, 400)
        self.model_label.setAlignment(Qt.AlignCenter)

        model_shadow = QGraphicsDropShadowEffect()
        model_shadow.setBlurRadius(25)
        model_shadow.setOffset(0, 0)
        model_shadow.setColor(QColor(80, 80, 80, 150))
        self.model_label.setGraphicsEffect(model_shadow)

        preview_layout = QVBoxLayout()
        preview_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        preview_layout.setSpacing(0)
        preview_layout.addWidget(self.model_title)
        preview_layout.addWidget(self.model_label)

        self.list_title = QLabel("Список моделей")
        self.list_title.setObjectName('listTitle')
        self.list_title.setFixedHeight(40)
        self.list_title.setFixedWidth(600)
        self.list_title.setAlignment(Qt.AlignCenter)
        self.list_title.setContentsMargins(0, 0, 0, 0)

        # Список загруженных 3D моделей
        self.model_list = QListWidget()
        self.model_list.setFixedWidth(600)
        self.model_list.setMinimumHeight(300)
        self.model_list.setStyleSheet("""
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

        self.model_list.setContentsMargins(0, 0, 0, 0)
        self.model_list.currentRowChanged.connect(self.update_model_preview)

        self.delete_button = GifButton("Source/icons/delete.gif")
        self.rename_button = GifButton("Source/icons/rename.gif")
        self.download_button = GifButton("Source/icons/save.gif")
        self.delete_button.setFixedSize(60, 60)
        self.rename_button.setFixedSize(60, 60)
        self.download_button.setFixedSize(60, 60)

        self.delete_button.clicked.connect(self.delete_3Dmodel)
        self.rename_button.clicked.connect(self.rename_model)
        self.download_button.clicked.connect(self.upload_model)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.rename_button)
        button_layout.addWidget(self.delete_button)
        button_layout.setSpacing(10)

        list_layout = QVBoxLayout()
        list_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        list_layout.setSpacing(0)
        list_layout.addWidget(self.list_title)
        list_layout.addWidget(self.model_list)

        list_and_button_layout = QVBoxLayout()
        list_and_button_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        list_and_button_layout.addLayout(list_layout)
        list_and_button_layout.setSpacing(10)
        list_and_button_layout.addLayout(button_layout)

        model_frame = QVBoxLayout()
        model_frame.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        model_frame.addLayout(preview_layout)
        model_frame.addSpacing(25)
        model_frame.addLayout(list_and_button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(image_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(model_frame)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.setLayout(layout)
        self.load_models()

        return layout

    def rig_label_reset(self):
        default_pixmap = QPixmap("Source/Images/no_image.jpg")
        if not default_pixmap.isNull():
            self.rigging_label.setPixmap(default_pixmap.scaled(self.rigging_label.size(), Qt.KeepAspectRatio))
        else:
            self.rigging_label.setText("Изображение не выбрано")

    def toggle_input_mode(self):
        self.accept_json_mode = not self.accept_json_mode

        if self.accept_json_mode:
            self.image_path = None
            self.toggle_mode_btn.setText("Режим: JSON")
            self.choose_image_btn.setText("Выбрать JSON файл")
            self.rigging_label.setText("JSON не выбран")
        else:
            self.json_path = None
            self.toggle_mode_btn.setText("Режим: Изображение")
            self.choose_image_btn.setText("Выбрать изображение")
            self.rig_label_reset()

    def handle_json_img_file_drop(self, file_path):
        if not os.path.exists(file_path):
            return

        if self.accept_json_mode:
            if file_path.endswith(".json"):
                self.rigging_label.setText(f"JSON файл загружен:\n{os.path.basename(file_path)}")
            else:
                self.rigging_label.setText("Неверный формат файла (только .json)")
        else:
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_path = file_path
                default_pixmap = QPixmap(file_path)
                if not default_pixmap.isNull():
                    self.rigging_label.setPixmap(default_pixmap.scaled(self.rigging_label.size(), Qt.KeepAspectRatio))
                else:
                    self.rigging_label.setText("Изображение не выбрано")
            else:
                self.rigging_label.setText("Неверный формат файла (только изображение)")

    def select_json_img_file(self):
        if self.accept_json_mode:
            file, _ = QFileDialog.getOpenFileName(self, "Выберите JSON", "", "JSON Files (*.json)")
            if file:
                self.handle_json_img_file_drop(file)
                self.json_path = file
        else:
            file, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Video Files (*.png *.jpg *.jpeg)")
            if file:
                self.rigging_label.setText("Загрузка изображения...")
                default_pixmap = QPixmap(file)
                if not default_pixmap.isNull():
                    self.rigging_label.setPixmap(default_pixmap.scaled(self.rigging_label.size(), Qt.KeepAspectRatio))
                else:
                    self.rigging_label.setText("Ошибка загрузки изображения")
                self.image_path = file

    def create_sceleton(self):
        if self.json_path is not None or self.image_path is not None:
            self.start_skeleton_thread()

    def start_skeleton_thread(self):
        file_path = self.json_path if self.accept_json_mode else self.image_path
        from_json = self.accept_json_mode

        self.thread = QThread()
        self.worker = SkeletonWorker(file_path, from_json)
        self.worker.moveToThread(self.thread)

        # Связь сигналов
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_overlay.update_progress)
        self.worker.finished.connect(self.on_skeleton_created)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.progress_overlay.show_overlay("Создание скелета...")
        self.thread.start()

    def on_skeleton_created(self, fbx_path):
        self.progress_overlay.hide_overlay()

        if fbx_path:
            name = fbx_path.split('/')[-1]
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить как", name, "FBX файлы (*.fbx)"
            )
            if save_path:
                try:
                    import shutil
                    shutil.copy2(fbx_path, save_path)
                except Exception as e:
                    QMessageBox.warning(self, "Ошибка при копировании", str(e))
                finally:
                    import os
                    os.remove(fbx_path)

            if self.accept_json_mode:
                self.json_path = None
                self.rigging_label.setText("JSON не выбран")
            else:
                self.image_path = None
                self.rig_label_reset()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось создать скелет. Попробуйте другое изображение.")

    def delete_3Dmodel(self):
        index = self.model_list.currentRow()
        if index < 0:
            return

        item = self.model_list.item(index)
        name = item.text()
        path = f"Source/rigged_models/{name}"
        image_path = path.strip('.fbx') + '.png'

        reply = QMessageBox().question(
            self,
            "Удаление",
            f"Удалить анимацию {name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                os.remove(path)
                if os.path.exists(image_path):
                    os.remove(image_path)
                self.model_list.takeItem(index)
                # self.play_delete_animation()
                self.model_label.setText("Модель удалена")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

    def play_delete_animation(self):
        gif_path = "Source/icons/anim_deleted.gif"
        if not os.path.exists(gif_path):
            self.model_label.setText("Удалено")  # fallback
            return

        movie = QMovie(gif_path)
        movie.setCacheMode(QMovie.CacheAll)
        movie.setScaledSize(self.model_label.size() * 0.5)

        self.model_label.setMovie(movie)
        movie.start()

        # Отслеживаем текущий кадр
        def on_frame_changed(frame_number):
            if frame_number == movie.frameCount() - 1:
                movie.stop()

        movie.frameChanged.connect(on_frame_changed)

    def rename_model(self):
        index = self.model_list.currentRow()
        if index < 0:
            return
        item = self.model_list.item(index)
        old_name = item.text()
        old_path = f"Source/rigged_models/{old_name}"

        old_model_img_path = old_path.strip('.fbx') + '.png'

        new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя файла (с расширением .fbx):",
                                            text=old_name)
        if ok and new_name:
            new_path = f"Source/rigged_models/{new_name}"
            new_model_img_path = new_path.rstrip('.fbx') + '.png'
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Ошибка", "Файл с таким именем уже существует.")
                return
            try:
                os.rename(old_path, new_path)
                if os.path.exists(old_model_img_path):
                    os.rename(old_model_img_path, new_model_img_path)
                item.setText(new_name)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", str(e))

    def upload_model(self):
        self.upload_window = upload_model_window.UploadWindow()
        self.upload_window.model_uploaded.connect(self.load_models)
        self.upload_window.show()

    def load_models(self):
        model_dir = "Source/rigged_models"
        if not os.path.exists(model_dir):
            return
        model_files = [
            f for f in os.listdir(model_dir)
            if f.lower().endswith('.fbx')  # возможно надо будет расширить
        ]

        self.model_list.clear()
        for f in model_files:
            self.model_list.addItem(f)

        if model_files:
            self.update_model_preview(0)
        else:
            self.model_label.setText("Нет доступных моделей")

    def update_model_preview(self, index):
        if index < 0:
            return

        model_name = self.model_list.item(index).text()
        base_name = os.path.splitext(model_name)[0]

        model_image_path = f"Source/rigged_models/{base_name}.png"

        if model_image_path is not None and os.path.exists(model_image_path):
            default_pixmap = QPixmap(model_image_path)
            if not default_pixmap.isNull():
                self.model_label.setPixmap(default_pixmap.scaled(self.model_label.size(), Qt.KeepAspectRatio))
            else:
                self.model_label.setText("Нет моделей")
        else:
            self.model_label.setText(f"Модель: {model_name}\n(Нет изображения)")
            return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelDownloadWindow()
    window.show()
    sys.exit(app.exec_())
