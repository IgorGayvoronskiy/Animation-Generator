import ctypes
import os
import sys
import vlc

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog,
    QMainWindow, QGraphicsDropShadowEffect, QToolButton, QStyle, QMessageBox, QFrame, QSlider, QComboBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPalette, QBrush, QImage, QPainterPath, QPainter, QMovie
from PyQt5.QtCore import Qt, QUrl, QTimer, QCoreApplication, QEvent

vlc_path = r"C:\Program Files\VideoLAN\VLC\libvlc.dll"  # проверь путь
ctypes.CDLL(vlc_path)


class GifButton(QPushButton):
    def __init__(self, gif_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gif_label = QLabel(self)
        self.gif_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setVisible(True)

        self.movie = QMovie(gif_path)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.jumpToFrame(0)
        self.gif_label.setMovie(self.movie)

        self.installEventFilter(self)

        # Задержка, чтобы подождать до тех пор, пока кнопка не получит реальный размер
        QTimer.singleShot(0, self.init_gif)

    def init_gif(self):
        self.movie.setScaledSize(self.size() * 0.6)
        self.gif_label.setGeometry(0, 0, self.width(), self.height())
        self.movie.start()
        self.movie.stop()  # Показать первый кадр

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.gif_label.setGeometry(0, 0, self.width(), self.height())
        self.movie.setScaledSize(self.size() * 0.6)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            self.movie.start()
        elif event.type() == QEvent.Leave:
            self.movie.stop()
            self.movie.jumpToFrame(0)
        return super().eventFilter(obj, event)


class TrapezoidWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("navBar")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        side_offset = 40  # глубина диагональных краёв

        path = QPainterPath()
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right(), rect.top())
        path.lineTo(rect.right() - side_offset, rect.bottom())
        path.lineTo(rect.left() + side_offset, rect.bottom())
        path.closeSubpath()

        gradient_color = QColor(0, 210, 255, 180)
        painter.setBrush(QColor('#2b2b2b'))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        super().paintEvent(event)


class VideoPlayerPopup(QWidget):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._released = False  # Флаг для избежания двойного освобождения

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(1000, 600)

        # Тень для внешнего окна
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

        # Контейнер
        container = QFrame(self)
        container.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-radius: 24px;
            }
        """)
        container.setGeometry(0, 0, self.width(), self.height())

        layout = QVBoxLayout(container)
        layout.setContentsMargins(43, 43, 43, 43)
        layout.setSpacing(16)

        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black; border-radius: 16px;")
        self.video_frame.setMinimumHeight(420)
        layout.addWidget(self.video_frame)

        # Шкала перемотки
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #222;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #6344BC;
                border: 1px solid #555;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            QSlider::sub-page:horizontal {
                background: #C4B5E0;
                border-radius: 4px;
            }

            QSlider::add-page:horizontal {
                background: #444;
                border-radius: 4px;
            }
        """)

        layout.addWidget(self.slider)

        self.slider_timer = QTimer(self)
        self.slider_timer.setInterval(1)
        self.slider_timer.timeout.connect(self.update_slider)
        self.slider_timer.start()

        # Кнопки управления
        controls = QHBoxLayout()
        controls.setSpacing(14)
        controls.setAlignment(Qt.AlignCenter)

        self.back_btn = QPushButton("↻")
        self.play_pause_btn = QPushButton("▐▐ ")
        self.forward_btn = QPushButton("↺")

        for btn in [self.back_btn, self.play_pause_btn, self.forward_btn]:
            btn.setMinimumHeight(40)
            btn.setMinimumWidth(80)

        self.back_btn.clicked.connect(self.skip_back)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.forward_btn.clicked.connect(self.skip_forward)

        controls.addWidget(self.back_btn)
        controls.addWidget(self.play_pause_btn)
        controls.addWidget(self.forward_btn)
        layout.addLayout(controls)

        # Кнопка закрытия
        self.close_button = QToolButton(self)
        self.close_button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
            QToolButton {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 6px;
            }
            QToolButton:hover {
                background-color: rgba(255, 0, 0, 0.25);
            }
        """)
        self.close_button.setFixedSize(32, 32)
        self.close_button.move(self.width() - 48, 16)
        self.close_button.raise_()

        # VLC
        self.instance = vlc.Instance()
        self.mediaplayer = self.instance.media_player_new()

        if sys.platform.startswith('linux'):
            self.mediaplayer.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.mediaplayer.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform == "darwin":
            self.mediaplayer.set_nsobject(int(self.video_frame.winId()))

        if os.path.exists(video_path):
            media = self.instance.media_new(video_path)
            self.mediaplayer.set_media(media)
            self.mediaplayer.event_manager().event_attach(
                vlc.EventType.MediaPlayerEndReached, self.video_ended
            )
            self.mediaplayer.play()
        else:
            self.close()

    def update_slider(self):
        if self.mediaplayer is None:
            return

        if self.mediaplayer.get_state() in (vlc.State.Ended, vlc.State.NothingSpecial):
            return

        duration = self.mediaplayer.get_length()
        if duration > 0:
            pos = self.mediaplayer.get_time()
            self.slider.blockSignals(True)
            self.slider.setValue(int(pos / duration * 1000))
            self.slider.blockSignals(False)

    def set_position(self, value):
        duration = self.mediaplayer.get_length()
        if duration > 0:
            new_time = int(duration * value / 1000)
            if self.mediaplayer.get_state() == vlc.State.Ended:
                media = self.mediaplayer.get_media()
                self.mediaplayer.stop()
                self.mediaplayer.set_media(media)
                self.mediaplayer.play()
                QTimer.singleShot(200, lambda: self.mediaplayer.set_time(new_time))
                self.play_pause_btn.setText("▐▐")
            else:
                self.mediaplayer.set_time(new_time)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def toggle_play_pause(self):
        state = self.mediaplayer.get_state()
        if state == vlc.State.Ended:
            self.mediaplayer.stop()
            self.mediaplayer.set_time(0)
            self.mediaplayer.play()
            self.play_pause_btn.setText("▐▐")
        elif self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            self.play_pause_btn.setText("▶")
        else:
            self.mediaplayer.play()
            self.play_pause_btn.setText("▐▐")

    def video_ended(self, event=None):
        self.play_pause_btn.setText("▶")

    def skip_forward(self):
        current_time = self.mediaplayer.get_time()
        duration = self.mediaplayer.get_length()
        if duration > 0 and current_time + 10000 < duration:
            self.mediaplayer.set_time(current_time + 10000)
        else:
            self.mediaplayer.set_time(duration - 100)

    def skip_back(self):
        state = self.mediaplayer.get_state()
        if state == vlc.State.Ended:
            media = self.mediaplayer.get_media()
            self.mediaplayer.stop()
            self.mediaplayer.set_media(media)
            duration = media.get_duration()
            back_time = max(0, duration - 10000)
            self.mediaplayer.play()
            QTimer.singleShot(200, lambda: self.mediaplayer.set_time(back_time))
            self.play_pause_btn.setText("▐▐")
        else:
            self.mediaplayer.set_time(max(0, self.mediaplayer.get_time() - 10000))

    def closeEvent(self, event):
        try:
            if self.slider_timer.isActive():
                self.slider_timer.stop()
        except Exception:
            pass

        QTimer.singleShot(150, self.release_resources)

        if self.parent_window:
            self.parent_window.video_player_window = None

        super().closeEvent(event)

    def release_resources(self):
        if self._released:
            return
        self._released = True
        try:
            if self.mediaplayer:
                self.mediaplayer.stop()
                self.mediaplayer.release()
                self.mediaplayer = None
            if self.instance:
                self.instance.release()
                self.instance = None
        except Exception as e:
            print("Ошибка при освобождении ресурсов:", e)


class VideoDropLabel(QLabel):
    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if self.main_window.accept_json_mode and file_path.endswith('.json'):
                event.acceptProposedAction()
            elif not self.main_window.accept_json_mode and file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.main_window.handle_video_drop(file_path)


class JsonImageDropLabel(QLabel):
    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if self.main_window.accept_json_mode and file_path.endswith('.json'):
                event.acceptProposedAction()
            elif not self.main_window.accept_json_mode and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.main_window.handle_json_img_file_drop(file_path)


class ImageDropLabel(QLabel):
    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.main_window.handle_img_file_drop(file_path)


class FbxDropLabel(QLabel):
    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.endswith('.fbx'):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.main_window.handle_fbx_file_drop(file_path)


class ProgressOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setVisible(False)

        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setStyleSheet("""
            background-color: rgba(40, 40, 40, 220);
            border-radius: 12px;
        """)
        container.setFixedSize(300, 150)

        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.setSpacing(15)

        self.label = QLabel("Обработка...")
        self.label.setStyleSheet("color: white; font-size: 16px;")
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 6px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #6344BC;
            }
        """)

        container_layout.addWidget(self.label)
        container_layout.addWidget(self.progress_bar)
        layout.addWidget(container)

    def center_in_parent(self):
        if self.parent():
            parent_rect = self.parent().geometry()
            new_x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            new_y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(new_x, new_y)

    def show_overlay(self, message="Обработка..."):
        self.label.setText(message)
        self.progress_bar.setValue(0)
        self.center_in_parent()
        self.setVisible(True)
        self.raise_()

    def hide_overlay(self):
        self.setVisible(False)

    def update_progress(self, value, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)
        self.update_message(message)
        QApplication.processEvents()

    def update_message(self, message):
        self.label.setText(message)


from PyQt5.QtCore import QObject, pyqtSignal


class SkeletonWorker(QObject):
    progress = pyqtSignal(int, int, str)  # value, total, message
    finished = pyqtSignal(str)  # путь к FBX или пустая строка

    def __init__(self, file_path, from_json):
        super().__init__()
        self.file_path = file_path
        self.from_json = from_json

    def run(self):
        from bind_pose_creator_tool import create_bind_pose
        try:
            path = create_bind_pose(
                file_path=self.file_path,
                from_json=self.from_json,
                progress_callback=self.progress.emit
            )
            self.finished.emit(path or "")
        except Exception as e:
            print(f"Ошибка: {e}")
            self.finished.emit("")


class AnimationWorker(QObject):
    progress = pyqtSignal(int, int, str)  # value, total, message
    finished = pyqtSignal(str)  # путь к FBX или пустая строка

    def __init__(self, file_path, model_path, from_json, frame_rate, metr_model_path):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.from_json = from_json
        self.frame_rate = frame_rate
        self.metr_model_path = metr_model_path

    def run(self):
        from animation_generator import create_animation
        try:
            path = create_animation(
                key_points_data_path=self.file_path,
                model_path=self.model_path,
                from_json=self.from_json,
                progress_callback=self.progress.emit,
                frame_rate=self.frame_rate,
                mmodel_path=self.metr_model_path
            )
            self.finished.emit(path or '')
        except Exception as e:
            print(f"Ошибка: {e}")
            self.finished.emit('')
