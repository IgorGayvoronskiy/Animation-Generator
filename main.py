import sys
import tensorflow as tf
import tensorflow_hub as tf_hub
import mediapipe as mp

from PyQt5.QtCore import QPropertyAnimation, QPoint, QRect, QParallelAnimationGroup
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QSizePolicy, QMainWindow

from useful_classes import resource_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AniMotion")
        self.setWindowIcon(QIcon(resource_path("Source/Images/AM2.ico")))
        self.resize(800, 600)

        self.stack = QStackedWidget(self)
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.first_window = CreateAnimationWindow(self)
        self.results_window = ResultsWindow(self)
        self.model_window = ModelDownloadWindow(self)

        self.stack.addWidget(self.first_window)
        self.stack.addWidget(self.results_window)
        self.stack.addWidget(self.model_window)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.stack)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.stack.setCurrentIndex(0)
        self.showMaximized()

    def get_res_win(self):
        return self.results_window

    def get_create_win(self):
        return self.first_window

    def slide_to(self, index):
        current_index = self.stack.currentIndex()
        if current_index == index:
            return

        current_widget = self.stack.currentWidget()
        next_widget = self.stack.widget(index)

        width = self.stack.frameRect().width()

        if current_index == 0:
            direction = 1 if index == 2 else -1
        elif current_index == 1:
            direction = 1
        elif current_index == 2:
            direction = -1
        else:
            direction = 1

        offset = width * direction

        next_widget.setGeometry(QRect(offset, 0, width, self.stack.height()))
        next_widget.show()

        anim_current = QPropertyAnimation(current_widget, b"pos", self)
        anim_current.setDuration(300)
        anim_current.setStartValue(current_widget.pos())
        anim_current.setEndValue(QPoint(-offset, 0))

        anim_next = QPropertyAnimation(next_widget, b"pos", self)
        anim_next.setDuration(300)
        anim_next.setStartValue(QPoint(offset, 0))
        anim_next.setEndValue(QPoint(0, 0))

        # Сгруппируем анимации, чтобы следить за завершением
        group = QParallelAnimationGroup(self)
        group.addAnimation(anim_current)
        group.addAnimation(anim_next)

        def on_animation_finished():
            self.stack.setCurrentIndex(index)

        group.finished.connect(on_animation_finished)
        group.start()


if __name__ == "__main__":
    from create_animation_window import CreateAnimationWindow
    from model_download_window import ModelDownloadWindow
    from results_window import ResultsWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
