import sys
import tensorflow as tf
import tensorflow_hub as tfhub
import mediapipe as mp

from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    from create_animation_window import CreateAnimationWindow

    app = QApplication(sys.argv)
    window = CreateAnimationWindow()
    window.show()
    sys.exit(app.exec_())
