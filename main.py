import sys

from PyQt5.QtWidgets import QApplication

from create_animation_window import CreateAnimationWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CreateAnimationWindow()
    window.show()
    sys.exit(app.exec_())
