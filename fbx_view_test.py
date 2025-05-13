import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QCoreApplication, Qt

import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-software-rasterizer"

QT_OPENGL='software python fbx_view_test.py'



class ThreeJSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Three.js in PyQt")
        self.setGeometry(100, 100, 1200, 800)

        webview = QWebEngineView()
        webview.load(QUrl('http://localhost:8000/examples/webgl_loader_fbx.html'))

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(webview)

        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ThreeJSViewer()
    viewer.show()
    sys.exit(app.exec_())
