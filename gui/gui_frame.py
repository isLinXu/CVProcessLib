import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageProcessing(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Image Processing')

        # 添加控件
        self.label = QLabel(self)
        self.label.setGeometry(20, 20, 400, 400)

        self.openBtn = QPushButton('打开', self)
        self.openBtn.setGeometry(20, 450, 80, 30)
        self.openBtn.clicked.connect(self.openImage)

        self.saveBtn = QPushButton('保存', self)
        self.saveBtn.setGeometry(120, 450, 80, 30)
        self.saveBtn.clicked.connect(self.saveImage)

        self.processBtn = QPushButton('处理', self)
        self.processBtn.setGeometry(220, 450, 80, 30)
        self.processBtn.clicked.connect(self.processImage)

        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(20, 500, 80, 30)
        self.comboBox.addItem('模糊')
        self.comboBox.addItem('边缘检测')
        self.comboBox.addItem('灰度化')
        self.comboBox.currentIndexChanged.connect(self.processImage)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setGeometry(120, 500, 80, 30)
        self.lineEdit.setText('3')
        self.lineEdit.setAlignment(Qt.AlignCenter)
        self.lineEdit.editingFinished.connect(self.processImage)

        self.label1 = QLabel('参数:', self)
        self.label1.setGeometry(220, 500, 50, 30)

        self.label2 = QLabel('图像路径:', self)
        self.label2.setGeometry(20, 550, 80, 30)

        self.lineEdit2 = QLineEdit(self)
        self.lineEdit2.setGeometry(120, 550, 400, 30)

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.bmp)')
        if fileName:
            self.lineEdit2.setText(fileName)
            self.image = cv2.imread(fileName)
            self.showImage()

    def saveImage(self):
        if hasattr(self, 'image'):
            fileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.bmp)')
            if fileName:
                cv2.imwrite(fileName, self.image)

    def processImage(self):
        if hasattr(self, 'image'):
            index = self.comboBox.currentIndex()
            if index == 0:
                ksize = int(self.lineEdit.text())
                self.image = cv2.blur(self.image, (ksize, ksize))
                self.showImage()
            elif index == 1:
                self.image = cv2.Canny(self.image, 100, 200)
                self.showImage()
            elif index == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.showImage()

    def showImage(self):
        # height, width = self.image.shape
        size = self.image.shape

        height = size[1]  # 宽度

        width = size[0]  # 高度
        bytesPerLine = 3 * width
        qImg = QImage(self.image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessing()
    ex.show()
    sys.exit(app.exec_())
