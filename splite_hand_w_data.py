from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication,QSlider, QMenu, QMenuBar, QAction, QFileDialog, QLabel, QInputDialog,QTreeView
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint,QBuffer,QEvent

from PIL import Image
import cv2
import numpy


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.crate_main_menubar()
        
        self.main_wiget = self.create_main_wiget()
        self.setCentralWidget(self.main_wiget)
        self.setGeometry(0,100,1000,700)
        self.setWindowTitle("메인창")
        
        self.paint_img = QImage()

    # 메뉴 생성
    def crate_main_menubar(self):
        self.mainMenu = self.menuBar()
        self.menubar = self.mainMenu.addMenu("메뉴")


        self.openAction = QAction('jpg 이미지 불러오기', self)
        self.openAction.triggered.connect(self.open_img_file)
        self.menubar.addAction(self.openAction)
        
        self.closeAction = QAction('나가기', self)
        self.closeAction.triggered.connect(self.close)
        self.menubar.addAction(self.closeAction)
       
    # 작업창 메인 레이아웃 위젯 생성
    def create_main_wiget(self):
        main_wiget = QWidget()
        
        main_layout = QHBoxLayout()
        self.img_view = CView(self)

        self.left_layout = QVBoxLayout()
        # groupbox = QGroupBox('도구')           
        
        self.threshold_slider = QSlider(Qt.Horizontal, self)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.set
        self.threshold_slider.setSingleStep(1)
        self.threshold_label = QLabel('0')
        
        
        self.left_layout.addWidget(self.threshold_slider)
        
        
        main_layout.addLayout(self.left_layout,stretch=1)
        main_layout.addWidget(self.img_view,stretch=5)    
          
        
        main_wiget.setLayout(main_layout)
        return main_wiget
   
       ## 손글씨 이미지 파일 load
    def open_img_file(self):
        img_list = ('jpg','png')
        # filename,_ = QFileDialog.getOpenFileNames(self, 'Select Multi File', 'default', 'All Files (*)')
        filename,_ = QFileDialog.getOpenFileNames(self, 'Select Multi File', 'default')
        filename = [file for file in filename if file.lower().endswith(img_list)]
        
        if len(filename) > 0:
            pil_img = Image.open(filename[0])
            cv_img = cv2.imread(filename[0])
            
            self.img_view.paint_img = QImage(cv_img, cv_img.shape[1], cv_img.shape[0],cv_img.strides[0],QImage.Format_RGB888) 
            self.img_view.overlay_img = QtGui.QImage(self.img_view.paint_img.size(), QtGui.QImage.Format_ARGB32)
            self.img_view.overlay_img.fill(QtCore.Qt.transparent)
            self.img_view.scene = QGraphicsScene()
            self.img_view.item = QGraphicsPixmapItem(QPixmap.fromImage(self.img_view.paint_img))
            self.img_view.scene.addItem(self.img_view.item)
            self.img_view.setScene(self.img_view.scene)
        self.update()

class CView(QGraphicsView):
    def __init__(self,parent):
        super().__init__(parent)       
        self.setAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.setStyleSheet("background:transparent;")       
        
        self.parent = parent
        
        # 1 그냥, 2 그리기, 3 지우기 모드
        self.mode = 1
        
        self.change = False
        self.drawing = False
        self.brushSize = 20
        self.brushColor = QtGui.QColor(QtCore.Qt.red)
        self.lastPoint = QtCore.QPoint()

        self.paint_img = QImage()
        
        self.overlay_img = QtGui.QImage(self.paint_img.size(), QtGui.QImage.Format_ARGB32)
        self.overlay_img.fill(QtCore.Qt.transparent)
        
        self.scene = QGraphicsScene()
        self.item = QGraphicsPixmapItem(QPixmap.fromImage(self.paint_img))

        self.scene.addItem(self.item)
        self.setScene(self.scene)
            
   
        
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
    sys.exit(app.exec_())
