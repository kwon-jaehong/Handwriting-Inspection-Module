from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog, QLabel, QInputDialog,QTreeView
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
        
        self.main_wiget = QWidget()
        self.setCentralWidget(self.main_wiget)
        self.setGeometry(0,100,1000,700)
        self.setWindowTitle("메인창")
        

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
        
        
        # https://stackoverflow.com/questions/51456403/mouseover-event-for-a-pyqt5-label
        # self.img_view.enterEvent = lambda e: print("들어옴")
        # self.img_view.leaveEvent = lambda e: print("떠남")
        

        self.left_layout = QVBoxLayout()
        

        # QTreewidget 생성 및 설정
        # self.pdf_file_tree_wiget = QTreeWidget(self)       
        self.pdf_file_tree_wiget = Treeview(self)            
        
        self.pdf_file_tree_wiget.setAutoScroll(True)
        self.pdf_file_tree_wiget.setAlternatingRowColors(True)
        self.pdf_file_tree_wiget.header().setVisible(False)
        self.pdf_file_tree_wiget.setColumnWidth(0,800)
        self.pdf_file_tree_wiget.itemClicked.connect(self.click_list_item)
        
        
        self.left_layout.addWidget(self.pdf_file_tree_wiget)     
        
        
        groupbox = QGroupBox('도구')


            
        
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
            pass
                
            
   
        
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
    sys.exit(app.exec_())
