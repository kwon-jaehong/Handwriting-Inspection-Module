from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication,QSlider, QMenu, QMenuBar, QAction, QFileDialog, QLabel, QInputDialog,QTreeView
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint,QBuffer,QEvent

from PIL import Image
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = ""
        self.crate_main_menubar()
        self.main_wiget = self.create_main_wiget()
        self.setCentralWidget(self.main_wiget)
        self.setGeometry(0,100,1000,700)
        self.setWindowTitle("메인창")
        
        self.paint_img = QImage()

        self.image_splite_wiget = self.create_image_splite_wiget()
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
        
        
        self.tool_layout = QVBoxLayout()
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(0)
        self.tool_groupbox = QGroupBox('커널 사이즈')
        self.kernel_size_value = QLabel('1')        
        self.kernel_size_slider = QSlider(Qt.Horizontal, self)
        self.kernel_size_slider.valueChanged.connect(self.kernel_size_changed)
        self.kernel_size_slider.sliderReleased.connect(self.value_end)
        self.kernel_size_slider.setRange(1, 50)
        self.kernel_size_slider.setValue(3)
        self.kernel_size_slider.setSingleStep(1)
        self.tool_layout.addWidget(self.kernel_size_slider)
        self.tool_layout.addWidget(self.kernel_size_value)
        self.tool_groupbox.setLayout(self.tool_layout)
        self.left_layout.addWidget(self.tool_groupbox)
        
        
        self.canny_layout = QVBoxLayout()
        self.canny_layout.setContentsMargins(0, 0, 0, 0)
        self.canny_layout.setSpacing(0)
        self.canny_groupbox = QGroupBox('canny edge min & max 설정')
        self.canny_min_value = QLabel('0')        
        self.canny_min_slider = QSlider(Qt.Horizontal, self)
        self.canny_min_slider.valueChanged.connect(self.canny_min_changed)
        self.canny_min_slider.sliderReleased.connect(self.value_end)
        self.canny_min_slider.setRange(0, 999)
        self.canny_min_slider.setValue(50)
        self.canny_min_slider.setSingleStep(1)
        self.canny_max_value = QLabel('0')    
        self.canny_max_slider = QSlider(Qt.Horizontal, self)
        self.canny_max_slider.valueChanged.connect(self.canny_max_changed)
        self.canny_max_slider.sliderReleased.connect(self.value_end)
        self.canny_max_slider.setRange(0, 999)
        self.canny_max_slider.setValue(110)
        self.canny_max_slider.setSingleStep(1)
        self.canny_layout.addWidget(self.canny_min_slider)
        self.canny_layout.addWidget(self.canny_min_value)
        
        self.canny_layout.addWidget(self.canny_max_slider)
        self.canny_layout.addWidget(self.canny_max_value)
        self.canny_groupbox.setLayout(self.canny_layout)
        self.left_layout.addWidget(self.canny_groupbox)
        
        
        
        
        self.result_factor = QVBoxLayout()
        self.result_factor.setContentsMargins(0, 0, 0, 0)
        self.result_factor.setSpacing(0)
        self.result_groupbox = QGroupBox('결과값 이미지 비율')
        self.factor_value = QLabel('0.3')        
        self.factor_slider = QSlider(Qt.Horizontal, self)
        self.factor_slider.valueChanged.connect(self.factor_value_changed)
        self.factor_slider.sliderReleased.connect(self.value_end)
        self.factor_slider.setRange(10,100)
        self.factor_slider.setValue(30)
        self.factor_slider.setSingleStep(1)
        self.result_factor.addWidget(self.factor_slider)
        self.result_factor.addWidget(self.factor_value)
        self.result_groupbox.setLayout(self.result_factor)
        self.left_layout.addWidget(self.result_groupbox)
        
        
        self.check_button = QPushButton('이미지 확정', self)
        self.check_button.clicked.connect(self.image_render)
        
        self.left_layout.addWidget(self.check_button)
        
        main_layout.addLayout(self.left_layout,stretch=1)
        main_layout.addWidget(self.img_view,stretch=5)    
          
        
        main_wiget.setLayout(main_layout)
        return main_wiget
    
    def create_image_splite_wiget(self):
        image_splite_main_wiget = QWidget()
        self.image_splite_image_view = CView(self)
        self.image_splite_left_layout = QVBoxLayout()
        
        # main_layout.addLayout(self.left_layout,stretch=1)
        # main_layout.addWidget(self.img_view,stretch=5)   
        
        return image_splite_main_wiget
   
    def image_splite_wiget_render(self):
        self.setCentralWidget(self.image_splite_wiget)

        
        
    ## 손글씨 이미지 파일 load
    def open_img_file(self):
        img_list = ('jpg','png')
        filename,_ = QFileDialog.getOpenFileNames(self, 'Select Multi File', 'default')
        filename = [file for file in filename if file.lower().endswith(img_list)]
        if len(filename) > 0:
            self.filename = filename[0]
            self.refresh_image()


    ## 임계값 설정할때마다 바뀌는 함수
    ## 슬라이더 바 관련 https://spec.tistory.com/m/439
    def kernel_size_changed(self,value):
        self.kernel_size_value.setText(str(value))
    def factor_value_changed(self,value):
        self.factor_value.setText(str(value/100))     
    def canny_min_changed(self,value):
        self.canny_min_value.setText(str(value))
    def canny_max_changed(self,value):
        self.canny_max_value.setText(str(value))
    
    
    ## 임계값 설정 끝날때
    def value_end(self):
        if self.filename!="":
            self.refresh_image()
        
    ## 이미지 새로고침
    def refresh_image(self):
        
        img = cv2.imread(self.filename)
        widthImg =img.shape[1]
        heightImg = img.shape[0]
        
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # 프린터할 블링크 이미지 생성
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR 2 그레이
        imgBlur = cv2.GaussianBlur(imgGray, (int(self.kernel_size_value.text()),int(self.kernel_size_value.text())), 1) # 가우시안 블러 입힘

        imgThreshold = cv2.Canny(imgBlur,int(self.canny_min_value.text()),int(self.canny_max_value.text()))
        kernel = np.ones((int(self.kernel_size_value.text()), int(self.kernel_size_value.text())))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
        imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

        biggest, maxArea = self.biggestContour(contours) # FIND THE BIGGEST CONTOUR
        
        if biggest.size != 0:
            biggest=self.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
            imgBigContour = self.drawRectangle(imgBigContour,biggest,2)
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            #REMOVE 20 PIXELS FORM EACH SIDE
            imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))


            # Image Array for Display
            imageArray = ([img,imgThreshold],
                            [imgBigContour,imgWarpColored])

        else:
            imageArray = ([img,imgThreshold],
                            [imgBlank, imgBlank])

        # LABELS FOR DISPLAY
        lables = [["Original","Threshold"],
                    ["Biggest Contour","Warp Prespective"]]

        stackedImage = self.stackImages(imageArray,float(self.factor_value.text()),lables)       
        self.img_view.paint_img = QImage(stackedImage, stackedImage.shape[1], stackedImage.shape[0],stackedImage.strides[0],QImage.Format_RGB888) 
        self.img_view.scene = QGraphicsScene()
        self.img_view.item = QGraphicsPixmapItem(QPixmap.fromImage(self.img_view.paint_img))
        self.img_view.scene.addItem(self.img_view.item)
        self.img_view.setScene(self.img_view.scene)
        
        self.update()
        
    ## 결과 이미지 stack
    def stackImages(self,imgArray,scale,lables=[]):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
                hor_con[x] = np.concatenate(imgArray[x])
            ver = np.vstack(hor)
            ver_con = np.concatenate(hor)
        else:
            for x in range(0, rows):
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            hor_con= np.concatenate(imgArray)
            ver = hor
        if len(lables) != 0:
            eachImgWidth= int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)

            for d in range(0, rows):
                for c in range (0,cols):
                    cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                    cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
        return ver

    def reorder(self,myPoints):

        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        return myPointsNew

    def biggestContour(self,contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area

    def drawRectangle(self,img,biggest,thickness):
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

        return img



        
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
