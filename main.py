from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QApplication,QSlider, QMenu, QMenuBar, QAction, QFileDialog, QLabel, QInputDialog,QTreeView
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import math
import os

from infer_MX import run


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = ""
        
        self.result_dir_path = "./split_handwriting_image"
        self.gan_result_dir_path = "./g_handwriting_image"
        
        
        self.gan_models_dir_path = "./config_data/gan_models"
        self.vit_models_dir_path = "./config_data/vit_models"
        self.source_ttf_dir_path = "config_data/source_ttf_data"
        self.gan_text_path = "./config_data/generation_char.txt"
        
        
        self.crate_main_menubar()
        self.main_wiget = self.create_main_wiget()
        

        
        
        self.image_splite_wiget = self.create_image_splite_wiget()
        
        
        ## 위젯 스택에 쌓음
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.main_wiget)
        self.stackedWidget.addWidget(self.image_splite_wiget)
        
        ## 화면에 띄어줄 위젯 셋팅
        self.stackedWidget.setCurrentWidget(self.main_wiget)

        
        self.setCentralWidget(self.stackedWidget)
        
        
        
        
        self.paint_img = QImage()


        ## 삭제해도됨
        self.filename = "./GUI_pyqt5/20220726_122213.jpg"
        self.refresh_image()
        
        
        # self.setCentralWidget(self.main_wiget)
        self.setGeometry(0,100,1000,700)
        self.setWindowTitle("메인창")
        
        

    # 메뉴 생성
    def crate_main_menubar(self):
        self.mainMenu = self.menuBar()
        self.menubar = self.mainMenu.addMenu("메뉴")
        
        self.menubar_gan = self.mainMenu.addMenu("손글씨 생성")
        self.hr_gan_Action = QAction('손글씨 생성', self)
        self.hr_gan_Action.triggered.connect(self.hr_gan)
        self.menubar_gan.addAction(self.hr_gan_Action)

        self.openAction = QAction('jpg 이미지 불러오기', self)
        self.openAction.triggered.connect(self.open_img_file)
        self.menubar.addAction(self.openAction)
        
        self.closeAction = QAction('나가기', self)
        self.closeAction.triggered.connect(self.close)
        self.menubar.addAction(self.closeAction) 
        
    ## 손글씨 생성 시작
    def hr_gan(self):
        
        
        # 손글씨 생성에 GPU 사용
        cuda_flg = False
        
        
        
        ## 생성할 문자열 불러옴
        f = open(self.gan_text_path,'r')
        line = f.readline()
        line = line.replace(' ','')
        gan_chars = line
        
        
        ## 생성에 뼈대가될 base line ttf 파일 불러옴
        ## 첫번쨰는 무조건 source.ttf로시작 (구현체에서 그렇게 했음)
        source_ttf_file_list = [ os.path.join(self.source_ttf_dir_path,fname) for fname in os.listdir(self.source_ttf_dir_path) if fname.lower().endswith('.ttf')]
        
        
        ## gan model list 불러옴
        gan_model_list = [ os.path.join(self.gan_models_dir_path,fname) for fname in os.listdir(self.gan_models_dir_path) if fname.lower().endswith('.pth')]
        
        ## 손글씨 생성 참조할 손글씨 이미지 리스트
        ref_image_list = [ os.path.join(self.gan_models_dir_path,fname) for fname in os.listdir(self.vit_models_dir_path) if fname.lower().endswith('.png')]

        
        
        ## vit model list 불러옴(손글씨 검수 모델)
        vit_model_list = [ os.path.join(self.gan_models_dir_path,fname) for fname in os.listdir(self.vit_models_dir_path) if fname.lower().endswith('.pth')]


        
        
                
        
        
        print("a")
        pass
        
    # 작업창 메인 레이아웃 위젯 생성
    def create_main_wiget(self):
        main_wiget = QWidget()
        
        main_layout = QHBoxLayout()
        self.img_view = CView()

        self.left_layout = QVBoxLayout()
        
        
        self.tool_layout = QVBoxLayout()
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(0)
        self.tool_groupbox = QGroupBox('침식,팽창 커널 사이즈')
        self.kernel_size_value = QLabel('1')        
        self.kernel_size_slider = QSlider(Qt.Horizontal, self)
        self.kernel_size_slider.valueChanged.connect(self.kernel_size_changed)
        self.kernel_size_slider.sliderReleased.connect(self.value_end)
        self.kernel_size_slider.setRange(1, 50)
        self.kernel_size_slider.setValue(2)
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
        self.canny_min_slider.setRange(0, 255)
        self.canny_min_slider.setValue(50)
        self.canny_min_slider.setSingleStep(1)
        self.canny_max_value = QLabel('0')    
        self.canny_max_slider = QSlider(Qt.Horizontal, self)
        self.canny_max_slider.valueChanged.connect(self.canny_max_changed)
        self.canny_max_slider.sliderReleased.connect(self.value_end)
        self.canny_max_slider.setRange(0, 255)
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
        self.left_layout.addStretch(3)
        
        self.check_button = QPushButton('이미지 확정', self)
        self.check_button.clicked.connect(self.image_splite_wiget_render)
        
        self.left_layout.addWidget(self.check_button)
        
        main_layout.addLayout(self.left_layout,stretch=1)
        main_layout.addWidget(self.img_view,stretch=5)    
          
        
        main_wiget.setLayout(main_layout)
        return main_wiget
    
    def create_image_splite_wiget(self):
        
        self.input_text = ""
        
        image_splite_main_wiget = QWidget()
        self.image_splite_main_layout = QHBoxLayout()
          
        self.image_splite_image_view = CView()
        self.image_splite_left_layout = QVBoxLayout()
        
        self.rect_padding_layout = QVBoxLayout()
        self.rect_padding_layout.setContentsMargins(0, 0, 0, 0)
        self.rect_padding_layout.setSpacing(0)
        self.rect_padding_layout_groupbox = QGroupBox('rect 패딩')
        self.rect_value = QLabel('1')        
        self.rect_slider = QSlider(Qt.Horizontal, self)
        self.rect_slider.valueChanged.connect(self.rect_value_changed)
        self.rect_slider.sliderReleased.connect(self.image_splite_end)
        self.rect_slider.setRange(1,500)
        self.rect_slider.setValue(50)
        self.rect_slider.setSingleStep(1)
        self.rect_padding_layout.addWidget(self.rect_slider)
        self.rect_padding_layout.addWidget(self.rect_value)
        self.rect_padding_layout_groupbox.setLayout(self.rect_padding_layout)
        self.image_splite_left_layout.addWidget(self.rect_padding_layout_groupbox)
        
        
        
        self.hough_tr_layout = QVBoxLayout()
        self.hough_tr_layout.setContentsMargins(0, 0, 0, 0)
        self.hough_tr_layout.setSpacing(0)
        self.hough_tr_layout_groupbox = QGroupBox('허프변환 임계값')
        self.hough_tr_value = QLabel('1')        
        self.hough_tr_slider = QSlider(Qt.Horizontal, self)
        self.hough_tr_slider.valueChanged.connect(self.hough_tr_value_changed)
        self.hough_tr_slider.sliderReleased.connect(self.image_splite_end)
        self.hough_tr_slider.setRange(1,1500)
        self.hough_tr_slider.setValue(550)
        self.hough_tr_slider.setSingleStep(1)
        self.hough_tr_layout.addWidget(self.hough_tr_slider)
        self.hough_tr_layout.addWidget(self.hough_tr_value)
        self.hough_tr_layout_groupbox.setLayout(self.hough_tr_layout)
        self.image_splite_left_layout.addWidget(self.hough_tr_layout_groupbox)       
        
        
        
        

        self.canny_minmax_layout = QVBoxLayout()
        self.canny_minmax_layout.setContentsMargins(0, 0, 0, 0)
        self.canny_minmax_layout.setSpacing(0)
        self.canny_minmax_layout_groupbox = QGroupBox('canny edge min_max')
        self.image_split_canny_min = QLabel('1')        
        self.image_split_canny_min_slider = QSlider(Qt.Horizontal, self)
        self.image_split_canny_min_slider.valueChanged.connect(self.img_canny_min_changed)
        self.image_split_canny_min_slider.sliderReleased.connect(self.image_splite_end)
        self.image_split_canny_min_slider.setRange(1,500)
        self.image_split_canny_min_slider.setValue(100)
        self.image_split_canny_min_slider.setSingleStep(1)
        self.image_split_canny_max = QLabel('1')        
        self.image_split_canny_max_slider = QSlider(Qt.Horizontal, self)
        self.image_split_canny_max_slider.valueChanged.connect(self.img_canny_max_changed)
        self.image_split_canny_max_slider.sliderReleased.connect(self.image_splite_end)
        self.image_split_canny_max_slider.setRange(1,500)
        self.image_split_canny_max_slider.setValue(200)
        self.image_split_canny_max_slider.setSingleStep(1)
        self.canny_minmax_layout.addWidget(self.image_split_canny_min_slider)
        self.canny_minmax_layout.addWidget(self.image_split_canny_min)
        self.canny_minmax_layout.addWidget(self.image_split_canny_max_slider)
        self.canny_minmax_layout.addWidget(self.image_split_canny_max)
        self.canny_minmax_layout_groupbox.setLayout(self.canny_minmax_layout)
        self.image_splite_left_layout.addWidget(self.canny_minmax_layout_groupbox)
        
        
        
        self.thresh_hold_layout = QVBoxLayout()
        self.thresh_hold_layout.setContentsMargins(0, 0, 0, 0)
        self.thresh_hold_layout.setSpacing(0)
        self.thresh_hold_layout_groupbox = QGroupBox('thresh hold')
        self.thresh_hold_var = QLabel('1')        
        self.thresh_hold_slider = QSlider(Qt.Horizontal, self)
        self.thresh_hold_slider.valueChanged.connect(self.thresh_hold_var_change)
        self.thresh_hold_slider.sliderReleased.connect(self.image_splite_end)
        self.thresh_hold_slider.setRange(1,255)
        self.thresh_hold_slider.setValue(127)
        self.thresh_hold_slider.setSingleStep(1)
        
        
        
        self.thresh_hold_slider.setDisabled(True)
        self.thresh_hold_otsu_check_box = QCheckBox('Otsu 알고리즘 적용', self)        
        self.thresh_hold_otsu_check_box.toggle()
        self.thresh_hold_otsu_check_box.stateChanged.connect(self.click_otus_check)
        
                
        

        self.thresh_hold_layout.addWidget(self.thresh_hold_slider)
        self.thresh_hold_layout.addWidget(self.thresh_hold_var)
        self.thresh_hold_layout.addWidget(self.thresh_hold_otsu_check_box)
        self.thresh_hold_layout_groupbox.setLayout(self.thresh_hold_layout)
        self.image_splite_left_layout.addWidget(self.thresh_hold_layout_groupbox)
        
        
        
                
        self.image_split_factor_layout = QVBoxLayout()
        self.image_split_factor_layout.setContentsMargins(0, 0, 0, 0)
        self.image_split_factor_layout.setSpacing(0)
        self.image_split_factor_layout_groupbox = QGroupBox('결과 이미지 비율')
        self.image_split_factor_value = QLabel('0.3')        
        self.image_split_factor_slider = QSlider(Qt.Horizontal, self)
        self.image_split_factor_slider.valueChanged.connect(self.image_split_factor_value_change)
        self.image_split_factor_slider.sliderReleased.connect(self.image_splite_end)
        self.image_split_factor_slider.setRange(10,100)
        self.image_split_factor_slider.setValue(30)
        self.image_split_factor_slider.setSingleStep(1)
        self.image_split_factor_layout.addWidget(self.image_split_factor_slider)
        self.image_split_factor_layout.addWidget(self.image_split_factor_value)
        self.image_split_factor_layout_groupbox.setLayout(self.image_split_factor_layout)
        self.image_splite_left_layout.addWidget(self.image_split_factor_layout_groupbox)
        
        
        
        
                
                
        self.input_text_btn = QPushButton('문자열 입력', self)
        self.input_text_btn.clicked.connect(self.showDialog)
        self.image_splite_left_layout.addWidget(self.input_text_btn)

        
        self.input_text_btn = QPushButton('이전페이지', self)
        self.input_text_btn.clicked.connect(self.ROI_detector_render)
        self.image_splite_left_layout.addWidget(self.input_text_btn)
        
        
        self.input_text_btn = QPushButton('이미지 자르기', self)
        self.input_text_btn.clicked.connect(self.split_handwriting_img)
        self.image_splite_left_layout.addWidget(self.input_text_btn)
        self.image_splite_left_layout.addStretch(3)
        
        self.image_splite_main_layout.addLayout(self.image_splite_left_layout,stretch=1)
        self.image_splite_main_layout.addWidget(self.image_splite_image_view,stretch=5)   
        
          
        image_splite_main_wiget.setLayout(self.image_splite_main_layout)
        
        return image_splite_main_wiget
   
   ## 이미지 스플릿 화면 전환
    def image_splite_wiget_render(self):
        if self.filename != "":
            self.stackedWidget.setCurrentWidget(self.image_splite_wiget)
            self.image_splite_end()
    ## ROI 화면 전환
    def ROI_detector_render(self):
        if self.filename != "":
            self.stackedWidget.setCurrentWidget(self.main_wiget)
            self.refresh_image()
    
    ## ROI를 기준으로 입력문자열과 비교하여 이미지 자름
    def split_handwriting_img(self):
        if  self.thresh_hold_img is not None and self.labelposition is not None:
            if not os.path.exists(self.result_dir_path ):
                os.makedirs(self.result_dir_path)
                
            for i,rect in enumerate(self.labelposition):
                x,y,w,h,_=rect
                try:
                    cv2.imwrite(os.path.join(self.result_dir_path,self.input_text[i]+".png"),self.thresh_hold_img[y:y+h,x:x+w])
                except:
                    cv2.imwrite(os.path.join(self.result_dir_path,str(i)+".png"),self.thresh_hold_img[y:y+h,x:x+w])

            
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
        if value==1:
            self.kernel_size_value.setText(str(value))
        else:
            self.kernel_size_value.setText(str((value*2)-1))
    def factor_value_changed(self,value):
        self.factor_value.setText(str(value/100))     
    def canny_min_changed(self,value):
        self.canny_min_value.setText(str(value))
    def canny_max_changed(self,value):
        self.canny_max_value.setText(str(value))
    def rect_value_changed(self,value):
        self.rect_value.setText(str(value))
    def img_canny_max_changed(self,value):
        self.image_split_canny_max.setText(str(value))
    def img_canny_min_changed(self,value):
        self.image_split_canny_min.setText(str(value))
    def thresh_hold_var_change(self,value):
        self.thresh_hold_var.setText(str(value))
    def hough_tr_value_changed(self,value):
        self.hough_tr_value.setText(str(value))
    def image_split_factor_value_change(self,value):
        self.image_split_factor_value.setText(str(value/100))

    
    ## 오츠 알고리즘 적용
    def click_otus_check(self,state):
        if state==Qt.Checked:
            self.thresh_hold_slider.setDisabled(True)
        else:
            self.thresh_hold_slider.setDisabled(False)
        self.refresh_image_split()
    
    ## 임계값 설정 끝날때 실행될 함수
    def value_end(self):
        print(self.filename)
        if self.filename!="":
            self.refresh_image()
    def image_splite_end(self):
        if self.filename!="":
            self.refresh_image_split()
        
    ## 자를 기준의 문자열 받아오기
    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', '자를 문자열을 입력하세요')
        if ok==True:
            self.input_text = text
            self.image_splite_end()
            
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
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 40) # DRAW THE BIGGEST CONTOUR
            imgBigContour = self.drawRectangle(imgBigContour,biggest,5)
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

        self.imgWarpColored_img = imgWarpColored
        stackedImage = self.stackImages(imageArray,float(self.factor_value.text()),lables)       
        self.img_view.paint_img = QImage(stackedImage, stackedImage.shape[1], stackedImage.shape[0],stackedImage.strides[0],QImage.Format_RGB888) 
        self.img_view.scene = QGraphicsScene()
        self.img_view.item = QGraphicsPixmapItem(QPixmap.fromImage(self.img_view.paint_img))
        self.img_view.scene.addItem(self.img_view.item)
        self.img_view.setScene(self.img_view.scene)
        
        self.update()
        
    def refresh_image_split(self):
        img = self.imgWarpColored_img.copy()
        
 
        #허프변환 인자값 높을수록 직선을 덜찾음
        # -> 550개의 점들이 뭉쳐져있어야만 직선으로 봄
        # HoughLines_t = 550
        HoughLines_t = int(self.hough_tr_value.text())
        
        
        ## 캐니앳지 변수값
        c_min = int(self.image_split_canny_min.text())
        c_max = int(self.image_split_canny_max.text())
        
        # 자를 선 넓이
        rect_padding = int(self.rect_value.text())
        
        ## 이진화 스래쉬홀드값
        thresh_hold_var = int(self.thresh_hold_var.text())
        
        ## 가까이 붙어 있는 픽셀들 비율 -> 숫자가 높을수록 더 멀리 보겠다는 뜻, 수직,수평선 중복제거에 쓰이는 값
        merge_rate = 0.005
        
        
        delete_rate = 0.95
        
        
        ## 빈칸 구하는 비율 -> 
        # 면적의 var 비율만큼, 숫자가 커지면 글씨데이터도 캡쳐됨, -> 숫자가 작아야함
        white_space_threshold_var = 0.006

        ## 사각형 노이즈때문에 w,h에 패딩을 곱 사이즈 비율
        padding_fator = 0.05
        
        ## 결과이미지 프린터 비율
        factot = float(self.image_split_factor_value.text())
        
        ## 군집화에서 탐색할 거리 (distanse)
        Epsilon = 5
        
        w = img.shape[1]
        h = img.shape[0]
        line_img = np.zeros((img.shape),np.uint8)    
        line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(img, c_min, c_max )
        lines = cv2.HoughLines(edges, 1,math.pi/2, HoughLines_t)
        
        vertical_line = []
        horizon_line = []        
        position_list = []

        # 검출된 모든 선 순회하며 수직, 수평 선 찾음
        for line in lines:
            r,theta = line[0] # 거리와 각도
            tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
            x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표

            # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
            x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
            x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)

            ## abs 값이 0이면 수직,수평으로 판단
            if abs(x1+x2) < 10:
                horizon_line.append([x1, y1, x2, y2])
            if abs(y1+y2) < 10:
                vertical_line.append([x1, y1, x2, y2])

        w_th = int(merge_rate * w)
        vertical_line = list(sorted(vertical_line,key=lambda x:x[0]))

        h_th = int(merge_rate * h)
        horizon_line = list(sorted(horizon_line,key=lambda x:x[1]))
        
        ## 1차 중복 제거
        vertical_line = self.Deduplication_v(vertical_line,w_th)
        horizon_line = self.Deduplication_h(horizon_line,h_th)


        ## 없는 수평선 유추
        horizon_line = self.horizon_line_inference(horizon_line,h_th,w,h)
        ## 유추 후 , 중복제거
        horizon_line = self.Deduplication_h(horizon_line,h_th)


        ## 없는 수평선 유추
        vertical_line = self.vertical_line_inference(vertical_line,w_th,w,h)
        ## 유추 후 , 중복제거
        vertical_line = self.Deduplication_v(vertical_line,w_th)
        
        ## 스플릿할 기준이미지 생성
        for x1,y1,x2,y2 in vertical_line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255), rect_padding)
        for x1,y1,x2,y2 in horizon_line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255), rect_padding)
        cv2.rectangle(line_img, (0, 0), (w,h), (255), rect_padding)


        ## 레이블링 작업 진행, 좌표값 획득
        contours, hierarchy = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in reversed(contours):
            x, y, w, h = cv2.boundingRect(c)
            if line_img.shape[0]*delete_rate > h and line_img.shape[1]*delete_rate:
                position_list.append([x,y,w,h])
                
        ## 좌표값 y기준으로 정렬
        position_list = sorted(position_list,key=lambda x:x[1])
        
        
        ## DBSCAN 군집화 알고리즘
        x = np.array([ [var,-1,False,i ] for i,(_,var,_,_) in enumerate(position_list)])
        # init list , 큐 역활
        calculate_list = []        
        ## 첫번째 값부터 라벨 시작, 검색 여부
        label_pointer = 0
        x[0][2] = True
        ## 라벨 포인터
        x[0][1] = label_pointer
        calculate_list.append(x[0])

        ## 2차원 이상일 경우 np로 계산
        def distance(a,b):
            return abs(a-b)
        
        ## DBSCAN 알고리즘 시작
        while len(calculate_list) > 0:
            serch_var = calculate_list.pop(0) 
            ## 거리계산값, 검색 여부의 index만 뽑아옴
            candidate_index_list = [ i for var,label,flg,i in x if flg==False and distance(serch_var[0],var) < Epsilon ]
            
            for index in candidate_index_list:
                ## 정보 바꾸어주고
                x[index][2] = True
                x[index][1] = label_pointer
                
                ## 검색할 list에 더함
                calculate_list.append(x[index])
            
            
            ## 다 검사하지 않았다면 검색 안된 x배열의 정보를 계산(큐) 배열에 추가
            if len(calculate_list)==0 and np.all(x[:,2]==1)==False:
                label_pointer +=1
                _,index_arr = np.where([x[:,2]==0])
                
                x[index_arr[0]][2] = True
                x[index_arr[0]][1] = label_pointer
                        
                calculate_list.append(x[index_arr[0]])

            # 라벨 정보를 좌표값에 concat 시킴
            label_position = np.concatenate((np.array(position_list),x[:,1].reshape(-1,1)),axis=1)
        
        
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        if self.thresh_hold_otsu_check_box.checkState() == 2:
            _,dst = cv2.threshold(img_gray,thresh_hold_var,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            _,thresh_hold_img = cv2.threshold(img_gray,thresh_hold_var,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _,dst = cv2.threshold(img_gray,thresh_hold_var,255,cv2.THRESH_BINARY_INV )
            _,thresh_hold_img = cv2.threshold(img_gray,thresh_hold_var,255,cv2.THRESH_BINARY)
            
        
        delete_index_list = []
        
        ## 빈칸인지 의심하는 함수
        for i,rect in enumerate(label_position):
            x,y,w,h,_ = rect
            ## 이미지를 자름
            crop_img = dst[y+int(padding_fator*h):y+h-int(padding_fator*h),x+int(padding_fator*w):x+w-int(padding_fator*w)]
            
            ## 빈칸인 격자 주소값 저장
            if len(crop_img[np.nonzero(crop_img)]) < w*h*white_space_threshold_var:
                delete_index_list.append(i)

        ## 빈칸의심 인덱스 삭제
        label_position = np.delete(label_position,delete_index_list,axis=0)
        ## 라벨 기준, x값으로 정렬
        label_position = np.array(sorted(label_position,key=lambda x:(x[4],x[0])))
        

        rect_img = thresh_hold_img.copy()
        rect_img = cv2.cvtColor(rect_img,cv2.COLOR_GRAY2BGR)
        
        _,hh,_= rect_img.shape
        
        source_ttf_file_list = [ os.path.join(self.source_ttf_dir_path,fname) for fname in os.listdir(self.source_ttf_dir_path) if fname.lower().endswith('.ttf')]
        font = ImageFont.truetype(source_ttf_file_list[0], size=int(hh*0.04))
        
        for i,rect in enumerate(label_position):
            x,y,w,h,_=rect
            
            
            pil_img=Image.fromarray(rect_img)
            draw = ImageDraw.Draw(pil_img)
            
            try:
                draw.text((x, y), self.input_text[i], fill='red',font=font)
            except:
                draw.text((x, y), str(i), fill='red',font=font)
                
            rect_img = np.array(pil_img)
            
            
            
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
        
        ## 정보 저장
        self.thresh_hold_img = thresh_hold_img
        self.labelposition = label_position
        

        line_img = cv2.cvtColor(line_img,cv2.COLOR_GRAY2BGR)

        result_img = np.hstack([cv2.resize(rect_img,(0,0),None,factot,factot),cv2.resize(line_img,(0,0),None,factot,factot)])
        
        
        self.image_splite_image_view.paint_img = QImage(result_img, result_img.shape[1], result_img.shape[0],result_img.strides[0],QImage.Format_RGB888) 
        self.image_splite_image_view.scene = QGraphicsScene()
        self.image_splite_image_view.item = QGraphicsPixmapItem(QPixmap.fromImage(self.image_splite_image_view.paint_img))
        self.image_splite_image_view.scene.addItem(self.image_splite_image_view.item)
        self.image_splite_image_view.setScene(self.image_splite_image_view.scene)
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
    
    ## 첫화면 이미지 계산용 함수들
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

    
    ## 세로 허프변환 비교하며 제거하는 함수
    def remove_element_v(self,list_var,w_th):
        for i in range(0,len(list_var)-1):
            for j in range(i+1,len(list_var)):
                if abs(list_var[i][0] - list_var[j][0]) <= w_th:
                    list_var[i][0] += int(abs(list_var[i][0] - list_var[j][0]) / 2)
                    list_var[i][2] += int(abs(list_var[i][2] - list_var[j][2]) / 2)
                    del list_var[j]
                    return True,list_var
        return False,list_var
    ## 세로 허프변환 직선 중복된값들 제거
    def Deduplication_v(self,list_var,w_th):
        flg = True
        while flg:
            flg,list_var=self.remove_element_v(list_var,w_th)
        return list_var
    ## 가로 허프변환 비교하며 제거하는 함수
    def remove_element_h(self,list_var,h_th):
        for i in range(0,len(list_var)-1):
            for j in range(i+1,len(list_var)):
                if abs(list_var[i][1] - list_var[j][1]) <= h_th:
                    list_var[i][1] += int(abs(list_var[i][1] - list_var[j][1]) / 2)
                    list_var[i][3] += int(abs(list_var[i][3] - list_var[j][3]) / 2)
                    del list_var[j]
                    return True,list_var
        return False,list_var
    ## 가로 허프변환 직선 중복된값들 제거
    def Deduplication_h(self,list_var,h_th):
        flg = True
        while flg:
            flg,list_var=self.remove_element_h(list_var,h_th)
        return list_var

    ## 리스트에서 var와 가장 근사한값 찾기(인덱스)
    def min_find_index(self,var_list,var):
        min = 999999999999
        for i,item in enumerate(var_list):
            if abs(item-var) < min:
                min = item
                index = i    
        return index

            
    ## 수평선 없는거 유추
    def horizon_line_inference(self,horizon_line,h_th,w,h):
        horizon_array = np.array(horizon_line)
        h_y1_array = np.array(horizon_line)[:,1]
        # 1차 미분결과에서 중앙값 뽑기 (너비 간격)
        median_h = int(np.median(np.diff(h_y1_array)))
        ## 근사값 인덱스 찾아옴
        h_index = self.min_find_index(h_y1_array,median_h)
        ## 연산 기준값
        reference_v = h_y1_array[h_index]    
        # 1. 기준점에서 -방향으로(위로) 생성
        while True:
            reference_v -=median_h
            if reference_v <= 0:
                break        
            horizon_line.append([horizon_array[h_index][0],reference_v,horizon_array[h_index][2],reference_v])
        
        reference_v = h_y1_array[h_index]
        # 2. 기준점에서 +방향으로(밑으로) 생성
        while True:
            reference_v += median_h
            if reference_v > h:
                break        
            horizon_line.append([horizon_array[h_index][0],reference_v,horizon_array[h_index][2],reference_v])
        # list(sorted(horizon_line,key=lambda x:x[1]))
        return list(sorted(horizon_line,key=lambda x:x[1]))


    ## 수직선 없는거 유추
    def vertical_line_inference(self,vertical_line,w_th,w,h):
        vertical_array = np.array(vertical_line)
        v_x1_array = np.array(vertical_line)[:,0]
        
        # 1차 미분결과에서 중앙값 뽑기 (너비 간격)
        median_h = int(np.median(np.diff(v_x1_array)))
        
        ## 근사값 인덱스 찾아옴
        v_index = self.min_find_index(v_x1_array,median_h)
        
        ## 연산 기준값
        reference_v = v_x1_array[v_index]    
        
        # 1. 기준점에서 -방향으로(위로) 생성
        while True:
            reference_v -=median_h
            if reference_v <= 0:
                break        
            vertical_line.append([reference_v,vertical_array[v_index][1],reference_v,vertical_array[v_index][3]])
        
        reference_v = v_x1_array[v_index]
        # 2. 기준점에서 +방향으로(밑으로) 생성
        while True:
            reference_v += median_h
            if reference_v > w:
                break        
            vertical_line.append([reference_v,vertical_array[v_index][1],reference_v,vertical_array[v_index][3]])

        return list(sorted(vertical_line,key=lambda x:x[1]))

        
class CView(QGraphicsView):
    def __init__(self):
        super().__init__()       
        self.setAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.setStyleSheet("background:transparent;")       
        self.paint_img = QImage()
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
