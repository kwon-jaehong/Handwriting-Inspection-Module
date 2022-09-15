
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import cv2
import os

class Hanguldataset(Dataset):
    def __init__(self,g_target_char_txt,ttf_dir,transform):

        self.ttf_dir = ttf_dir
        self.ttf_list = os.listdir(ttf_dir)

        f = open(g_target_char_txt,'r')
        self.char_list = list(f.readline())
        self.char_len = len(self.char_list)
        
        self.transform = transform
    
    def __len__(self):
        return self.char_len * len(self.ttf_list)
    
    def __getitem__(self,index):
        
        ## ttf 파일 인덱스
        font_index = index // self.char_len
        
        ## 문자 인덱스
        char_index = index % self.char_len
        
        img = self.ttf_to_img_render(font_index,char_index)
        img = np.array(img)
        # ret, img = cv2.threshold(img,240,255, cv2.THRESH_BINARY)
        img= cv2.bitwise_not(img)
        
        
        ## 글자 크기에 맞추어 크롭
        nz = cv2.findNonZero(img)
        left = nz[:,0,0].min()
        right = nz[:,0,0].max()
        bottom = nz[:,0,1].min()
        top = nz[:,0,1].max()
        img = img[bottom:top,left:right]
        ## 크롭끝
        

        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img,char_index
    
    ## ttf파일 기반으로 이미지 생성 사이즈는 128,128
    def ttf_to_img_render(self,font_index, char_index, size=(128, 128), pad=20):
        char = self.char_list[char_index]
        font =  ImageFont.truetype(self.ttf_dir+"/"+self.ttf_list[font_index],size=150)
        
        width, height = font.getsize(char)
        max_size = max(width, height)

        if width < height:
            start_w = (height - width) // 2 + pad
            start_h = pad
        else:
            start_w = pad
            start_h = (width - height) // 2 + pad

        img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
        draw = ImageDraw.Draw(img)
        draw.text((start_w, start_h), char, font=font)
        img = img.resize(size, 2)
        return img
    
# def main():
#     img_size = 32

#     train_transform = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.RandomRotation(25), transforms.ToTensor()])

#     dataset = Hanguldataset(g_target_char_txt="../data/target.txt",ttf_dir="../data/ttf_file",transform=train_transform)
#     # print(dataset.__len__())
#     print(dataset.__getitem__(7049))
# if __name__ == '__main__':
#     main()
