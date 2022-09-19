
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from infer_MX.MX.models import Generator
from pathlib import Path
import os



TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def normalize(tensor, eps=1e-5):
    """ Normalize tensor to [0, 1] """
    # eps=1e-5 is same as make_grid in torchvision.
    minv, maxv = tensor.min(), tensor.max()
    tensor = (tensor - minv) / (maxv - minv + eps)

    return tensor

def save_tensor_to_image(tensor, filepath, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    tensor = normalize(tensor)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)

def render(font, char, size=(128, 128), pad=20):
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

########################################################

## 손글씨 생성에 뼈대가 될 ttf 파일
source_path = "./infer_MX/source.ttf"

## 손글씨 생성 이미지 저장 폴더
# save_dir = "./result_426000"
save_dir = "./gen_result_last"

## 모델 저장파일
# weight_path = "./infer_MX/426000.pth"
weight_path = "./config_data/gan_models/0-last.pth"

## 모델 config 관찰자 수
n_experts = 12

## 생성 모델 배치사이즈
batch_size = 16
########################################################


## 생성모델 가중치 불러옴
device = torch.device('cpu')
gen = Generator(n_experts=n_experts, n_emb=2).eval()
weight = torch.load(weight_path,map_location=device)
if "generator_ema" in weight:
    weight = weight["generator_ema"]
gen.load_state_dict(weight)



## 생성할 글자 정보 파싱 (완성형 한글 2350자)
f = open('./infer_MX/generation_char.txt','r')
line = f.readline()
line = line.replace(' ','')
gen_chars = line
## 생성할 글자 정보 파싱 끝


gen_chars="안녕하세요동해물과백두산이"



########################################################
ref_path = "./infer_MX/example_png_data/"
ref_path = "./infer_MX/tempdata2/"


file_list = os.listdir(ref_path)
ref_chars = [ fname.lower().replace('.png','') for fname in file_list if fname.lower().endswith('.png')]
ref_chars = "".join(ref_chars)


## 이미지 불러오고 쌓음
ref_imgs_list = []
for img_path in [ fname for fname in file_list if fname.lower().endswith('.png')]:
    ref_img = Image.open(os.path.join(ref_path,img_path))
    ref_imgs_list.append(ref_img)

ref_imgs = torch.stack([TRANSFORM(img) for img in ref_imgs_list])






# ########################################################

def infer_MX(gen, save_dir, source_path, gen_chars,ref_imgs,batch_size=32):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ## 뼈대 글자(고딕) 의 ttf파일 불러옴
    source = ImageFont.truetype(str(source_path), size=150)
        
    ## 이미지를 배치사이즈로 나눔
    ref_batches = torch.split(ref_imgs, batch_size)

    style_facts = {}
    
    ## 참조 문자의 스타일 특징값 계산,불러옴 
    for batch in ref_batches:
        style_fact = gen.factorize(gen.encode(batch), 0)
        for k in style_fact:
            style_facts.setdefault(k, []).append(style_fact[k])

    #(스타일 특징 평균값냄)
    style_facts = {k: torch.cat(v).mean(0, keepdim=True) for k, v in style_facts.items()}
    # style_facts = {'last': tensor([[[[[ 2.7287e...ackward1>), 'skip': tensor([[[[[ 7.6727e...ackward1>)}
    
    
    ## 손글씨 이미지 생성
    for char in gen_chars:        
        ## 소스 이미지 불러옴 (뼈대 & 글자 프린트 이미지)
        source_img = TRANSFORM(render(source, char)).unsqueeze(0)
        
        char_facts = gen.factorize(gen.encode(source_img), 1)
        gen_feats = gen.defactorize(style_facts, char_facts)
        out = gen.decode(gen_feats)[0].detach().cpu()
        path = save_dir / f"{char}.bmp"
        
        ## 저장하고 끝
        save_tensor_to_image(out, path)


infer_MX(gen, save_dir, source_path,  gen_chars, ref_imgs, batch_size)

a = 100


