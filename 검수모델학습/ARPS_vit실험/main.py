import os
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf,open_dict
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
import shutil
import pathlib
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import mlflow
from utils import AverageMeter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.special import softmax
from matplotlib import pyplot as plt

from models import ResNet50
import gan
from ARPLoss import ARPLoss
from d_set import Hanguldataset


def cleanup():
    dist.destroy_process_group()
def setup(rank, world_size,backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def trainer(rank, gpus, args):
    ## 멀티 프로세싱 셋업
    setup(rank, gpus,args.env_.backend)    
    ## 시드 41로 고정
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    seed_everything(41)
    
    
    train_transform = transforms.Compose([
                            transforms.Resize((args.env_.image_size,args.env_.image_size)),
                            transforms.ToTensor()
                            ])
    
    train_dataset = Hanguldataset(g_target_char_txt=args.env_.target_char_txt,ttf_dir=args.env_.train_ttf_dir,transform=train_transform)

    trainloader = DataLoader(train_dataset, batch_size = args.env_.batch_size, num_workers = args.env_.num_worker,pin_memory=True)

    # 채널, 클래스 갯수
    # net = classifier32ABN(1,num_classes=args.env_.num_classes).to(rank)
    net = ResNet50(1,args.env_.num_classes,args.env_.feat_dim).to(rank)
    
    
    ## 생성 모델 함수, 노이즈 디멘젼, 크기, 생성할 채널
    netG = gan.Generator32(args.parameters_.nz, 64, 1).to(rank)
    
    
    ## 채널, 최종 크기인듯
    netD = gan.Discriminator32(1, 64).to(rank)
    
    
    # fixed_noise = torch.FloatTensor(64, args.parameters_.nz, 1, 1).normal_(0, 1).to(rank)
    criterionD = nn.BCELoss()
    

    
    # weight_pl,temp,num_classes,feat_dim
    criterion = ARPLoss(0.1,1.0,args.env_.num_classes,args.env_.feat_dim).to(rank)

    params_list = [{'params': net.parameters()},{'params': criterion.parameters()}]
    
    optimizer = torch.optim.Adam(params_list, lr=args.parameters_.lr)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.parameters_.gan_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.parameters_.gan_lr, betas=(0.5, 0.999))


    # 에폭
    for epoch in range(0,args.env_.epochs):
        train_cs(net, netD, netG, criterion, criterionD,optimizer, optimizerD, optimizerG,trainloader,rank,epoch,args)
        train(net, criterion, optimizer, trainloader,rank,epoch)
        embeding_visualization(net,netG, criterion,train_dataset,trainloader, epoch, rank, args)
        if rank == 0:
            # torch.save(net.module.state_dict(), os.path.join('./last.pth'))
            torch.save(net.state_dict(), os.path.join('./last.pth'))
 
 
        

    cleanup()
    
def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader,device,epoch,args):
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()
    net.train()
    netD.train()
    netG.train()
    criterion.train()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)

        data = data.to(device=device)
        labels = labels.to(device=device)
        gan_target = gan_target.to(device=device)
        
        
        ## 노이즈 생성
        noise = torch.FloatTensor(data.size(0), args.parameters_.nz, args.parameters_.ns, args.parameters_.ns).normal_(0, 1).to(device=device) 
        fake = netG(noise)
        
                
        ###########################
        # (1) Update D network    #
        ###########################
        ## 판별자 네트워크에 진짜 실데이터 넣어서 로스값 구함
        # train with real
        gan_target.fill_(real_label)
        ## 진짜 리얼 데이터 targetv = 1로 가득채워짐
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()
        
        
        # train with fake
        ## 이후 판별자 네트워크에 가짜 데이터 넣어서 업데이트
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()



        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        ## 판별자에 페이크 데이터를 넣고 나온 결과값을 
        targetv = Variable(gan_target.fill_(real_label))
        # targetv = 1(리얼로 생각하라는)로 채워진 라벨링
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        # 실제 분포를 최소화
        x, y = net(device,fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + args.parameters_.beta * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(device,data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
        ## y torch.Size([64, 6])
        # 여기서 에러남
        _, loss = criterion(x, y,device, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), args.parameters_.nz, args.parameters_.ns, args.parameters_.ns).normal_(0, 1).to(device=device)
        fake = netG(noise)
        
        x, y = net(device,fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + args.parameters_.beta * F_loss_fake
        total_loss.backward()
        optimizer.step()
        
        losses.update(total_loss.item(), labels.size(0))
        
        print(f"epoch [{epoch}] ARPS학습 {batch_idx+1}/{len(trainloader)}\t Net {losses.val:.3f} ({losses.avg:.3f}) G {lossesG.val:.3f} ({lossesG.avg:.3f}) D {lossesD.val:.3f} ({lossesD.avg:.3f})")
    
    
    if device==0:
        noise = torch.FloatTensor(data.size(0), args.parameters_.nz, args.parameters_.ns, args.parameters_.ns).normal_(0, 1).to(device=device) 
        fake = netG(noise)
        save_image(fake,os.path.join(args.env_.data_save_dir,str(epoch)+"_gen_image_sample.jpg"))
        
        
    
def train(net, criterion, optimizer, trainloader,device,epoch):
    net.train()
    losses = AverageMeter()
    
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        data = data.to(device=device)
        labels = labels.to(device=device)
        
        optimizer.zero_grad()
        
        
        x, y = net(device,data, True) ## 마지막단 피쳐 값 줄거냐 말거냐 True
        logits, loss = criterion(x, y,device,labels)
        # criterion.points.shape는 torch.Size([2350, 2048])
        
        loss.backward()
        optimizer.step()
    
        losses.update(loss.item(), labels.size(0))

        print(f"epoch [{epoch}] 트레인 {batch_idx+1}/{len(trainloader)}\t Loss {losses.val:.6f} ({losses.avg:.6f})")
        ## 엑큐러시 넣자
        loss_all += losses.avg

    return loss_all


def embeding_visualization(net,netG, criterion,train_dataset, trainloader, epoch, device, args):
    net.eval()
    netG.eval()
    criterion.eval()
    


    with torch.no_grad():
        for data,label in trainloader:
            data = data.to(device=device)
            label = label.to(device=device)
            x, y = net(device,data, True)
            logits, _ = criterion(x, y,device)
            
            p_real = logits.cpu().numpy()
            break
            
        noise = torch.FloatTensor(data.size(0), args.parameters_.nz, args.parameters_.ns, args.parameters_.ns).normal_(0, 1).to(device=device) 
        fake = netG(noise)
        x, y = net(device,fake, True)
        logits, _ = criterion(x, y,device)
        p_fake = logits.cpu().numpy()
        
    
    print(f"{epoch} 에폭 - real 데이터 logit 값 max {np.max(np.round(softmax(p_real,1),3)[0]):.4f}")
    print(f"{epoch} 에폭 - fake 데이터 logit 값 max {np.max(np.round(softmax(p_fake,1),3)[0]):.4f}")
    
            
    # plt.figure(figsize=(10, 10))
    # pca = PCA(n_components=2)
    
    # #np.array(deep_features) shape =  (4700, 2048)
    
            
    # ## PCA 학습
    # center_point = np.array(pca.fit_transform(np.array(criterion.points.detach().cpu().numpy())))      
        
    # ## 실데이터 시각화
    # cluster = np.array(pca.transform(np.array(deep_features)))
    # actual = np.array(actual)
    # label_list = train_dataset.char_list
    # for i, label in enumerate(label_list):
    #     idx = np.where(actual == i)
    #     plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', color='black')
    
    
    # ## 센터 좌표 시각화
    # for i, label in enumerate(center_point):
    #     plt.scatter(center_point[i, 0], center_point[i, 1], marker='s', label=label)
    # # 설명 가능
    


    
    # # PCA 주성분 설명력 출력
    # # print(pca.explained_variance_ratio_)
    # plt.savefig(os.path.join(args.env_.data_save_dir,str(epoch)+'_'+str(np.sum(pca.explained_variance_ratio_))[:5]+'_feature.png'), dpi=300)
        
            


@hydra.main(version_base=None, config_path='config',config_name="c_config")
def main(config: DictConfig):
    ## 하이드라용 job 넘버 0으로 설정
    hydra_job_num = 0
    ## hydra config 수정
    config.env_.mlflow_run_tag
    if "num" in HydraConfig.get().job:
        hydra_job_num = HydraConfig().get().job.num
    ## mlflow run 네임 설정
    mlflow_run_name = config.env_.mlflow_run_tag+"_"+str(hydra_job_num) + "_run"

    
    ## mlflow 클라이언트 생성
    mlflow_client = MlflowClient()
    ## 실험, 런 생성
    mlflow_run = mlflow_client.create_run(experiment_id=str(config.env_.experiments_id),run_name=mlflow_run_name)
    OmegaConf.set_struct(config, True)

    ## config dict에 mlflow run id 정보 추가
    with open_dict(config):
        config.env_.mlflow_run_id = mlflow_run.info.run_id

                
    ## 결과 임시 저장 폴더 생성
    pathlib.Path(config.env_.data_save_dir).mkdir(parents=True, exist_ok=True) 
    ## 멀티 프로세싱 시작
    
    mp.spawn(trainer, args=(config.env_.gpus, config), nprocs=config.env_.gpus, join=True)
    # trainer(0, config.env_.gpus, config)
    
    ## 멀티 프로세싱 작업이 끝났으므로, 결과 임시저장 폴더 삭제
    # if os.path.exists(config.env_.data_save_dir):
    #     shutil.rmtree(config.env_.data_save_dir)
    

    
if __name__ == '__main__':
    main()