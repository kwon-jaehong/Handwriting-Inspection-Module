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
import mlflow
from utils import AverageMeter

from models import classifier32ABN
import gan
from ARPLoss import ARPLoss


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
    # setup(rank, gpus,args.env_.backend)    
    ## 시드 41로 고정
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    seed_everything(41)
    
    
    train_transform = transforms.Compose([
                            transforms.Resize((args.env_.image_size,args.env_.image_size)),
                            transforms.ToTensor()
                            ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=train_transform,
                                               download=True)
    trainloader = DataLoader(train_dataset, batch_size = args.env_.batch_size, num_workers = args.env_.num_worker,pin_memory=True)

    # 채널, 클래스 갯수
    net = classifier32ABN(1,num_classes=args.env_.num_classes)
    
    ## 노이즈 디멘젼, 크기, 생성할 채널
    netG = gan.Generator32(args.parameters_.nz, 64, 1)
    
    ## 채널, 최종 크기인듯
    netD = gan.Discriminator32(1, 64)
    
    fixed_noise = torch.FloatTensor(64, args.parameters_.nz, 1, 1).normal_(0, 1)
    criterionD = nn.BCELoss()
    
    
    # weight_pl,temp,num_classes,feat_dim
    criterion = ARPLoss(0.1,1.0,args.env_.num_classes,args.env_.feat_dim)

    params_list = [{'params': net.parameters()},{'params': criterion.parameters()}]

    optimizer = torch.optim.Adam(params_list, lr=args.parameters_.lr)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.parameters_.gan_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.parameters_.gan_lr, betas=(0.5, 0.999))

 
    train_cs(net, netD, netG, criterion, criterionD,optimizer, optimizerD, optimizerG,trainloader,args.env_.epochs,device,args)
 
    
 
 
 
        
    # with mlflow.start_run(run_id=args.env_.mlflow_run_id) as run:
    #     mlflow.log_metric("똘끼", 100)
    #     mlflow.end_run()
    
def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, epoch , device,args):
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()
    net.train()
    netD.train()
    netG.train()
    torch.cuda.empty_cache()
    
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
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
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
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
        ## y torch.Size([64, 6])
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), args.parameters_.nz, args.parameters_.ns, args.parameters_.ns).normal_(0, 1).to(device=device)
        noise = Variable(noise)
        fake = netG(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + args.parameters_.beta * F_loss_fake
        total_loss.backward()
        optimizer.step()
        
        losses.update(total_loss.item(), labels.size(0))
        
        print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
    

# def test(net, criterion, testloader, outloader, epoch, device, args):
#     net.eval()
#     correct, total = 0, 0

#     torch.cuda.empty_cache()

#     _pred_k, _pred_u, _labels = [], [], []

#     with torch.no_grad():
#         for data, labels in testloader:
            
#             with torch.set_grad_enabled(False):
#                 data = data.to(device=device)
#                 labels = labels.to(device=device)
#                 x, y = net(data, True)
#                 logits, _ = criterion(x, y)
#                 predictions = logits.data.max(1)[1]
#                 total += labels.size(0)
#                 correct += (predictions == labels.data).sum()
            
#                 _pred_k.append(logits.data.cpu().numpy())
#                 _labels.append(labels.data.cpu().numpy())

#         for batch_idx, (data, labels) in enumerate(outloader):
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
            
#             with torch.set_grad_enabled(False):
#                 x, y = net(data, True)
#                 # x, y = net(data, return_feature=True)
#                 logits, _ = criterion(x, y)
#                 _pred_u.append(logits.data.cpu().numpy())

#     # Accuracy
#     acc = float(correct) * 100. / float(total)
#     print('Acc: {:.5f}'.format(acc))

#     _pred_k = np.concatenate(_pred_k, 0)
#     _pred_u = np.concatenate(_pred_u, 0)
#     _labels = np.concatenate(_labels, 0)
    
#     # Out-of-Distribution detction evaluation
#     x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
#     results = evaluation.metric_ood(x1, x2)['Bas']
    
#     # OSCR
#     _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

#     results['ACC'] = acc
#     results['OSCR'] = _oscr_socre * 100.

#     return results



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
    
    # mp.spawn(trainer, args=(config.env_.gpus, config), nprocs=config.env_.gpus, join=True)
    trainer(0, config.env_.gpus, config)
    
    ## 멀티 프로세싱 작업이 끝났으므로, 결과 임시저장 폴더 삭제
    if os.path.exists(config.env_.data_save_dir):
        shutil.rmtree(config.env_.data_save_dir)
    

    
if __name__ == '__main__':
    main()