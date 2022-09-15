import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import Vit
from dataset import Hanguldataset





def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    total_correct = 0
    for i,(img,target) in enumerate(train_loader):
        optimizer.zero_grad() # model의 gradient 값을 0으로 설정
        
        # summary(model,(1,64,64))
        outputs = model(img.to(device))
        
        
        
        loss = criterion(outputs, target.to(device))
        loss.backward() # backward 함수를 호출해 gradient 계산
        optimizer.step() # 모델의 학습 파라미터 갱신
        
        running_loss += loss.item() / len(train_loader)
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target.to(device)).sum().item() 
        total_correct += correct
        
        if i % 20 == 0:
            print(f'[{len(train_loader)}/{i}]\t loss : {loss:.4f} \t accuracy : {correct/args.batch_size*100:.2f}% \t{correct}/{args.batch_size}')    
    

    print(f"{epoch} epoch 평균 train loss : {running_loss} \t accuracy : {total_correct/(args.batch_size*len(train_loader))*100:.2f}% \t {total_correct}/{args.batch_size*len(train_loader)}")
    print("\n")
    torch.save(model.state_dict(), './vit_s4.pt')
    



def test(model,device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    data_len = 0
    test_loss_tensor = 0
    correct_tensor = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            test_loss_tensor += F.cross_entropy(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_tensor += pred.eq(target.view_as(pred)).sum()
            data_len += len(data)

    test_loss /= data_len


    print("Test average loss: {}, correct predictons: {}, total: {}, accuracy: {:3f}% \n".format(test_loss_tensor.item() / len(test_loader.dataset), correct_tensor.item(), len(test_loader.dataset),
            100.0 * correct_tensor.item() / len(test_loader.dataset)))

    #print('\nTest  set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #        test_loss, correct, data_len,
    #        100. * correct / data_len))
    
    
def trainer(args):


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.RandomRotation(25), transforms.ToTensor()])
    train_dataset = Hanguldataset(g_target_char_txt=args.target_char_txt,ttf_dir=args.train_ttf_dir,transform=train_transform)
  

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True)    


    test_transform = transforms.Compose([transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()])
    test_dataset = Hanguldataset(g_target_char_txt=args.target_char_txt,ttf_dir=args.val_ttf_dir,transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True)
    


    model = Vit(img_size=args.image_size,patch_size=16,in_chans=1,n_classes=2350,embed_dim=768,n_heads=12,depth=12)
    model.load_state_dict(torch.load('./vit.pth'))
    model = model.to(device)
    
    

    

    # for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch)
        # test(model,device,test_loader)


    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--image_size', type=int, default=128, metavar='N',
                        help='input image size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gpus', type=int, default=2, metavar='N',
                        help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train_ttf_dir', default="./data/ttf_file",
                        help='train ttf file dir')
    parser.add_argument('--val_ttf_dir', default="./data/ttf_file",
                        help='val ttf file dir')
    parser.add_argument('--target_char_txt', default="./data/target.txt",
                        help='val ttf file dir')
    
    args = parser.parse_args()
    trainer(args)




if __name__ == '__main__':
    main()