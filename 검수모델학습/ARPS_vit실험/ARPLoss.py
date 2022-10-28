import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist

class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self,weight_pl,temp,num_classes,feat_dim):
        super(ARPLoss, self).__init__()

        ## options['weight_pl'] = 가중치 0.1
        self.weight_pl = weight_pl
        self.temp = 1.0
        
        ## 배치 사이즈 64
        
        # 알고있는 클래스 options['known': [6, 3, 4, 9, 8, 2]]
        # 모르는 클래스 options['unknown': [0, 1, 5, 7]]
        ## 그래서 options['num_classes'] == 6이된다
        ## 피쳐 디멘젼은 feat_dim = 128
        
        
        ## 거리 함수 정의
        self.Dist = Dist(num_classes=num_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, y, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        ## 라벨정보가 없으면 0
        if labels is None: return logits, 0
        
        
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size())
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
