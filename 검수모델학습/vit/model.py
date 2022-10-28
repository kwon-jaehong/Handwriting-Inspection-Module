import torch
import torch.nn as nn

# https://www.youtube.com/watch?v=ovB0ddFtzzA&t=876s

class patchembed(nn.Module):
    """ 원본이미지 -> 패치이미지로 만듬 패치 이미지 임베드
    
    Paramters
    ---------
    img_size : int
        이미지의 사이즈 (정사각형)
        변수값 들어갈때는 (img_size,img_size)로 들어감
    
    patch_size : int
        패치가 될 사이즈
        변수값 들어갈때는 (patch_size,patch_size)로 들어감
    
    int_chans : int
        입력이미지 채널수
        
    embed_dim : int
        임베딩할 차원

    """
    def __init__(self,img_size,patch_size,int_chans=3,embed_dim=768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        ## 패치 갯수
        self.n_patches = (img_size // patch_size)**2

        self.proj = nn.Conv2d(int_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
        
    def forward(self,x):
        """ 피드포워드 계산
        
        Parameters
        -----------
        x : torch.Tensor
            모양 '(배치,채널수,이미지사이즈,이미지사이즈)'
            
        Returns
        -------
        torch.tensor
            모양 '(배치,패치갯수,임베딩 차원)'
            
        """
        
        x = self.proj(x)
        x = x.flatten(2) # (배치,임배딩차원수,패치수)
        x = x.transpose(1,2) # (배치,패치수,임배딩차원수)
        return x
    
    
class Attention(nn.Module):
    """ 어텐션 메커니즘
    Parameters
    ----------
    dim : int
        인풋 차원
        
    n_heads : int
        어텐션 메카니즘 헤더 갯수

    qkv_bias : bool
        쿼리,키,벨류 바이어스 변수 설정할건지
        
    attn_p : float
        드롭아웃 확률 (쿼리,키,벨류)
    
    proj_p : float
        드롭아웃 확률 (출력 텐서)    
    
    
    Attributes
    ----------
    scale : float
        노멀라이징 
    qkv : nn.Linear
        키,쿼리,벨류
        
    proj : nn.Linear
        어텐션 값들 덴스레이어
        
    attn_drop, proj_drop : nn.Dropout
        드롭아웃 레이어    
    """
    
    def __init__(self,dim,n_heads=12,qkv_bias=True,attn_p=0.,proj_p=0.) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # 멀티헤드 어텐션 헤드는... 인코더의 전체차원에서 n_heads만큼 나누어줌
        self.scale = self.head_dim ** -0.5 ## 어텐션 벡터 스케일링
        
        
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim) ## 멀티헤더 어텐션은 입력,출력 차원의 갯수는 똑같음
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self,x):
        """ 전방향 연산 시작, (멀티헤더 어텐션은 입력,출력 차원의 갯수는 똑같음)
        
        Parameters
        ----------
        x : torch.Tensor
            모양 '(배치,패치수+1,dim)'
            패치수+1은 앞에 클래스 토큰
            
        Returns
        -------
        torch.Tensor
            모양 '(배치,패치수+1,dim)'
        
        """
        
        ## 배치수, 패치수, x의 차원
        ## 여기서 패치수는 임베딩된 벡터라 하나의 토큰으로 보아도 무방함
        n_samples, n_tokens, dim = x.shape
        
        
        ## 멀티헤더 셀프 어텐션은 입력과 출력의 차원이 같아야하는데 맞지 않다면 오류임 
        if dim != self.dim:
            raise ValueError
        
        
        ## qkv를 한꺼번에 계산 -> 리쉐이프
        qkv = self.qkv(x) # (배치,패치+1,3*dim)
        

        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim) # (배치,패치수+1,3,해더수,해더 차원)
        qkv = qkv.permute(2,0,3,1,4) # (3,배치,해더수,패치수+1,해더 차원)
        
        
        
        
        ## 쿼리,키,벨류 값 가져오기
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        ## 키값 전치
        k_t = k.transpose(-2,-1) # (배치,해더수,해더차원,패치수+1)
        
        ## 두행렬을 곱하고 스케일 조정
        dp = (q@k_t) * self.scale # (배치,해더수,패치수+1,패치수+1)
        
        
        ## 어텐션 맵 만듬 (소프트 맥스 & 드롭아웃)
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        
        weighted_avg = attn @ v # (배치,해더수,패치수+1,해더차원)
        weighted_avg = weighted_avg.transpose(1,2) # (배치,패치수+1,해더수,해더차원)
        weighted_avg = weighted_avg.flatten(2) # (배치,패치수+1,)
        
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        
        return x
        
class MLP(nn.Module):
    """ 멀티 레이어
    
    Parameters
    ----------
    in_features: int
        입력데이터 사이즈
        
    hidden_feactures : int
        히든 레이어 갯수
    
    out_feactures : int
        출력 사이즈
    
    p : float
        드롭아웃 확률
        
    """
    def __init__(self,in_features,hidden_feactures,out_feactures,p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden_feactures)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_feactures,out_feactures)
        self.drop = nn.Dropout(p)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
        
        
        
class Block(nn.Module):
    """ 트랜스 포머 블럭
    
    Parameters
    ----------
    dim : int
        임베딩 차원
    
    n_heads : int
        어텐션 해더 갯수
        
    mlp_ratio : float
        'dim'에 대한 'MLP' 모듈의 숨겨진 차원 크기를 결정
    
    qkv_bias : bool
        키,쿼리,블럭 바이어스 변수 설정
        
    p, attn_p : float
        드롭아웃 확률
    
    
    """
    def __init__(self,dim,n_heads,mlp_ratio=4.0,qkv_bias=True,p=0,attn_p=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,n_heads=n_heads,qkv_bias=qkv_bias,attn_p=attn_p,proj_p=p)
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        
        ## MLP레이어 임베딩차원은 -> 트랜스포머의 출력 벡터의 4배로   
        hidden_feactures = int(dim*mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_feactures=hidden_feactures,
            out_feactures=dim,
        )
        
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
        
          
class Vit(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=1,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 p=0.,
                 attn_p=0.,                 
                 ):
        super().__init__()
        
        self.patch_embed = patchembed(
            img_size=img_size,
            patch_size=patch_size,
            int_chans=in_chans,
            embed_dim=embed_dim
        )
        
        ## 임베드 벡터의 맨앞에 붙일 클래스 토큰
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        
        ## 포지션 파라미터들
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches,embed_dim))
        
        self.pos_drop = nn.Dropout(p=p)
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,                    
                )
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim,eps=1e-6)
        self.head = nn.Linear(embed_dim,n_classes)
        # self.initialize_weights()
         
    def forward(self,x):
        ## 배치수
        n_samples = x.shape[0]
        x = self.patch_embed(x) #
        
        cls_token = self.cls_token.expand(n_samples,-1,-1) # (배치,1,임베드차원)
        
        
        ## cls 토큰을 붙임
        x = torch.cat((cls_token,x),dim=1)
        
        x = x + self.pos_embed # (qocl,1+패치수,임베딩차원)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_token_final = x[:,0] # vit 마지막 결과값 가져옴
        x = self.head(cls_token_final)
        
        return x  

