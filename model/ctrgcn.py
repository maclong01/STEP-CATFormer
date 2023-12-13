import math
import pdb
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


import torchvision.ops
import einops
import torch
import torchvision.ops
from torch import heaviside, nn, per_channel_affine_float_qparams
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from torch import einsum
from einops import rearrange, reduce, repeat

import torch.nn.functional as F
from torch.nn.modules.utils import _triple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import clip
from Text_Prompt import *
from tools import *
from einops import rearrange, repeat


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V   N,T,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        # self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y
    
    
    
    
    
##################################################################################################################
##################################################################################################################

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


    



class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        
        # initialize
        nn.init.normal_(self.to_k.weight, 0, math.sqrt(2. / inner_dim))
        nn.init.normal_(self.to_q.weight, 0, math.sqrt(2. / inner_dim))
        nn.init.normal_(self.to_v.weight, 0, math.sqrt(2. / inner_dim))
        self.apply(weights_init)
        
    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out





class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, input_dim, output_dim,num_point=25 ,cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0., residual=True, att=False):
        super().__init__()


        large_dim = (output_dim//2)//2
        small_dim = large_dim+(output_dim//2)
        
        self.transformer_enc_small1 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small2 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(num_point, small_dim,num_heads=8,ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        self.transformer_enc_large1 =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_large2 =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_large_T = Spatial_TransformerEncoder(num_point, large_dim,num_heads=8, ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        
        
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = 8, dim_head = large_dim // 8, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = 8, dim_head = small_dim // 8, dropout = dropout)),
            ]))

        self.pool = 'cls'

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, output_dim)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, output_dim)
        )
        
        
        self.feedforward = FeedForward(in_features=output_dim*num_point, hidden_features=int(output_dim*num_point), 
                                        out_features=output_dim*num_point, do_rate=0.)
        self.norm2 = nn.LayerNorm(output_dim*num_point)
        
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, x1,x2):

        up_list = torch.Tensor([2,3,10,11,6,7,8,9,4,5,20,21,22,23,24]).long()
        down_list = torch.Tensor([16,17,18,19,12,13,14,15,0,1]).long()

        headhand_feature1 = (x1[:,:,:,up_list])
        foothip_feature1 = (x1[:,:,:,down_list])
        
        headhand_feature2 = (x2[:,:,:,up_list])
        foothip_feature2 = (x2[:,:,:,down_list])
        
        foothip_feature1 = self.transformer_enc_small1(foothip_feature1)
        foothip_feature2 = self.transformer_enc_small2(foothip_feature2)
        
        headhand_feature1 = self.transformer_enc_large1(headhand_feature1)
        headhand_feature2 = self.transformer_enc_large2(headhand_feature2)
        
        xs = self.transformer_enc_small_T(foothip_feature1,foothip_feature2,foothip_feature1)
        xl = self.transformer_enc_large_T(headhand_feature1,headhand_feature2,headhand_feature1)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs= rearrange(xs, 'b c t v -> (b t) v c')
        xl= rearrange(xl, 'b c t v -> (b t) v c')
  
        

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs= rearrange(xs, '(b t) v c -> b c t v', t = ts,v=vs)
        xl= rearrange(xl, '(b t) v c -> b c t v', t = tl,v=vl)

        branch_out=[]
        branch_out.append(xs)
        branch_out.append(xl)
        x=torch.cat(branch_out,dim=3)

        b,c,f,j=x.shape
        x_tm = rearrange(x, 'b c f j   -> b f (j c)')
        x_out = x_tm + self.feedforward(self.norm2(x_tm))        
        x_out = rearrange(x_out, 'b f (j c)  -> b c f j', j=j)
        return x_out



    
    
    
    
    
    
    
class MultiScaleTransformerEncoder_hand_leg(nn.Module):

    def __init__(self, input_dim, output_dim,num_point=25 ,cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0., residual=True, att=False):
        super().__init__()

        

        large_dim = (output_dim//2)//2
        small_dim = large_dim+(output_dim//2)
        
                       


        self.transformer_enc_small1 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small2 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(num_point, small_dim,num_heads=8,ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        
        self.transformer_enc_large1 =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_large2 =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))                                 
        self.transformer_enc_large_T = Spatial_TransformerEncoder(num_point, large_dim,num_heads=8, ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        
        


        
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = 8, dim_head = large_dim // 8, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = 8, dim_head = small_dim // 8, dropout = dropout)),
            ]))

        self.pool = 'cls'

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, output_dim)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, output_dim)
        )
        
        
        self.feedforward = FeedForward(in_features=output_dim*num_point, hidden_features=int(output_dim*num_point), 
                                        out_features=output_dim*num_point, do_rate=0.)
        self.norm2 = nn.LayerNorm(output_dim*num_point)
        
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, x1,x2):

        up_list = torch.Tensor([2,3,10,11,6,7,8,9,4,5,20,21,22,23,24]).long()
        down_list = torch.Tensor([16,17,18,19,12,13,14,15,0,1]).long()
        
        headhand_feature1 = (x1[:,:,:,up_list])
        foothip_feature1 = (x1[:,:,:,down_list])
        
        headhand_feature2 = (x2[:,:,:,up_list])
        foothip_feature2 = (x2[:,:,:,down_list])
        
        foothip_feature1 = self.transformer_enc_small1(foothip_feature1)
        foothip_feature2 = self.transformer_enc_small2(foothip_feature2)
        
        headhand_feature1 = self.transformer_enc_large1(headhand_feature1)
        headhand_feature2 = self.transformer_enc_large2(headhand_feature2)
        
        xs = self.transformer_enc_small_T(foothip_feature1,foothip_feature2,foothip_feature1)
        xl = self.transformer_enc_large_T(headhand_feature1,headhand_feature2,headhand_feature1)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs= rearrange(xs, 'b c t v -> (b t) v c')
        xl= rearrange(xl, 'b c t v -> (b t) v c')
  
        

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs= rearrange(xs, '(b t) v c -> b c t v', t = ts,v=vs)
        xl= rearrange(xl, '(b t) v c -> b c t v', t = tl,v=vl)

        branch_out=[]
        branch_out.append(xs)
        branch_out.append(xl)
        x=torch.cat(branch_out,dim=3)

        b,c,f,j=x.shape
        x_tm = rearrange(x, 'b c f j   -> b f (j c)')
        x_out = x_tm + self.feedforward(self.norm2(x_tm))        
        x_out = rearrange(x_out, 'b f (j c)  -> b c f j', j=j)
        return x_out
    
    



class MultiScaleTransformerEncoder_hand(nn.Module):

    def __init__(self, input_dim, output_dim,num_point=25 ,cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0., residual=True, att=False):
        super().__init__()

        
        large_dim = (output_dim//2)//2
        small_dim = large_dim+(output_dim//2)
        
                       

        self.transformer_enc_small = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(num_point, small_dim,num_heads=8,ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        self.transformer_enc_large =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))                               
        self.transformer_enc_large_T = Spatial_TransformerEncoder(num_point, large_dim,num_heads=8, ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = 8, dim_head = large_dim // 8, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = 8, dim_head = small_dim // 8, dropout = dropout)),
            ]))

        self.pool = 'cls'

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, output_dim)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, output_dim)
        )
        
        
        self.feedforward = FeedForward(in_features=output_dim*num_point, hidden_features=int(output_dim*num_point), 
                                        out_features=output_dim*num_point, do_rate=0.)
        self.norm2 = nn.LayerNorm(output_dim*num_point)
        
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, x1,x2):

        up_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        down_list = torch.Tensor([16,17,18,19,12,13,14,15,0,1,2,3,20]).long()

        headhand_feature = (x1[:,:,:,up_list])
        foothip_feature = (x2[:,:,:,down_list])
        
        xs = self.transformer_enc_small(foothip_feature)
        xs = self.transformer_enc_small_T(xs,xs,xs)
        
        xl = self.transformer_enc_large(headhand_feature)
        xl = self.transformer_enc_large_T(xl,xl,xl)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs= rearrange(xs, 'b c t v -> (b t) v c')
        xl= rearrange(xl, 'b c t v -> (b t) v c')
  
        

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)


        xs= rearrange(xs, '(b t) v c -> b c t v', t = ts)
        xl= rearrange(xl, '(b t) v c -> b c t v', t = tl)


        branch_out=[]
        branch_out.append(xs)
        branch_out.append(xl)
        x=torch.cat(branch_out,dim=3)

        b,c,f,j=x.shape
        x_tm = rearrange(x, 'b c f j   -> b f (j c)')
        x_out = x_tm + self.feedforward(self.norm2(x_tm))        
        x_out = rearrange(x_out, 'b f (j c)  -> b c f j', j=j)
        return x_out




    
class MultiScaleTransformerEncoder_leg(nn.Module):

    def __init__(self, input_dim, output_dim,num_point=25 ,cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0., residual=True, att=False):
        super().__init__()

        

        large_dim = (output_dim//2)//2
        small_dim = large_dim+(output_dim//2)

        self.transformer_enc_small = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1,1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(num_point, small_dim,num_heads=8,ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        self.transformer_enc_large =  nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1,1))                               
        self.transformer_enc_large_T = Spatial_TransformerEncoder(num_point, large_dim,num_heads=8, ff_expand=1.0,
                                       qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = 8, dim_head = large_dim // 8, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = 8, dim_head = small_dim // 8, dropout = dropout)),
            ]))

        self.pool = 'cls'

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, output_dim)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, output_dim)
        )
        
        
        self.feedforward = FeedForward(in_features=output_dim*num_point, hidden_features=int(output_dim*num_point), 
                                        out_features=output_dim*num_point, do_rate=0.)
        self.norm2 = nn.LayerNorm(output_dim*num_point)
        
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, x1,x2):

        up_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24,1,2,3,20]).long()
        down_list = torch.Tensor([16,17,18,19,12,13,14,15,0]).long()

        headhand_feature = (x1[:,:,:,up_list])
        foothip_feature = (x2[:,:,:,down_list])
        
        xs = self.transformer_enc_small(foothip_feature)
        xs = self.transformer_enc_small_T(xs,xs,xs)
        
        xl = self.transformer_enc_large(headhand_feature)
        xl = self.transformer_enc_large_T(xl,xl,xl)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs= rearrange(xs, 'b c t v -> (b t) v c')
        xl= rearrange(xl, 'b c t v -> (b t) v c')
  
        

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs= rearrange(xs, '(b t) v c -> b c t v', t = ts,v=vs)
        xl= rearrange(xl, '(b t) v c -> b c t v', t = tl,v=vl)

        branch_out=[]
        branch_out.append(xs)
        branch_out.append(xl)
        x=torch.cat(branch_out,dim=3)

        b,c,f,j=x.shape
        x_tm = rearrange(x, 'b c f j   -> b f (j c)')
        x_out = x_tm + self.feedforward(self.norm2(x_tm))        
        x_out = rearrange(x_out, 'b f (j c)  -> b c f j', j=j)
        return x_out



    
    
    
    
    
    
class DTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DTemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=out_channels, 
            bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class Multi_DTemporalConv_Branch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        global iii
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        
        branch_channels = branch_channels
        branch_channels2 = branch_channels
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                DTemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels2, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            #nn.BatchNorm2d(branch_channels2)  
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels2, kernel_size=1, padding=0, stride=(stride,1)),
            #nn.BatchNorm2d(branch_channels2)
        ))
        
         # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

        # initialize
        self.apply(weights_init)
        
    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

    
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        
        hidden_channels = int(channels * expansion_factor)
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,groups=hidden_channels * 2, bias=False)  
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x
    
    
    
class MDTA_T(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_T, self).__init__()
        self.num_heads = num_heads
        self.dim = channels /self.num_heads 
        
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
        self.v = Multi_DTemporalConv_Branch(channels , channels , kernel_size=7,stride=1, dilations=[2,3])
        
        self.q = nn.Conv2d(channels, channels , kernel_size=1, bias=False)
        self.q_conv = nn.Conv2d(channels , channels , kernel_size=3, padding=1, groups=channels , bias=False)
        
        self.k = nn.Conv2d(channels, channels , kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(channels , channels , kernel_size=3, padding=1, groups=channels , bias=False)
        
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # initialize
        self.apply(weights_init)
        
    def forward(self, q,k,v):
        b, c, t, j = v.shape
        
        q = self.q_conv(self.q(q))
        k = self.k_conv(self.k(k))
        v = self.v(v)
        
        q = rearrange(q, 'b c t j  -> b t (j c)').contiguous()
        k = rearrange(k, 'b c t j  -> b t (j c)').contiguous()
        v = rearrange(v, 'b c t j  -> b t (j c)').contiguous()
        
        q = q.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        x = rearrange(x, 'b t (j c) -> b c t j', j=j)
        out = self.project_out(x)
        return out
    
    
    
class Temporal_TransformerEncoderv2(nn.Module):
    def __init__(self, num_joint=25, dim_emb=128, 
                num_heads=8, ff_expand=4.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.,):

        super(Temporal_TransformerEncoderv2, self).__init__()

        self.norm_q = nn.LayerNorm(dim_emb)
        self.norm_k = nn.LayerNorm(dim_emb)
        self.norm_v = nn.LayerNorm(dim_emb)
        self.attn = MDTA_T(dim_emb, 8)
        self.norm2 = nn.LayerNorm(dim_emb)
        self.ffn = GDFN(dim_emb, ff_expand)

        self.num_joints = num_joint
        self.add_coeff = nn.Parameter(torch.zeros(5,self.num_joints))
        
        self.transform = nn.Sequential(
            nn.BatchNorm2d(dim_emb), nn.GELU(), nn.Conv2d(dim_emb, dim_emb, kernel_size=1))

        self.bn = nn.BatchNorm2d(dim_emb)
        
    def forward(self, q,k,v,p,mask=None):
        b_o, c_o, t_o, v_o = q.shape
        
        q = torch.cat([q, p,q.mean(-1, keepdim=True)], -1)
        k = torch.cat([k, p,k.mean(-1, keepdim=True)], -1)
        v = torch.cat([v, p,v.mean(-1, keepdim=True)], -1)
        
        b, c, t, j = q.shape
        
        q_unnorm=q
        
        q=self.norm_q(q.reshape(b, c, -1).transpose(-2, -1).contiguous())
        k=self.norm_k(k.reshape(b, c, -1).transpose(-2, -1).contiguous())
        v=self.norm_v(v.reshape(b, c, -1).transpose(-2, -1).contiguous())
        
        q=q.transpose(-2, -1).contiguous().reshape(b, c, t, j)
        k=k.transpose(-2, -1).contiguous().reshape(b, c, t, j)
        v=v.transpose(-2, -1).contiguous().reshape(b, c, t, j)
        
        x = q_unnorm + self.attn(q,k,v)
        
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, t, j))
        
        local_feat = x[..., :v_o]
        global_feat = x[..., v_o:]
        
        global_feat = torch.einsum('nctd,dv->nctv', global_feat, self.add_coeff[:v_o])
        feat = local_feat + global_feat
        feat = self.transform(feat)
        x = self.bn(feat)
        
        return x
    
    
    
    
class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.W_q = nn.Linear(dim_emb, dim_emb , bias=qkv_bias)
        self.W_k = nn.Linear(dim_emb, dim_emb , bias=qkv_bias)
        self.W_v = nn.Linear(dim_emb, dim_emb , bias=qkv_bias)
        
        self.proj = nn.Linear(dim_emb, dim_emb)  
        
        nn.init.normal_(self.W_q.weight, 0, math.sqrt(2. / dim_emb ))
        nn.init.normal_(self.W_k.weight, 0, math.sqrt(2. / dim_emb ))
        nn.init.normal_(self.W_v.weight, 0, math.sqrt(2. / dim_emb ))
        
        nn.init.normal_(self.proj.weight, 0, math.sqrt(2. / dim_emb))
    def forward(self, q,k,v ,mask=None):

        B, N, C = q.shape  

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.reshape(B, N,self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N,self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N,self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    
class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / hidden_features))
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / out_features))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class Spatial_TransformerEncoder(nn.Module):
    def __init__(self, num_joint=50, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.,):
        super(Spatial_TransformerEncoder, self).__init__()
        self.normq_sp = nn.LayerNorm(dim_emb)
        self.normk_sp = nn.LayerNorm(dim_emb)
        self.normv_sp = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)
        
        self.feedforward = FeedForward(in_features=dim_emb, hidden_features=int(dim_emb*4), 
                                        out_features=dim_emb, do_rate=proj_do_rate)
        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        
    def forward(self, q,k,v,mask=None):

        b, c, f, j = q.shape

        q = rearrange(q, 'b c f j   -> (b f) j c')
        k = rearrange(k, 'b c f j   -> (b f) j c')
        v = rearrange(v, 'b c f j   -> (b f) j c')
        
        ## spatial-MHA attention 
        x_sp = q + self.attention_sp(self.normq_sp(q),self.normk_sp(k),self.normv_sp(v), mask=None)
        ## spatial-MHA ffn
        x_out = x_sp
        x_out = x_out + self.feedforward(self.norm2(x_out))
        x_out = rearrange(x_out, '(b f) j c  -> b c f j', b=b,f=f)
        return x_out   
    
    
    
##################################################################################################################
##################################################################################################################
    
    

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

######################################################################################
######################################################################################

class Model_lst_4part(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=0):
        super(Model_lst_4part, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        base_channel = 64
        base_channel2=base_channel*4
        
# Temporal fusion
        self.fusion1= TemporalConv(base_channel,base_channel*4,kernel_size=1,stride=4,dilation=1)
        self.fusion2= TemporalConv(base_channel*2,base_channel*4,kernel_size=1,stride=2,dilation=1)
        self.fusion3= TemporalConv(base_channel*4,base_channel*4,kernel_size=1,stride=1,dilation=1)
        self.fusion4= TemporalConv(base_channel*4,base_channel*4,kernel_size=1,stride=1,dilation=1)
        self.fusion_last= nn.Conv2d(base_channel*4,base_channel*4,kernel_size=1)
        
# Embedding
        self.to_joint_embedding = nn.Linear(base_channel2, base_channel2)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel2))
        
# Temporal
        self.Temporal_TransformerEncoder1 = Temporal_TransformerEncoderv2(num_point, base_channel2, 
                      num_heads=8, ff_expand=4.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        self.Temporal_TransformerEncoder2 = Temporal_TransformerEncoderv2(num_point, base_channel2, 
                      num_heads=8, ff_expand=4.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)

# Spratial cross 
        self.cross_hand = MultiScaleTransformerEncoder_hand(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_leg = MultiScaleTransformerEncoder_leg(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_hand_leg = MultiScaleTransformerEncoder_hand_leg(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_up_dowm = MultiScaleTransformerEncoder(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
# MLP
        self.mlp = nn.Sequential(
                                nn.Linear(base_channel2*num_point, base_channel2*num_point),
                                nn.GELU(),
                                nn.LayerNorm(base_channel2*num_point),
                                nn.Linear(base_channel2*num_point, base_channel2*num_point),
                                nn.GELU(),
                                nn.LayerNorm(base_channel2*num_point),
                                )
        self.norm = nn.LayerNorm(base_channel2*num_point)
        
# Predict layer 

        self.fc1 = nn.Linear(base_channel2, num_class)
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / num_class))

        
        
        if drop_out:
            self.drop_out1 = nn.Dropout(drop_out)
        else:
            self.drop_out1 = lambda x: x

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

# Feature extract 
        x = self.l1(x)
        x_1=self.fusion1(x)             #T
        x_l1=x_1.mean(-1, keepdim=True)
        
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        x_5=self.fusion2(x)             #T/2
        x_l5=x_5.mean(-1, keepdim=True)
        
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        
        x_8=self.fusion3(x)             #T/4
        x_l8=x_8.mean(-1, keepdim=True)
        
        x = self.l9(x)
        x = self.l10(x)
        
        x_10=self.fusion4(x)          #T/4
        x_l10=x_10.mean(-1, keepdim=True)
        

        # N*M,C,T,V
        c_new = x.size(1)
        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3,20]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,2,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))
        
        x_lst = x.view(N, M, c_new, -1)
        x_lst = x_lst.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x_lst)

        
        # N*M,C,T,V
        c_new = x.size(1)
        x_ab_ = x.view(N, M, c_new, -1)
        x_ab_ = x_ab_.mean(3).mean(1)
        x_ab_ = self.drop_out1(x_ab_)
        output_ab=self.fc1(x_ab_)

        
#Temporal fusion concat

        p=torch.cat([x_l1,x_l5,x_l8,x_l10], -1)
        p = self.fusion_last(p)
        
#Transformer

        N1, C1, T1, V1 = x.size()
        
        x_ = rearrange(x, 'n c t v -> (n t) v c').contiguous()
        x_ = self.to_joint_embedding(x_)
        x_ += self.pos_embedding[:, :self.num_point]
        x_ = rearrange(x_, '(n t) v c -> n c t v', n=N1, t=T1).contiguous()
        
        x_s1=self.cross_hand(x_,x_)             #hand
        x_s2=self.cross_leg(x_,x_)              #leg 
        x_t1=self.Temporal_TransformerEncoder1(x_s1,x_s2,x_s1,p)  #q,k,v,p
        
        x_s1=self.cross_hand_leg(x_s1,x_s2)     #hand leg
        x_s2=self.cross_up_dowm(x_s2,x_s1)      #down up
        x_t2=self.Temporal_TransformerEncoder2(x_s1,x_s2,x_s1,p)  #q,k,v,p
        
        x_t=x_t1+x_t2
        
        x_t = rearrange(x_t, 'b c f j -> b f (j c)')
        x_t = self.mlp(self.norm(x_t))
        x = rearrange(x_t, 'b f (j c) -> b c f j',j=V1)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x),output_ab , feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    
    

######################################################################################
######################################################################################


class Model_lst_4part_bone(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=1):
        super(Model_lst_4part_bone, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

##############################################################################################
        base_channel = 64
        base_channel2=base_channel*4
        
# Temporal fusion
        self.fusion1= TemporalConv(base_channel,base_channel*4,kernel_size=1,stride=4,dilation=1)
        self.fusion2= TemporalConv(base_channel*2,base_channel*4,kernel_size=1,stride=2,dilation=1)
        self.fusion3= TemporalConv(base_channel*4,base_channel*4,kernel_size=1,stride=1,dilation=1)
        self.fusion4= TemporalConv(base_channel*4,base_channel*4,kernel_size=1,stride=1,dilation=1)
        self.fusion_last= nn.Conv2d(base_channel*4,base_channel*4,kernel_size=1)
        
# Embedding
        self.to_joint_embedding = nn.Linear(base_channel2, base_channel2)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel2))
        
# Temporal
        self.Temporal_TransformerEncoder1 = Temporal_TransformerEncoderv2(num_point, base_channel2, 
                      num_heads=8, ff_expand=4.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)
        
        self.Temporal_TransformerEncoder2 = Temporal_TransformerEncoderv2(num_point, base_channel2, 
                      num_heads=8, ff_expand=4.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0.)

# Spratial cross 
        self.cross_hand = MultiScaleTransformerEncoder_hand(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_leg = MultiScaleTransformerEncoder_leg(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_hand_leg = MultiScaleTransformerEncoder_hand_leg(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        self.cross_up_dowm = MultiScaleTransformerEncoder(base_channel2, base_channel2,  num_point=num_point,
                                                              cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.)
        
# MLP
        self.mlp = nn.Sequential(
                                nn.Linear(base_channel2*num_point, base_channel2*num_point),
                                nn.GELU(),
                                nn.LayerNorm(base_channel2*num_point),
                                nn.Linear(base_channel2*num_point, base_channel2*num_point),
                                nn.GELU(),
                                nn.LayerNorm(base_channel2*num_point),
                                )
        self.norm = nn.LayerNorm(base_channel2*num_point)
        
# Predict layer 
        self.fc1 = nn.Linear(base_channel2, num_class)
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / num_class))

        
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out1 = nn.Dropout(drop_out)
        else:
            self.drop_out1 = lambda x: x


        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))
        
        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

# Feature extract 
        x = self.l1(x)
        x_1=self.fusion1(x)             #T
        x_l1=x_1.mean(-1, keepdim=True)
        
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        x_5=self.fusion2(x)             #T/2
        x_l5=x_5.mean(-1, keepdim=True)
        
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        
        x_8=self.fusion3(x)             #T/4
        x_l8=x_8.mean(-1, keepdim=True)
        
        x = self.l9(x)
        x = self.l10(x)
        
        x_10=self.fusion4(x)          #T/4
        x_l10=x_10.mean(-1, keepdim=True)
        
        
        
        # N*M,C,T,V
        c_new = x.size(1)
        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,20,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x_lst = x.view(N, M, c_new, -1)
        x_lst = x_lst.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x_lst)
            
            
        # N*M,C,T,V
        c_new = x.size(1)
        x_ab_ = x.view(N, M, c_new, -1)
        x_ab_ = x_ab_.mean(3).mean(1)
        x_ab_ = self.drop_out1(x_ab_)
        output_ab=self.fc1(x_ab_)

        
#Temporal fusion concat
        p=torch.cat([x_l1,x_l5,x_l8,x_l10], -1)
        p = self.fusion_last(p)
        
#Transformer

        N1, C1, T1, V1 = x.size()
        
        x_ = rearrange(x, 'n c t v -> (n t) v c').contiguous()
        x_ = self.to_joint_embedding(x_)
        x_ += self.pos_embedding[:, :self.num_point]
        x_ = rearrange(x_, '(n t) v c -> n c t v', n=N1, t=T1).contiguous()
        
        x_s1=self.cross_hand(x_,x_)             #hand
        x_s2=self.cross_leg(x_,x_)              #leg 
        x_t1=self.Temporal_TransformerEncoder1(x_s1,x_s2,x_s1,p)  #q,k,v,p
        
        x_s1=self.cross_hand_leg(x_s1,x_s2)     #hand leg
        x_s2=self.cross_up_dowm(x_s2,x_s1)      #down up
        x_t2=self.Temporal_TransformerEncoder2(x_s1,x_s2,x_s1,p)  #q,k,v,p
        
        x_t=x_t1+x_t2
        
        x_t = rearrange(x_t, 'b c f j -> b f (j c)')
        x_t = self.mlp(self.norm(x_t))
        x = rearrange(x_t, 'b f (j c) -> b c f j',j=V1)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x),output_ab , feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    
######################################################################################
######################################################################################  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

