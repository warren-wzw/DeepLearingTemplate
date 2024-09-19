import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride=1,padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h) #X*Wq
        k = self.proj_k(h) #X*Wk
        v = self.proj_v(h) #X*Wv

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))#QKt/sqrt(dk)[1,4096,4096]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)#softmax(QKt/sqrt(dk))

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v) #softmax(QKt/sqrt(dk))*V=Attention(Q,K,V)=head
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        out =  self.gamma+h + x #resnet
        return out
       
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_feedforward=1024, num_layers=3):
        super(TransformerEncoder, self).__init__()
        
        # Patch embedding: Flatten spatial dimensions and map to reduced feature dimension
        self.patch_embed = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels // 2, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reshape back to original dimensions after Transformer processing
        self.unpatch_embed = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, C, H, W = x.shape
        
        # Patch Embedding: Flatten the spatial dimensions
        x = self.patch_embed(x).view(batch_size, C // 2, -1).permute(2, 0, 1)  # (N, B, C)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (N, B, C)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(batch_size, C // 2, H, W)
        x = self.unpatch_embed(x)
        
        return x

class TEMPLATE(nn.Module):
    def __init__(self):
        super(TEMPLATE, self).__init__()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        print()

