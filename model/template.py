import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        # Query, Key, Value
        Q = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C//8)
        K = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, N)
        V = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, N)
        # Attention
        attention = F.softmax(torch.bmm(Q, K), dim=1)  # (B, N, N)
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N) 
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out
    
class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_feedforward=1024, num_layers=3):
        super(TransformerBottleneck, self).__init__()
        
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

