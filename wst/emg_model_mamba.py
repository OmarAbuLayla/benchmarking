# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. HELPER BLOCKS (From your GRU Model)
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialDropout(nn.Module):
    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p <= 0:
            return x
        b, c, h, w = x.shape
        # Drop entire channels/feature maps, not just individual neurons
        mask = x.new_ones(b, c, 1, 1)
        mask = F.dropout(mask, self.p, training=True)
        return x * mask

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ScatteringAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(6)
        self.dw = nn.Conv2d(6, 6, kernel_size=(1, 3), padding=(0, 1),
                            groups=6, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if x.dim() != 4:
            pass # Assume correct shape or handle error
        
        # Standardize input shape handling
        if x.shape[1] == 1:
            x = x.repeat(1, 6, 1, 1)
        if x.shape[1] == 36 and x.shape[-1] == 6:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.drop(x)
        return x

# ==========================================
# 2. BIDIRECTIONAL MAMBA BLOCK
# ==========================================
class PyMambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Kernel 17 to capture local context like a small RNN window
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=17, padding=8, groups=d_model)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        x = self.act(x) * F.sigmoid(z)
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x

class BiMambaStack(nn.Module):
    def __init__(self, d_model=512, n_layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'fwd': PyMambaBlock(d_model, dropout),
                'bwd': PyMambaBlock(d_model, dropout)
            }) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            # Forward pass
            out_fwd = layer['fwd'](x)
            # Backward pass
            x_flip = torch.flip(x, [1])
            out_bwd = layer['bwd'](x_flip)
            out_bwd = torch.flip(out_bwd, [1])
            
            # Sum fusion (keeps dimension same, simple and effective)
            x = out_fwd + out_bwd
            
        return self.norm(x)

# ==========================================
# 3. MAIN MODEL (CNN Backbone + Mamba)
# ==========================================
class EMGMamba(nn.Module):
    def __init__(self, n_classes=101, every_frame=True, d_model=512):
        super().__init__()
        self.every_frame = every_frame
        
        # --- A. CNN Feature Extractor (Copied from valid GRU model) ---
        self.scat_adapter = ScatteringAdapter()

        self.fronted2D = nn.Sequential(
            conv_bn_relu(6, 64, 3, 1, 1),
            ChannelAttention(64),
            SpatialDropout(0.20)
        )
        self.fronted2D1 = nn.Sequential(
            conv_bn_relu(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(0.25)
        )
        self.fronted2D2 = nn.Sequential(
            conv_bn_relu(64, 128, 3, 1, 1),
            ChannelAttention(128)
        )
        self.fronted2D3 = nn.Sequential(
            conv_bn_relu(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(0.25)
        )
        self.fronted2D4 = nn.Sequential(
            conv_bn_relu(128, 256, 3, 1, 1),
            ChannelAttention(256)
        )
        self.fronted2D5 = nn.Sequential(
            conv_bn_relu(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(0.25)
        )
        self.fronted2D6 = nn.Sequential(
            conv_bn_relu(256, 256, 3, 1, 1)
        )
        self.fronted2D7 = nn.Sequential(
            conv_bn_relu(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(3, 1)),
            nn.Dropout(0.25)
        )

        self.norm_tail = nn.LayerNorm(256)
        
        # Temporal fusion (Context Windowing)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        # --- B. Projector (CNN 256 -> Mamba 512) ---
        self.adapter_proj = nn.Linear(256, d_model)

        # --- C. Mamba Backend (Replaces GRU) ---
        self.mamba = BiMambaStack(d_model=d_model, n_layers=3, dropout=0.25)
        
        # --- D. Classifier ---
        self.head = nn.Linear(d_model, n_classes)

        self._initialize_weights()

    def forward(self, x):
        # 1. Feature Extraction (CNNs)
        x = self.scat_adapter(x)
        x = self.fronted2D(x)
        x = self.fronted2D1(x)
        x = self.fronted2D2(x)
        x = self.fronted2D3(x)
        x = self.fronted2D4(x)
        x = self.fronted2D5(x)
        x = self.fronted2D6(x)
        x = self.fronted2D7(x)

        # 2. Gaussian Noise Injection (Crucial Regularization)
        if self.training and torch.rand(1).item() < 0.15:
            x = x * (1.0 + 0.05 * torch.randn_like(x))

        # 3. Reshape for Sequence Modeling
        B, C, H, W = x.shape
        # Flatten H and W to create sequence
        x = x.reshape(B, C, -1).transpose(1, 2)  # (B, SeqLen, C)
        x = self.norm_tail(x)
        
        # 4. Temporal Conv
        x = x.transpose(1, 2) # (B, C, SeqLen)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2) # (B, SeqLen, C)

        # 5. Project and Mamba
        x = self.adapter_proj(x) # 256 -> 512
        x = self.mamba(x)
        
        # 6. Classify
        logits = self.head(x)
        
        if self.every_frame:
            return logits
        else:
            return logits[:, -1, :]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)