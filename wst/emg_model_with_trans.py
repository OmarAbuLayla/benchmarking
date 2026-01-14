# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Small helper
# ------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


# ------------------------------------------------------------
# Positional Encoding (for Transformer)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


# ------------------------------------------------------------
# GRU MODULE (fixed)
# ------------------------------------------------------------
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers, batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0),
            self.hidden_size, device=x.device
        )
        out, _ = self.gru(x, h0)

        if self.every_frame:
            return self.fc(out)
        else:
            return self.fc(out[:, -1, :])


# ------------------------------------------------------------
# MAIN EMG MODEL (Conv frontend + GRU or Transformer backend)
# ------------------------------------------------------------
class EMGNet(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True):
        super().__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses

        # -----------------------------
        # Convolutional frontend
        # -----------------------------
        self.fronted2D = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.fronted2D1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fronted2D3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.fronted2D5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fronted2D6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.fronted2D7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.5)
        )

        # -----------------------------
        # Backend GRU
        # -----------------------------
        self.gru = GRU(
            input_size=self.inputDim,
            hidden_size=hiddenDim,
            num_layers=2,
            num_classes=nClasses
        )

        # -----------------------------
        # Backend Transformer
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.pe = PositionalEncoding(256)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        self.transformer_fc = nn.Linear(256, nClasses)

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------
    def forward(self, x):
        B = x.size(0)

        # conv frontend
        x = self.fronted2D(x)
        x = self.fronted2D1(x)
        x = self.fronted2D2(x)
        x = self.fronted2D3(x)
        x = self.fronted2D4(x)
        x = self.fronted2D5(x)
        x = self.fronted2D6(x)
        x = self.fronted2D7(x)

        # final shape = (B, 256,  ? , ?)
        # flatten into sequence: tokens = time steps
        x = x.view(B, -1, 256)   # (B, T, 256)

        if self.mode in ["backendGRU", "finetuneGRU"]:
            return self.gru(x)

        elif self.mode == "transformer":
            x = self.pe(x)
            x = self.transformer(x)
            x = self.transformer_fc(x.mean(dim=1))   # global pooling
            return x

        else:
            raise Exception("Invalid mode selected")


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------
def emg_model(mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True):
    return EMGNet(
        mode, inputDim=inputDim,
        hiddenDim=hiddenDim,
        nClasses=nClasses,
        frameLen=frameLen,
        every_frame=every_frame
    )
