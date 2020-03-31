import torch
import torch.nn as nn
from torch.autograd import Variable


class Transpose3D(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride, padding):
        super(Transpose3D, self).__init__()
        self.transpose3d = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, k_size, stride, padding),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(),
        )

    def forward(self,x):
        x = self.transpose3d(x)
        return x


class Conv3D(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride, padding, l_relu):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, k_size, stride, padding),
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(l_relu),
        )

    def forward(self,x):
        x = self.conv3d(x)
        return x


class IWGenerator(nn.Module):
    def __init__(self):
        super(IWGenerator, self).__init__()
        output_size, half, forth, eighth, sixteenth = 32, 16, 8, 4, 2
	    gf_dim = 256
        
        self.Linear = nn.Linear(200, gf_dim*sixteenth*sixteenth*sixteenth)
        self.main = nn.Sequential(
            Transpose3D(200, gf_dim, 4, 2, 0),
            Transpose3D(gf_dim, gf_dim/2, 4, 2, 1),
            Transpose3D(gf_dim/2, gf_dim/4, 4, 2, 1),
            Transpose3D(gf_dim/4, gf_dim/8, 4, 2, 1),
            Transpose3D(gf_dim/8, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = x.view(1, 200)
        x = self.Linear(x)
        x = x.view(1, 2, 2, 2, 256)
        x = nn.BatchNorm3d(256)
        x = nn.ReLU(x)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, output_size, output_unit):
        super(IWDiscriminator, self).__init__()
        df_dim = output_size
        self.main = nn.Sequential(
            Conv3D(1, df_dim, 4, 2, 1, 0.2),
            Conv3D(df_dim, df_dim*2, 4, 2, 1, 0.2),
            Conv3D(df_dim*2, df_dim*4, 4, 2, 1, 0.2),
            Conv3D(df_dim*4, df_dim*8, 4, 2, 1, 0.2),
        )
        self.Linear = nn.Linear(2048, output_unit)

    def forward(self, x):
        # x's size: batch_size * 1 * 64 * 64 * 64
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        x = nn.Sigmoid(x)
        return x
        