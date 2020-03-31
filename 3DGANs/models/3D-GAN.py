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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            Transpose3D(200, 512, 4, 2, 0),
            Transpose3D(512, 256, 4, 2, 1),
            Transpose3D(256, 128, 4, 2, 1),
            Transpose3D(128, 64, 4, 2, 1),
            Transpose3D(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            Conv3D(1, 64, 4, 2, 1, 0.2),
            Conv3D(64, 128, 4, 2, 1, 0.2),
            Conv3D(128, 256, 4, 2, 1, 0.2),
            Conv3D(256, 512, 4, 2, 1, 0.2),
            Conv3D(512, 1, 4, 2, 0, 0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x's size: batch_size * 1 * 64 * 64 * 64
        x = self.main(x)
        return x.view(-1, x.size(1))
        

if __name__ == "__main__":
    G = Generator()
    D = Discriminator()
    # G = torch.nn.DataParallel(G, device_ids=[0,1])
    # D = torch.nn.DataParallel(D, device_ids=[0,1])

    # z = Variable(torch.rand(16,512,4,4,4))
    # m = nn.ConvTranspose3d(512, 256, 4, 2, 1)
    z = Variable(torch.rand(16, 200, 1,1,1))
    X = G(z)
    m = nn.Conv3d(1, 64, 4, 2, 1)
    D_X = D(X)
    print(X.shape, D_X.shape)
