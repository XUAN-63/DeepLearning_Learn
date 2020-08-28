import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 图像的读入和预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([96, 96]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder("E:/Data/", transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         drop_last=True)


class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super().__init__()

        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz,
                               ngf * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False), nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True))

        # layer2输出尺寸(ngf*4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(inplace=True))

        # layer3 输出尺寸(ngf*2)*16*16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(inplace=True))

        # layer4 输出尺寸ngf *32 *32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(inplace=True))

        # layer5 输出尺寸3 * 96 *96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False), nn.Tanh())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


class NetD(nn.Module):
    def __init__(self, ndf):
        super().__init__()

        # layer1 输入3*96*96,输出ndf *32

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, inplace=True))

        # layer2, 输出(ndf*2)*16*16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True))

        # layers3, 输出(ndf*4)*8 *8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True))

        # layer4, 输出(ndf*8)*4*4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))

        # layer5 输出一个概率
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


netG = NetG(64, 100).to(device)
netD = NetD(64).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

label = torch.FloatTensor(64)
real_label = 1
fake_label = 0

for epoch in range(1, 25 + 1):
    for i, (imgs, _) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        ## 让D尽可能的把真图片判别为1
        imgs = imgs.to(device)
        output = netD(imgs)
        label.data.fill_(real_label)
        label = label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()
        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        noise = torch.randn(64, 100, 1, 1)
        noise = noise.to(device)
        fake = netG(noise)  # 生成假图

        output = netD(fake.detach())  #避免梯度传到G，因为G不用更新
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f' %
              (epoch, 25, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % ('imgs', epoch),
                      normalize=True)
torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))

print("debug")