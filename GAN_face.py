import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from torchvision.utils import save_image
from dataset import trainDataset
import os


d_save_path = r"model\D_net.pth"
g_save_path = r"model\G_net.pth"
class ChunkSampler(sampler.Sampler): # 定义一个取样的函数
    """从某个偏移量对元素进行顺序采样。
        参数:
        num_samples:所需数据点的
        start:偏移我们应该开始选择的地方
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NOISE_DIM = 128
batch_size = 128

class build_dc_classifier(nn.Module):
    def __init__(self):
        super(build_dc_classifier, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(3, 32, 5, 1),  # 92*92
            # nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2, 2),  # 46*46
            # nn.Conv2d(32, 64, 5, 1),  # 42*42
            # nn.LeakyReLU(0.2),
            # nn.MaxPool2d(3, 2),  # 20*20
            # nn.Conv2d(64, 32, 5, 1),  # 16*16
            # nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2, 2),  # 8*8
            # nn.Conv2d(32, 64, 5, 1),  #8x8
            # nn.LeakyReLU(0.2),
            nn.Conv2d(3, 32, 4, 2, 1),  # 48*48
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),  # 24*24
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 4, 2, 1),  # 12*12
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 2, 2),  # 6x6
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 64, 2, 2, 1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class build_dc_generator(nn.Module):
    def __init__(self, noise_dim=128):
        super(build_dc_generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 6 * 6 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(6 * 6 * 128),

        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),   #14*14
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 4 , 2 ,1),   #28*28
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 48*48
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, 4, 2, padding=1),    #96*96
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 6, 6)  # reshape 通道是 128，大小是 6x6
        x = self.conv(x)
        return x


bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    false_labels = torch.zeros(size, 1).float().cuda()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss

def generator_loss(logits_fake): # 生成器的 loss
    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    loss = bce_loss(logits_fake, true_labels)
    return loss

# 使用 adam 来进行训练，学习率是 2e-4, beta1 是 0.5, beta2 是 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer


def train_dc_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss):
    iter_count = 0
    train_path = r"E:\faces"
    traindata = trainDataset(train_path)
    train = DataLoader(traindata, batch_size=130, shuffle=True)
    for epoch in range(50):
        for x in train:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x).cuda()  # 真实数据
            logits_real = D_net(real_data)  # 判别网络得分

            sample_noise = torch.randn(bs, 128)   # -1 ~ 1 的均匀分布
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 优化生成网络

            if (iter_count % 100 == 0):
                print('rount:{},Iter: {}/{}, D_loss: {:.4}, G_loss:{:.4}'.format(epoch,iter_count,len(train),d_total_error.item(), g_error.item()))
                save_image(fake_images,"./image/{}-fake_image.jpg".format(iter_count),nrow=10,normalize=True,scale_each=True)
                save_image(x,"./image/{}-real_image.jpg".format(iter_count),nrow=10,normalize=True,scale_each=True)
                torch.save(D_net.state_dict(),d_save_path)
                torch.save(G_net.state_dict(),g_save_path)
            iter_count += 1

D_DC = build_dc_classifier().cuda()
G_DC = build_dc_generator().cuda()

if os.path.exists(d_save_path):
    D_DC.load_state_dict(torch.load(d_save_path))

if os.path.exists(g_save_path):
    G_DC.load_state_dict(torch.load(g_save_path))

D_DC_optim = get_optimizer(D_DC)
G_DC_optim = get_optimizer(G_DC)

train_dc_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, discriminator_loss, generator_loss)