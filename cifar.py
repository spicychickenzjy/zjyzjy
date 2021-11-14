import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

trans = transforms.Compose((transforms.Resize(32),transforms.ToTensor()))


cifar_train = datasets.CIFAR10('cifar',train = True,transform=trans,download=False)
cifar_train_batch = DataLoader(cifar_train,batch_size=30,shuffle=True)

cifar_test = datasets.CIFAR10('cifar',train = False,transform=trans)
cifar_test_batch = DataLoader(cifar_test,batch_size=30,shuffle=True)


class resblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(resblock, self).__init__()
        self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)
        self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(ch_out)
        self.ch_in, self.ch_out, self.stride = ch_in, ch_out, stride
        self.ch_trans = nn.Sequential()
        if ch_in != ch_out:
            self.ch_trans = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(self.ch_out))
        # ch_trans表示通道数转变。因为要做short_cut,所以x_pro和x_ch的size应该完全一致

    def forward(self, x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))

        # short_cut:
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        return out


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64))
        self.block1 = resblock(64, 128, 2)  # 长宽减半 32/2=16
        self.block2 = resblock(128, 256, 2)  # 长宽再减半 16/2=8
        self.block3 = resblock(256, 512, 1)
        self.block4 = resblock(512, 512, 1)
        self.outlayer = nn.Linear(512, 10)  # 512*8*8=32768

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.reshape(x.size(0), -1)
        result = self.outlayer(x)
        return result
device = torch.device('cpu')
net = resnet()
net = net.to(device)#将网络部署到GPU上
loss_fn = nn.CrossEntropyLoss().to(device) #选好loss_function
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3) #选好优化方式

# 开始训练
for epoch in range(5):  # 只是简单跑一下，所以只设置了5个epoch
    for batchidx, (x, label) in enumerate(cifar_train_batch):
        x, label = x.to(device), label.to(device)  # x.size (bcs,3,32,32) label.size (bcs)
        logits = net.forward(x)
        loss = loss_fn(logits, label)  # logits.size:bcs*10,label.size:bcs

        # 开始反向传播：
        optimizer.zero_grad()
        loss.backward()  # 计算gradient
        optimizer.step()  # 更新参数
        if (batchidx + 1) % 400 == 0:
            print('这是本次迭代的第{}个batch'.format(batchidx + 1))  # 本例中一共有50000张照片，每个batch有30张照片，所以一个epoch有1667个batch

    print('这是第{}迭代，loss是{}'.format(epoch + 1, loss.item()))

net.eval()
with torch.no_grad():  # 这两行代码都可以不写，但是为了安全一点，还是写上为好
    correct_num = 0  # 预测正确的个数
    total_num = 0  # 总的照片张数
    batch_num = 0  # 第几个batch
    for x, label in cifar_test_batch:  # x的size是30*3*32*32（30是batch_size,3是通道数），label的size是30.
        # cifar_test中一共有10000张照片，所以一共有334个batch，因此要循环334次
        x, label = x.to(device), label.to(device)
        logits = net.forward(x)
        pred = logits.argmax(dim=1)
        correct_num += torch.eq(pred, label).float().sum().item()
        total_num += x.size(0)
        batch_num += 1
        if batch_num % 50 == 0:
            print('这是第{}个batch'.format(batch_num))  # 一共有10000/30≈334个batch

    acc = correct_num / total_num  # 最终的total_num是10000
    print('测试集上的准确率为：', acc)
