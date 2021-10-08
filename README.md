# zjyzjy
homework
---
First initialize the transform to perform some transformations on the images downloaded from cifar. Use the package that comes with datasets.CIFAR10 to download the CIFAR data. 

`trans = transforms.Compose((transforms.Resize(32),transforms.ToTensor()))`

`cifar_train = datasets.CIFAR10('cifar',train = True,transform=trans,download=False)`

---

Here we consider the classic deep residual network to construct a neural network. In each resblock, each input x is convolved to obtain x_pro through two layers. Since the result of the residual learning unit is x_pro+x, in order to ensure x_pro and x The number of channels is the same, we need ch_trans to change the number of channels of x to x_ch, and then output the result of this block 
```
    def forward(self, x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))

        # short_cut(Ensure that the sizes of the two are exactly the same ):
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        return out
```

---

Then build resnet on the basis of blocks, and perform 4 blocks after a layer of convolution, 
```
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
```

Since the computer does not have a cuda environment, here is torch.device('cpu'). The optimization method is adam. Due to the limitation of the CPU speed, set 5 epochs, then calculate the gradient and update parameters of the backpropagation, and print the batch and Iterative loss 
```
device = torch.device('cpu')
net = resnet()
net = net.to(device)
loss_fn = nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3) 
```

Finally, make a prediction. Because the batch_size is set to 30 and there are 10000 sheets in the test, there are a total of 334 batches. 

---

