"""model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init



class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
       
        if no_dec:
            return z.squeeze()
        else:
            
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

class Extractor(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10):
        super(Extractor, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, z_dim, 1)
        )
        

    def forward(self, x, no_dec=False):
       
        z = self.encode(x).squeeze(2).squeeze(2)
       
        return z







class FactorVAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dim=10):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        #eps1 = std.data.new(std.size(0),std.size(1)-1,std.size(2),std.size(3)).normal_()
        #eps2=std.data.new(std.size(0),1,std.size(2),std.size(3)).uniform_()
        #eps=torch.cat([eps1,eps2],1)
        eps = std.data.new(std.size()).normal_()
        #import pdb;pdb.set_trace()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        #import pdb;pdb.set_trace()
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE3(nn.Module):
    """Encoder and Decoder architecture for 3D Faces data."""
    def __init__(self, z_dim=10):
        super(FactorVAE3, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


class classifier(nn.Module):
    def __init__(self, z_dim=5,num_classes=2):
        super(classifier,self).__init__()
        self.fc=nn.Linear(z_dim,1000)
        self.Leacky_relu=nn.LeakyReLU(0.2,inplace=True)
        self.fc2=nn.Linear(1000,1000)
        self.Leacky_relu2=nn.LeakyReLU(0.2,inplace=True)
        self.fc3=nn.Linear(1000,1000)
        self.Leacky_relu3=nn.LeakyReLU(0.2,inplace=True)
        self.fc4=nn.Linear(1000,1000)
        self.Leacky_relu4=nn.LeakyReLU(0.2,inplace=True)
        self.fc5=nn.Linear(1000,1000)
        self.Leacky_relu5=nn.LeakyReLU(0.2,inplace=True)
        self.fc6=nn.Linear(1000,num_classes)
    def forward(self,x):
        x=self.Leacky_relu(self.fc(x))
        x=self.Leacky_relu2(self.fc2(x))
        x=self.Leacky_relu3(self.fc3(x))
        x=self.Leacky_relu4(self.fc4(x))
        x=self.Leacky_relu5(self.fc5(x))
        x=self.fc6(x)
       
       
        return x


class paclassifier(nn.Module):
    def __init__(self, z_dim=1,num_classes=2):
        super(paclassifier,self).__init__()
        self.fc=nn.Linear(z_dim,num_classes)
        self.sigmoid=nn.Sigmoid()
       
    def forward(self,x):
        x=self.sigmoid(self.fc(x))
        #x=self.sigmoid(x)
       
       
        return x





def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
