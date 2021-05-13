import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from model_DGAdaIn import DGAdaIN



class model_resnet50(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet50,self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        return feature



class model_AMeFu(nn.Module):
    def __init__(self, num_classes):
        super(model_AMeFu,self).__init__()
        self.submodel_rgb = model_resnet50(num_classes)
        self.submodel_depth = model_resnet50(num_classes)

        self.fea_dim = 2048

        self.instance_norm = DGAdaIN(in_channels=self.fea_dim, out_channels=self.fea_dim)
        self.fc = nn.Linear(self.fea_dim, num_classes)


    def forward(self, x,  L2_fusion=True):
        x_rgb = x['rgb']
        x_depth = x['depth']
        rgb_fea = self.submodel_rgb(x_rgb)
        depth_fea = self.submodel_depth(x_depth)

        rgb_fea = rgb_fea.view(-1, 16, self.fea_dim)
        depth_fea = depth_fea.view(-1, 16, self.fea_dim)
        fusion_fea = self.instance_norm(rgb_fea, depth_fea)
        fusion_fea = torch.mean(fusion_fea,1)

        if(L2_fusion):
            feature = torch.nn.functional.normalize(fusion_fea, p=2, dim=1)

        output = self.fc(feature)
        return feature, output


if __name__=='__main__':
    # uasge
    print('---------------testing model_fusion----------------')
    video = Variable(torch.randn(3*16, 3, 224, 224)).cuda()
    video_depth = Variable(torch.randn(3*16, 3, 224, 224)).cuda()

    x={}
    x['rgb']= video
    x['depth']= video_depth
    
    mymodel = model_AMeFu(num_classes=64)
    mymodel = mymodel.cuda().eval()
   
    fea, out = mymodel(x)
    print(fea.size(), out.size())

