import torch
import  torch.nn as nn
from torch.autograd import Variable


class DGAdaIN(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super(DGAdaIN, self).__init__()

        self.affine_scale = nn.Linear(in_channels, out_channels, bias=True)
        self.affine_bias = nn.Linear(in_channels, out_channels, bias=True)
        self.norm = nn.InstanceNorm1d(in_channels, affine = False, momentum=0.9,  track_running_stats=False)

    def forward(self, x, w):
        y_scale = 1 + self.affine_scale(w)
        y_bias = 0 + self.affine_bias(w)
        x = self.norm(x)
        x_scale = (x * y_scale) + y_bias

        return x_scale


if __name__=='__main__':
    video = Variable(torch.randn(6*16, 2048)).cuda()
    video_depth = Variable(torch.randn(6*16, 2048)).cuda()

    video = video.view(6, 16, 2048)
    video_depth = video_depth.view(6,16,2048)
    mymodel = DGAdaIN()
    mymodel = torch.nn.DataParallel(mymodel)
    mymodel.cuda().eval()

    fea = mymodel(video, video_depth)
    print(fea.size())


