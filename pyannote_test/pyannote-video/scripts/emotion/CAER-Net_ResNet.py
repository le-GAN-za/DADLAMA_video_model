import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch

resnet1 = models.resnet101(pretrained=True)
resnet2 = models.resnet101(pretrained=True)

class CAERNet(nn.Module):
	def __init__(self):
		super(CAERNet, self).__init__()
		self.features_f = nn.Sequential(
			# stop at conv4
			*list(resnet1.children())[:-3]
		)

		self.features_c = nn.Sequential(
			*list(resnet2.children())[:-3]
		)

		self.attention_net = nn.Sequential(
			nn.Conv2d(1024, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 1, kernel_size=3, padding=1),
		)

		self.fusion_f = nn.Sequential(
			nn.Conv2d(1024, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 1, kernel_size=1, padding=0)
		)

		self.fusion_c = nn.Sequential(
			nn.Conv2d(1024, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 1, kernel_size=1, padding=0)
		)

		self.classifier = nn.Sequential(
			nn.Conv2d(2048, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(128, 7, kernel_size=1, padding=0)
			)

	def forward(self, v_f, v_c, is_training=True):
		x_f = self.features_f(v_f)
		x_c = self.features_c(v_c)
		
		att = self.attention_net(x_c)
		att = F.softmax(att.view(att.size(0), att.size(1), -1), dim=-1)
		att = att.view(att.size(0), att.size(1), 7, 7)
		x_c = x_c * att

		x_f = F.avg_pool2d(x_f, (6,6))
		x_c = F.avg_pool2d(x_c, (7,7))

		lambda_f = self.fusion_f(x_f)
		lambda_c = self.fusion_c(x_c)


		lambdas = F.softmax(torch.cat([lambda_f, lambda_c], dim=1), dim=1)
		lambda_f, lambda_c = torch.split(lambdas, 1, dim=1)
		
		x_f = x_f * lambda_f
		x_c = x_c * lambda_c
		x_a = torch.cat([x_f, x_c], dim=1)
		
		y = self.classifier(x_a)
		
		return y


if __name__ == "__main__":
	x_f = torch.autograd.Variable(torch.zeros(2, 3, 96, 96)).cuda()
	x_c = torch.autograd.Variable(torch.zeros(2, 3, 112, 112)).cuda()
	net = CAERNet().cuda()
	y = net(x_f, x_c)
	print(y.size())
