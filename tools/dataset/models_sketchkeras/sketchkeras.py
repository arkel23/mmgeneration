import torch
import torch.nn as nn


class SketchKeras(nn.Module):
    def __init__(self):
        super(SketchKeras, self).__init__()

        self.downblock_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_6 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_1 = nn.Sequential(
            nn.Upsample((64, 64)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(1024, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_2 = nn.Sequential(
            nn.Upsample((128, 128)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(512, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_3 = nn.Sequential(
            nn.Upsample((256, 256)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_4 = nn.Sequential(
            nn.Upsample((512, 512)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.last_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.last_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1)

    def forward(self, x):
        d1 = self.downblock_1(x)
        d2 = self.downblock_2(d1)
        d3 = self.downblock_3(d2)
        d4 = self.downblock_4(d3)
        d5 = self.downblock_5(d4)
        d6 = self.downblock_6(d5)

        u1 = torch.cat((d5, d6), dim=1)
        u1 = self.upblock_1(u1)
        u2 = torch.cat((d4, u1), dim=1)
        u2 = self.upblock_2(u2)
        u3 = torch.cat((d3, u2), dim=1)
        u3 = self.upblock_3(u3)
        u4 = torch.cat((d2, u3), dim=1)
        u4 = self.upblock_4(u4)
        u5 = torch.cat((d1, u4), dim=1)

        out = self.last_conv(self.last_pad(u5))

        return out

