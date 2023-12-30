import torch
import torch.nn as nn
from .stripe import *

__all__ = ['VGG']
default_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = default_cfg['VGG16']
        self.features = self._make_layers(cfg)
        
        # Determine the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            conv_output = self.features(dummy_input)
            output_size = conv_output.view(conv_output.size(0), -1).size(1)
        
        # Adjust the input size of the first linear layer
        self.classifier = Linear(output_size, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [FilterStripe(in_channels, x),
                           BatchNorm(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def update_skeleton(self, sr, threshold):
        for key, m in self.named_modules():
            if isinstance(m, FilterStripe):
                out_mask = m.update_skeleton(sr, threshold)
            elif isinstance(m, BatchNorm):
                m.update_mask(out_mask)

    def prune(self, threshold):
        in_mask = torch.ones(3, dtype=torch.bool)
        for m in self.modules():
            if isinstance(m, FilterStripe):
                m.prune_in(in_mask)
                in_mask = m.prune_out(threshold)
                m._break(threshold)
            if isinstance(m, BatchNorm):
                m.prune(in_mask)
            if isinstance(m, Linear):
                if in_mask.size(0) == m.weight.size(1):  # check if sizes match
                    m.prune_in(in_mask)
                else:
                    print("The size of the mask does not match the size of the weight tensor.")
                    return
                in_mask = torch.ones(m.out_features, dtype=torch.bool)