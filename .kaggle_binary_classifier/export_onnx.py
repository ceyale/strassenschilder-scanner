from pathlib import Path

import torch
from torch import nn


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinySignBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, 2, 1, bias=False), nn.BatchNorm2d(12), nn.ReLU6(inplace=True),
            DSBlock(12, 16), DSBlock(16, 24, 2), DSBlock(24, 32, 2), DSBlock(32, 40, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(40, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


root = Path(__file__).resolve().parent
checkpoint = torch.load(root / 'output/sign-binary-cnn-v2-output/best.pt', map_location='cpu', weights_only=False)
model = TinySignBinary().eval()
model.load_state_dict(checkpoint['state_dict'])
output = root.parent / 'models' / 'sign-binary-32.onnx'
torch.onnx.export(
    model, torch.zeros(1, 3, 32, 32), output,
    input_names=['input'], output_names=['logits'], opset_version=12,
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, dynamo=False,
)
print(f'Exported {output}; bytes={output.stat().st_size}; epoch={checkpoint["epoch"]}; val_acc={checkpoint["val_accuracy"]:.6f}')
