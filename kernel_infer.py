from glob import glob
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import math
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os

kaggle_path = '../input/freesound-audio-tagging-2019/test/*wav'
model_path = '../input/<YOUR-DATASET>/model.pth'


LABELS = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
          'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
          'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink',
          'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd',
          'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close',
          'Drip', 'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
          'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss',
          'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone',
          'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing',
          'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard',
          'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
          'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet',
          'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']


class KFSDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.n_labels = len(LABELS)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data[item]
        wav, sr = torchaudio.load(path)
        logmel = self.transform(wav)
        return dict(
            logmel=logmel,
            fname=os.path.basename(path),
        )
        

# Your model code
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SingleChannelResnet(nn.Module):
    def __init__(self, block, layers, num_classes=80):
        self.inplanes = 64
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def compute_loss(self, entry, device=None):
        logmel = entry['logmel'].to(device)
        labels = entry['labels'].to(device)
        logits = self.forward(logmel)
        probs = torch.sigmoid(logits).cpu().data.numpy()
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        # todo: add storable metrics
        return dict(
            loss=bce.item(),
            probs=probs,
        )


def resnet34(**kwargs):
    model = SingleChannelResnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


device = torch.device("cuda")
ckpt = torch.load(model_path)
model = resnet34()
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

test_files = sorted(glob(kaggle_path))

mean_db, std_db, low_db, high_db = 1.0, 2.5, -10.0, 120.0

basic_transform = [
    torchaudio.transforms.MelSpectrogram(sr=44100, n_fft=1764, hop=220, n_mels=64),
    torchaudio.transforms.SpectrogramToDB(),
    lambda _: torch.clamp(_, low_db, high_db),
    lambda _: (_ - mean_db) / std_db,
]

ds = KFSDataset(test_files, transforms.Compose(basic_transform))
dataloader = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False)

predictions = []
for entry in tqdm(dataloader):
    x = entry['logmel'].to(device)
    p = torch.sigmoid(model(x)).detach().cpu().numpy()
    predictions.append((entry['fname'][0], p))

fnames, predictions = list(zip(*predictions))
df = pd.DataFrame(data=np.concatenate(predictions),
                  index=fnames,
                  columns=LABELS)
df['fname'] = fnames
df = df[['fname'] + LABELS]
df.to_csv('./submission.csv', index=False)