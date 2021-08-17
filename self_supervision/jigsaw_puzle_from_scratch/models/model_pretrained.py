import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

import sys

sys.path.append('Utils')


class Model(pl.LightningModule):

    def __init__(self, classes=1000):
        super(Model, self).__init__()

        res50_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(res50_model.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(18432, 4096))
        self.classifier.add_module('relu1', nn.ReLU(inplace=True))
        self.classifier.add_module('drop1', nn.Dropout(p=0.05))
        self.classifier.add_module('fc2', nn.Linear(4096, 2048))
        self.classifier.add_module('relu2', nn.ReLU(inplace=True))
        self.classifier.add_module('fc3', nn.Linear(2048, classes))

    def save(self, checkpoint):
        res50_model = models.resnet50(pretrained=True)
        pretrained_last_layers = nn.Sequential(*list(res50_model.children())[-1:])
        res50_model_finetuned = nn.Sequential([self.feature_extractor, pretrained_last_layers])
        torch.save(res50_model_finetuned.state_dict(), checkpoint)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.feature_extractor(x[i])
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(B, -1)
        x = self.classifier(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
