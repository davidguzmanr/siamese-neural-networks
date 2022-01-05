import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from torchvision import transforms

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

from utils.dataset import OmniglotPairs
from utils.network import SiameseNetwork

from typing import Any

class SiameseModel(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 64
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = SiameseNetwork()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        logits = self.forward(x1, x2)
        preds = torch.cat((1 - logits.sigmoid(), logits.sigmoid()), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(dim=1).float())

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc(preds, y), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        logits = self.forward(x1, x2)
        preds = torch.cat((1 - logits.sigmoid(), logits.sigmoid()), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(dim=1).float())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(preds, y), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return [optimizer], [scheduler]

    @property
    def transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.9221), (0.2623))
        ])

    def prepare_data(self) -> None:
        # Download the data, in case it hasn't been downloaded
        OmniglotPairs()

    def train_dataloader(self):
        train_dataset = OmniglotPairs(
            n_pairs=1_000_000,
            train=True,
            transform=self.transform
        )
        return DataLoader(
            dataset=train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=8
        )

    def val_dataloader(self):
        validation_dataset = OmniglotPairs(
            n_pairs=100_000,
            train=False,
            transform=self.transform
        )
        return DataLoader(
            dataset=validation_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=8
        )

    def test_dataloader(self):
        pass

def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = LightningCLI(SiameseModel, seed_everything_default=42, save_config_overwrite=True, run=False)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)


if __name__ == '__main__':
    cli_main()