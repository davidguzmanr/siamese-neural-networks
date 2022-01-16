import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import Omniglot

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

from dataset.dataset_pairs import OmniglotPairs
from model.network import SiameseNetwork

from typing import Any

class SiameseModel(LightningModule):
    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        num_workers: int = 4
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = SiameseNetwork()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # Original dataset
        self.omniglot_background = Omniglot(
            root='data',
            transform=self.transform,
            background=True,
            download=True 
        )

        self.omniglot_evaluation = Omniglot(
            root='data',
            transform=transforms.ToTensor(),
            background=False,
            download=True 
        )

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
        x1, x2, y = batch
        logits = self.forward(x1, x2)
        preds = torch.cat((1 - logits.sigmoid(), logits.sigmoid()), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(dim=1).float())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc(preds, y), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [scheduler]

        return optimizer

    @property
    def transform(self):
        return transforms.ToTensor()

    # def prepare_data(self) -> None:
        # Download the data, in case it hasn't been downloaded
        # OmniglotPairs()

    def train_dataloader(self):
        train_dataset = OmniglotPairs(
            dataset=self.omniglot_background,
            n_pairs=200_000
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers
        )

        return train_loader

    def val_dataloader(self):
        evaluation_dataset = OmniglotPairs(
            dataset=self.omniglot_evaluation,
            n_pairs=20_000
        )

        validation_dataset, _ = random_split(
            dataset=evaluation_dataset,
            lengths=[10_000, 10_000],
            generator=torch.Generator().manual_seed(42)
        )

        validation_loader = DataLoader(
            validation_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers
        )

        return validation_loader

    def test_dataloader(self):
        evaluation_dataset = OmniglotPairs(
            dataset=self.omniglot_evaluation,
            n_pairs=20_000
        )

        _, test_dataset = random_split(
            dataset=evaluation_dataset,
            lengths=[10_000, 10_000],
            generator=torch.Generator().manual_seed(42)
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers
        )

        return test_loader

def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = LightningCLI(SiameseModel, seed_everything_default=42, save_config_overwrite=True, run=False)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)

if __name__ == '__main__':
    cli_main()