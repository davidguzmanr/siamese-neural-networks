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

