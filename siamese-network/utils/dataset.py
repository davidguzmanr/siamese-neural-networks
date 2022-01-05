import torch
from torchvision.datasets import Omniglot
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from typing import Optional, Callable, Tuple, Any

np.random.seed(42)

class OmniglotPairs(Dataset):
    """
    Pairs of images to train a siamese neural network from the Omniglot dataset.
    
    References
    - https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.Omniglot
    - https://github.com/brendenlake/omniglot
    """
    _repr_indent = 4

    def __init__(
        self,
        root: str = 'data/',
        n_pairs: int = 1_000_000, 
        train: bool = True,
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__()

        self.root = root
        self.n_pairs = n_pairs
        self.train = train
        self.transform = transform

        self.omniglot = self.omniglot_dataset()
        self.idx_to_class = pd.DataFrame(
            data=[(i, x[1]) for (i, x) in enumerate(self.omniglot)],
            columns=['id', 'character']
        )
        self.n_characters = self.idx_to_class['character'].unique().shape[0]

    def omniglot_dataset(self):
        return Omniglot(
            root=self.root, 
            download=True,
            background=self.train,
            transform=self.transform
        )

    def __len__(self) -> int:
        return self.n_pairs

    def __repr__(self) -> str:
        train = 'train' if self.train else 'validation'
        head = f'Dataset: {self.__class__.__name__} ({train})'
        body = [f'Number of pairs: {self.__len__()}']
        
        if self.root is not None:
            body.append(f'Root location: {self.root}')
        
        if self.transform:
            body += [f'Transform: {repr(self.transform)}']

        lines = [head] + [f'{" " * self._repr_indent} {line}' for line in body]
        representation = '\n'.join(lines)

        return representation
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # Images from same character
        if index % 2 == 0:
            # Get a random character_id
            character = np.random.randint(low=0, high=self.n_characters)

            # Get two different ids from the same character
            idx = self.idx_to_class[self.idx_to_class['character'] == character]
            id_1, id_2 = idx.sample(n=2, replace=False)['id'].values
            image_1, image_2 = self.omniglot.__getitem__(id_1)[0], self.omniglot.__getitem__(id_2)[0]
            label = 1
        
        # Images from different characters
        else:
            id_1, id_2 = 0, 0
            while id_1 == id_2:
                id_1, id_2 = self.idx_to_class.sample(n=2, replace=False)['id'].values
            
            image_1, image_2 = self.omniglot.__getitem__(id_1)[0], self.omniglot.__getitem__(id_2)[0]
            label = 0

        return (image_1, image_2, label)