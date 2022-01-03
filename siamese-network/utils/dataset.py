from torchvision.datasets import Omniglot
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from typing import Any, Optional, Callable, Tuple

np.random.seed(42)

class OmniglotPairs(Dataset):
    """
    Pairs of images to train a siamese neural network from the Omniglot dataset.
    
    References
    - https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.Omniglot
    - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    _repr_indent = 4

    def __init__(
        self,
         n_pairs: int = 1_000_000, 
         transform: Optional[Callable] = None
    ) -> None:

        super(OmniglotPairs, self).__init__()

        self.omniglot = Omniglot(
            root='data/', 
            background=True,
            download=True,
            transform=transform
        )
        self.n_pairs = n_pairs
        self.idx_to_class = pd.DataFrame(
            data=[(i, x[1]) for (i, x) in enumerate(self.omniglot)],
            columns=['id', 'character']
        )
        self.n_characters = self.idx_to_class['character'].unique().shape[0]

    def __len__(self) -> int:
        return self.n_pairs

    def __repr__(self) -> str:
        head = f'Dataset: {self.__class__.__name__}'
        body = [f'Number of pairs: {self.__len__()}']
        lines = [head] + [f'{" " * self._repr_indent} {line}' for line in body]
        lines = '\n'.join(lines)

        return lines
        
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

