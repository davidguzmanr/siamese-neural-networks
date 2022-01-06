from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from typing import Optional, Callable, Tuple, Any

np.random.seed(42)

class OmniglotPairs(Dataset):
    """
    Pairs of images to train a siamese neural network from the Omniglot dataset.

    Parameters
    ----------
    dataset: torch dataset.
        PyTorch dataset to get pairs of images.

    n_pairs: int, default=100,000.
        Number of pairs to generate.

    transform: (callable, optional), defaul=None.
        A function/transform.
    
    References
    ----------
    - https://github.com/brendenlake/omniglot
    - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    _repr_indent = 4

    def __init__(
        self,
        dataset: Dataset,
        n_pairs: int = 100_000, 
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__()

        self.dataset = dataset
        self.n_pairs = n_pairs
        self.transform = transform
        
        self.idx_to_class = pd.DataFrame(
            data=[(i, x[1]) for (i, x) in enumerate(self.dataset)],
            columns=['id', 'character']
        )
        self.n_characters = self.idx_to_class['character'].unique().shape[0]

    def __len__(self) -> int:
        return self.n_pairs

    def __repr__(self) -> str:
        head = f'Dataset: {self.__class__.__name__}'
        body = [f'Number of pairs: {self.__len__()}']
        
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
            image_1, image_2 = self.dataset.__getitem__(id_1)[0], self.dataset.__getitem__(id_2)[0]
            label = 1
        
        # Images from different characters
        else:
            id_1, id_2 = 0, 0
            while id_1 == id_2:
                id_1, id_2 = self.idx_to_class.sample(n=2, replace=False)['id'].values
            
            image_1, image_2 = self.dataset.__getitem__(id_1)[0], self.dataset.__getitem__(id_2)[0]
            label = 0

        return (image_1, image_2, label)