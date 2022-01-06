from torchvision.datasets import Omniglot
from torch.utils.data import Dataset

import numpy as np

from PIL import Image

from typing import Optional, Callable, Tuple, Any, List

from os import listdir
from os.path import isfile, join, expanduser

np.random.seed(42)

class OmniglotAlphabet(Dataset):
    """
    Dataset of an specific alphabet (e.g., Latin, Greek, etc) from the Omniglot dataset.

    Parameters
    ----------
    alphabet: str.
        One of the alphabet from the Omniglot dataset.

    root: str, default='data/'.
        Directory where the dataset will be downloaded.

    transform: (callable, optional), defaul=None.
        A function/transform.

    References
    ----------
    - https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.Omniglot
    - https://github.com/brendenlake/omniglot
    """
    _repr_indent = 4
    target_folder = 'omniglot-py/images_background'

    def __init__(
        self,
        alphabet: str,
        root: str = 'data/',
        transform: Optional[Callable] = None
    ) -> None:

        super().__init__()

        self.root = root
        self.alphabet = alphabet
        self.alphabet_path = join(self.root, self.target_folder, self.alphabet)
        self.transform = transform
        self.download()

        self._characters = sorted(listdir(self.alphabet_path))
        self._character_images = [
            [(image, idx) for image in self.list_files(join(self.alphabet_path, character), '.png', True)]
            for idx, character in enumerate(self._characters)
        ]
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

    def download(self) -> None:
        """
        Download Omniglot dataset.
        """
        Omniglot(root=self.root, download=True)
    
    @staticmethod
    def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
        """
        List all files ending with a suffix at a given root.
        
        Parameters
        ----------
        root: str. 
            Path to directory whose folders need to be listed.
        
        suffix: str or tuple. 
            Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        
        prefix: bool, default=False. 
            If true, prepends the path to each result, otherwise only returns 
            the name of the files found.
        """
        root = expanduser(root)
        files = [f for f in listdir(root) if isfile(join(root, f)) and f.endswith(suffix)]
        if prefix is True:
            files = [join(root, f) for f in files]

        return files

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __repr__(self) -> str:
        head = f'Dataset: {self.__class__.__name__} ({self.alphabet})'
        body = [f'Number of pairs: {self.__len__()}']
        
        if self.root is not None:
            body.append(f'Root location: {self.root}')
        
        if self.transform:
            body += [f'Transform: {repr(self.transform)}']

        lines = [head] + [f'{" " * self._repr_indent} {line}' for line in body]
        representation = '\n'.join(lines)

        return representation
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, character_class = self._flat_character_images[index]
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        return (image, character_class)