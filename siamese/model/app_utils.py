"""
Some useful functions for the Streamlit app.
"""
import streamlit as st

import torch
import torchvision.transforms as transforms

from ..dataset.dataset_alphabet import OmniglotAlphabet
from ..model.network import SiameseNetwork

import os
import gdown
from tqdm import tqdm

from typing import Union, Any, List

@st.cache
def load_model(model_path: str):
    model_url = 'https://drive.google.com/uc?id=1abM973VRugP2xB_QCPgiYkBrYJReefDy'
    
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model

@st.cache
def characters_database(alphabet: str, subset: bool = True):
    omniglot_alphabet = OmniglotAlphabet(
        alphabet=alphabet,
        root='siamese/data'
    )

    if subset:
        characters = [x[0] for (i, x) in enumerate(omniglot_alphabet) if i % 20 == 0]
    else:
        characters = [x[0] for x in iter(omniglot_alphabet)]

    return characters

def get_predictions(model: Any, image: Any, characters: List, k: int = 3):
    transform = transforms.ToTensor()
    # Convert to tensor and add batch dimension
    image = transform(image).unsqueeze(dim=0)
    
    y_probs = []
    for character in tqdm(characters):
        # Compute logits
        y_lgts = model(image, transform(character).unsqueeze(dim=0))
        
        # Compute scores
        y_prob = torch.sigmoid(y_lgts)

        y_probs.append(y_prob)

    topk_prob, topk_index = torch.topk(torch.tensor(y_probs), k=3)

    return topk_prob, topk_index
