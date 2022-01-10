"""
Some useful functions for the Streamlit app.
"""
import streamlit as st

import torch

from ..dataset.dataset_pairs import OmniglotPairs
from ..model.network import SiameseNetwork

import os
import gdown


@st.cache
def load_model():
    model_path = 'siamese/model/siamese-network.pt'
    model_url = 'https://drive.google.com/uc?id=1abM973VRugP2xB_QCPgiYkBrYJReefDy'
    
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model

@st.cache
def characters_database(alphabet: str):
    pass
