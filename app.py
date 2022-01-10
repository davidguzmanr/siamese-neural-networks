
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from siamese.model.network import SiameseNetwork
from siamese.dataset.dataset_alphabet import OmniglotAlphabet
from siamese.model.app_utils import load_model

import json
import os
import gdown
from tqdm import tqdm

ALPHABETS = [
    'Alphabet_of_the_Magi',
    'Anglo-Saxon_Futhorc',
    'Arcadian',
    'Armenian',
    'Asomtavruli_(Georgian)',
    'Balinese',
    'Bengali',
    'Blackfoot_(Canadian_Aboriginal_Syllabics)',
    'Braille',
    'Burmese_(Myanmar)',
    'Cyrillic',
    'Early_Aramaic',
    'Futurama',
    'Grantha',
    'Greek',
    'Gujarati',
    'Hebrew',
    'Inuktitut_(Canadian_Aboriginal_Syllabics)',
    'Japanese_(hiragana)',
    'Japanese_(katakana)',
    'Korean',
    'Latin',
    'Malay_(Jawi_-_Arabic)',
    'Mkhedruli_(Georgian)',
    'N_Ko',
    'Ojibwe_(Canadian_Aboriginal_Syllabics)',
    'Sanskrit',
    'Syriac_(Estrangelo)',
    'Tagalog',
    'Tifinagh'
]

# Specify canvas parameters in application
stroke_width = st.sidebar.slider(
    label='Stroke width:',
    min_value=1, 
    max_value=25, 
    value=3
)
drawing_mode = st.sidebar.selectbox(
    label='Drawing tool:', 
    options=('freedraw', 'line', 'rect', 'circle', 'transform')
)
realtime_update = st.sidebar.checkbox(
    label='Update in realtime', 
    value=True
)

alphabet = st.sidebar.selectbox(
    label='Alphabet',
    options=ALPHABETS
)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='black',
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    key='canvas',
    display_toolbar=True
)



model = load_model()
transform = transforms.ToTensor()
omniglot_alphabet = OmniglotAlphabet(
    alphabet=alphabet,
    root='siamese/data'
)

if canvas_result.image_data is not None:
    image = canvas_result.image_data

    # Convert RGBA image to grayscale (PIL doesn't convert as I want)
    image = np.uint8(255 - image[:, :, 3])
    image = Image.fromarray(image, mode='L').resize(size=(105, 105))
    # Convert to tensor and add batch dimension
    image = transform(image).unsqueeze(dim=0)

    y_probs = []
    for i, (character, _) in tqdm(enumerate(omniglot_alphabet)):
        if True:
            # Compute logits
            y_lgts = model(image, transform(character).unsqueeze(dim=0))
            # Compute scores
            y_prob = torch.sigmoid(y_lgts)
            y_probs.append(y_prob)

    top3_prob, top3_index = torch.topk(torch.tensor(y_probs), k=3)

    columns = st.columns([3, 1])

    for col, prob, i in zip(columns, top3_prob, top3_index):
        st.subheader(f'Probability: {prob.item():.2f}')
        st.image(omniglot_alphabet.__getitem__(i)[0])