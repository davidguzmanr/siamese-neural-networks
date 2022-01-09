
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from siamese.model.network import SiameseNetwork

import json
import os
import gdown

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

model = load_model()
transform = transforms.ToTensor()

if canvas_result.image_data is not None:
    image = canvas_result.image_data

    # Convert RGBA image to grayscale (PIL doesn't convert as I want)
    image = np.uint8(255 - image[:, :, 3])
    image = Image.fromarray(image, mode='L').resize(size=(105, 105))

    # st.write(image.shape, image.mean())
    # st.image(Image.fromarray(image, mode='L'))
    # st.write(transform(Image.fromarray(image, mode='L')))

    # Convert to tensor and add batch dimension
    image = transform(image).unsqueeze(dim=0)

    # Compute logits
    y_lgts = model(image, image)
    # Compute scores
    y_prob = torch.sigmoid(y_lgts)

    st.write(image.mean())
    st.write(y_lgts, y_lgts.shape)
    st.write(y_prob, y_prob.item(), y_prob.shape)