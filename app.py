
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import numpy as np

from siamese.model.app_utils import load_model, characters_database, get_predictions

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

st.subheader('Draw a character and right click to submit')

# Specify canvas parameters in application
stroke_width = st.sidebar.slider(
    label='Stroke width:',
    min_value=1, 
    max_value=25, 
    value=15
)
drawing_mode = st.sidebar.selectbox(
    label='Drawing tool:', 
    options=['freedraw', 'line', 'rect', 'circle', 'transform']
)
realtime_update = st.sidebar.checkbox(
    label='Update in realtime', 
    value=False
)

alphabet = st.sidebar.selectbox(
    label='Alphabet',
    options=ALPHABETS,
    index=21 # Latin alphabet
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

model = load_model('siamese/model/siamese-network.pt')
characters_alphabet = characters_database(alphabet=alphabet, subset=True)

if canvas_result.image_data.mean() != 0.0:
    image = canvas_result.image_data

    # Convert RGBA image to grayscale (PIL doesn't convert as I want)
    image = np.uint8(255 - image[:, :, 3])
    image = Image.fromarray(image, mode='L').resize(size=(105, 105))
    
    topk_prob, topk_index = get_predictions(model, image, characters_alphabet)

    columns = st.columns(3)
    for col, prob, i in zip(columns, topk_prob, topk_index):
        col.markdown(f'Probability: {prob.item():.2f}')
        col.image(characters_alphabet[i])