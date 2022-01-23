# siamese-neural-networks
An implementation of siamese neural networks in PyTorch based on the original paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

My final accuracy in the "evaluation" set (different alphabets from the "background" set, this terminology is defined by the authors of the Omniglot dataset) is 92.62, which is almost the same accuracy reported in the paper, which is 92.0.

## Demo
You can test the one-shot image recognition for this task with different alphabets.

First create a virtual environment:
```
# Create and activate it with Conda
conda create --name=siamese-venv python=3.7
conda activate siamese-venv

# Or create and activate it with virtualenv
virtualenv siamese-venv
source siamese-venv/bin/activate
```

Then clone this repository and install the requirements:
```
git clone https://github.com/davidguzmanr/siamese-neural-networks.git
cd siamese-neural-networks
pip install -r requirements.txt
```

Then run the app:
```
streamlit run app.py
```

## References
- [Building a One-shot Learning Network with PyTorch](https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a)
- [Siamese Networks for One-Shot Learning](https://github.com/fangpin/siamese-pytorch)