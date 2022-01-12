To train the model with PyTorch Lightning

```
python train_lightning.py \
--trainer.gpus=1 \
--trainer.max_epoch=50
```

Or to train in native PyTorch
```
python train_pytorch.py \
--batch-size=128 \
--epochs=50 \
--lr=0.001 \
--weight_decay=0.1 \
--num_workers=4 \
--run=4 \
--cuda
```

To run Tensorboard 

```
tensorboard --logdir=<directory>
```