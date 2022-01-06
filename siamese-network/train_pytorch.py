import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset.dataset_alphabet import OmniglotAlphabet
from dataset.dataset_pairs import OmniglotPairs

from model.network import SiameseNetwork
from model.training_utils import train, eval_epoch

import argparse

def main():
    # See https://github.com/pytorch/examples/blob/master/mnist/main.py 
    # for a nice example for the ArgumentParser

    # Training settings
    parser = argparse.ArgumentParser(description='QuickDraw training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', dest='cuda',
                        help='Enables CUDA training.')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                        help='Disables CUDA training.')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    # If there is a GPU and CUDA is enabled the model will be trained in the GPU
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f'\nTraining in {device}\n')

    # Create model and move it to the device
    model = SiameseNetwork()
    model.to(device)
    summary(model)

    omniglot_latin = OmniglotAlphabet(
        alphabet='Latin',
        transform=transforms.ToTensor()    
    )

    omniglot_greek = OmniglotAlphabet(
        alphabet='Greek',
        transform=transforms.ToTensor()    
    )

    # I will use the Latin alphabet to train and the Greek alphabet to validate and test
    train_dataset = OmniglotPairs(
        dataset=omniglot_latin,
        n_pairs=100_000
    )

    validation_dataset, test_dataset = random_split(
        dataset=OmniglotPairs(
            dataset=omniglot_greek,
            n_pairs=10_000
        ),
        lengths=[5_000, 5_000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )

    writer = SummaryWriter(log_dir='runs/experiment-2')

    # Train the model, open TensorBoard to see the progress
    train(
        model, 
        train_loader, 
        validation_loader, 
        device, 
        lr=args.lr, 
        epochs=args.epochs, 
        writer=writer, 
        checkpoint_path='model/checkpoints/checkpoint-2.pt'
    )

    # Save the model
    torch.save(model, 'model/model-2.pt')

    # Add some metrics to evaluate different models and hyperparameters
    _, train_acc = eval_epoch(model, train_loader, device)
    _, val_acc = eval_epoch(model, validation_loader, device)
    _, test_acc = eval_epoch(model, test_loader, device)

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 
            'batch_size': args.batch_size, 
            'epochs': args.epochs
        },
        metric_dict={
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        },
        run_name='hparams'
    )

if __name__ == '__main__':
    main()