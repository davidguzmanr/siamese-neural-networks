import torch
from torchvision import transforms
from torchvision.datasets import Omniglot
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset.dataset_pairs import OmniglotPairs

from model.network import SiameseNetwork
from model.training_utils import train, eval_epoch

import argparse

# Reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='QuickDraw training')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Input batch size for training (default: 32)')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 20)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for L2 penalty. (default: 0.0)')
    
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    
    parser.add_argument('--run', type=int, default=1,
                        help='Current run, to track experiments (default: 1)')
    
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

    omniglot_background = Omniglot(
        root='data',
        transform=transforms.ToTensor(),
        background=True,
        download=True 
    )

    omniglot_validation = Omniglot(
        root='data',
        transform=transforms.ToTensor(),
        background=False,
        download=True 
    )

    train_dataset = OmniglotPairs(
        dataset=omniglot_background,
        n_pairs=200_000
    )

    evaluation_dataset = OmniglotPairs(
        dataset=omniglot_validation,
        n_pairs=20_000
    )
    
    validation_dataset, test_dataset = random_split(
        dataset=evaluation_dataset,
        lengths=[10_000, 10_000],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    writer = SummaryWriter(log_dir=f'runs/experiment-{args.run}')

    # Train the model, open TensorBoard to see the progress
    train(
        model, 
        train_loader, 
        validation_loader, 
        device, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        epochs=args.epochs, 
        patience=args.patience,
        writer=writer, 
        checkpoint_path=f'model/checkpoints/checkpoint-{args.run}.pt'
    )

    # Add some metrics to evaluate different models and hyperparameters
    _, train_acc = eval_epoch(model, train_loader, device)
    _, val_acc = eval_epoch(model, validation_loader, device)
    _, test_acc = eval_epoch(model, test_loader, device)

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 
            'batch_size': args.batch_size, 
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'patience': args.patience
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