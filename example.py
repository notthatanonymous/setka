# import torch
# import setka
# import torchvision.datasets as datasets
# import setka.base
# import setka.pipes
# import os
# import numpy
# import sys

# sys.path.append(os.getcwd())

# from datasets import MNIST
# from models import ThreeLayersFullyConnected, LeNetFullyConvolutional

# ds = MNIST()

# ## Trying fully connected

# net = ThreeLayersFullyConnected(n = [100, 100, 10])

# def loss(predictions, targets):
#     return torch.nn.functional.cross_entropy(
#         predictions[0], targets[0])

# def acc(predictions, targets):
#     return (predictions[0].argmax(dim=-1) == targets[0]).sum(), targets[0].numel()

# trainer = setka.base.Trainer(
#     net,
#     criterion=loss,
#     optimizers=[setka.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
#     pipes=[setka.pipes.ComputeMetrics([loss, acc]),
#                setka.pipes.ExponentialWeightAveraging(epoch_start=2)]
# )

# for index in range(5):
#     trainer.train_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='valid', batch_size=128)

# ## Trying fully connected

# net = ThreeLayersFullyConnected(n = [100, 100, 10], batch_norms=True)

# def loss(predictions, targets):
#     return torch.nn.functional.cross_entropy(
#         predictions[0], targets[0])

# def acc(predictions, targets):
#     return (predictions[0].argmax(dim=-1) == targets[0]).sum(), targets[0].numel()

# trainer = setka.base.Trainer(
#     net,
#     criterion=loss,
#     optimizers=[setka.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
#     pipes=[setka.pipes.ComputeMetrics([loss, acc]),
#                setka.pipes.ExponentialWeightAveraging(epoch_start=2)]
# )

# for index in range(5):
#     trainer.train_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='valid', batch_size=128)


# # assert(trainer._metrics['valid']['accuracy'] > 0.9)
# print(f"\n\n\nScore: {trainer._metrics['valid']['accuracy']}\n\n\n")









import torchvision.transforms
import torchvision.datasets

class CIFAR10(setka.base.DataSet):
    def __init__(self,
                 root='~/datasets'):

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_data = torchvision.datasets.CIFAR10(
            '~/datasets', train=True, download=True,
            transform=train_transforms)
        self.test_data = torchvision.datasets.CIFAR10(
            '~/datasets', train=False, download=True,
            transform=test_transforms)

        self.n_valid = int(0.05 * len(self.train_data))

        self.subsets = ['train', 'valid', 'test']

    def getlen(self, subset):
        if subset == 'train':
            return len(self.train_data) - self.n_valid
        elif subset == 'valid':
            return self.n_valid
        elif subset == 'test':
            return len(self.test_data)

    def getitem(self, subset, index):
        if subset == 'train':
            image, label = self.train_data[self.n_valid + index]
            return {'image': image, 'label': label}
        elif subset == 'valid':
            image, label = self.train_data[index]
            return {'image': image, 'label': label}
        elif subset == 'test':
            image, label = self.test_data[index]
            return {'image': image, 'label': label}

import torch.nn

class SimpleModel(torch.nn.Module):
    def __init__(self, channels, input_channels=3, n_classes=10):
        super().__init__()

        modules = []

        in_c = input_channels
        for out_c in channels:
            modules.append(torch.nn.Conv2d(in_c, out_c, 3, padding=1))
            modules.append(torch.nn.BatchNorm2d(out_c))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.MaxPool2d(2))

            in_c = out_c

        self.encoder = torch.nn.Sequential(*modules)
        self.decoder = torch.nn.Linear(in_c, n_classes)

    def __call__(self, input):
        x = input['image']
        # print(x.shape)
        # print(self.encoder)
        x = self.encoder(x).mean(dim=-1).mean(dim=-1)
        x = self.decoder(x)

        return x

import setka
import setka.base
import setka.pipes


def loss(pred, input):
    return torch.nn.functional.cross_entropy(pred, input['label'])


def acc(pred, input):
    return (input['label'] == pred.argmax(dim=1)).float().sum() / float(pred.size(0))



ds = CIFAR10()
model = SimpleModel(channels=[8, 16, 32, 64])

trainer = setka.base.Trainer(
    pipes=[
        setka.pipes.DatasetHandler(ds, 32, workers=4, timeit=True,
                                   shuffle={'train': True, 'valid': True, 'test': False},
                                   epoch_schedule=[
                                       {'mode': 'train', 'subset': 'train'},
                                       {'mode': 'valid', 'subset': 'train', 'n_iterations': 100},
                                       {'mode': 'valid', 'subset': 'valid'},
                                       {'mode': 'valid', 'subset': 'test'}]),
        setka.pipes.ModelHandler(model),
        setka.pipes.LossHandler(loss),
        setka.pipes.ComputeMetrics([loss, acc]),
        setka.pipes.ProgressBar(),
        setka.pipes.OneStepOptimizers([setka.base.Optimizer(model, torch.optim.Adam, lr=3.0e-2)]),
        setka.pipes.TuneOptimizersOnPlateau('acc', max_mode=True, subset='valid', lr_factor=0.3, reset_optimizer=True),
        setka.pipes.MakeCheckpoints('acc', max_mode=True)
    ]
)


trainer.run_train(10)