import torch
import setka.base
import setka.pipes
import sklearn.datasets
import numpy

name = 'iris_test'

class Iris(setka.base.Dataset):
    def __init__(self, valid_split=0.1, test_split=0.1):
        super()
        data = sklearn.datasets.load_iris()

        X = data['data']
        y = data['target']

        n_valid = int(y.size * valid_split)
        n_test =  int(y.size * test_split)

        order = numpy.random.permutation(y.size)

        X = X[order, :]
        y = y[order]

        self.data = {
            'valid': X[:n_valid, :],
            'test': X[n_valid:n_valid + n_test, :],
            'train': X[n_valid + n_test:, :]
        }

        self.targets = {
            'valid': y[:n_valid],
            'test': y[n_valid:n_valid + n_test],
            'train': y[n_valid + n_test:]
        }

    def getlen(self, subset):
        return len(self.targets[subset])

    def getitem(self, subset, index):
        features = torch.Tensor(self.data[subset][index])
        class_id = torch.Tensor(self.targets[subset][index:index+1])
        return [features], [class_id], subset + "_" + str(index)


class IrisNet(setka.base.Network):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 100)
        self.fc2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        return [self.fc2(self.fc1(x[0]))]


ds = Iris()
model = IrisNet()


def loss(pred, targ):
    return torch.nn.functional.cross_entropy(pred[0], targ[0][:, 0].long())


def accuracy(pred, targ):
    predicted = pred[0].argmax(dim=1)
    # print(predicted.size())
    # print(targ[0].size())
    return (predicted == targ[0][:, 0].long()).sum(), predicted.numel()


trainer = setka.base.Trainer(model,
                             optimizers=[setka.base.OptimizerSwitch(model, torch.optim.Adam, lr=3.0e-3)],
                             criterion=loss,
                             pipes=[
                                setka.pipes.ComputeMetrics(metrics=[loss, accuracy]),
                                setka.pipes.ReduceLROnPlateau(metric='loss'),
                                setka.pipes.ExponentialWeightAveraging(),
                                setka.pipes.WriteToTensorboard(name=name),
                                setka.pipes.Logger(name=name)
                              ])

for index in range(100):
    trainer.train_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='train')
    trainer.validate_one_epoch(ds, subset='valid')

# assert(trainer._metrics['valid']['accuracy'] > 0.9)
print(f"\n\n\nScore: {trainer._metrics['valid']['accuracy']}\n\n\n")
