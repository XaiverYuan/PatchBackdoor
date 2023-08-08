import torch, torchvision
import logging
from PatchTrainer import train
from PatchApply import getTransformations

ROOT = r"C:\Users\96585\Documents\GitHub\CodeCAP\Data\cifar10"


def getCIFAR10Dataset():
    """
    CIFAR 10 dataset
    :return: return the dataset for using
    """

    # FIXME: otherwise, change the ROOT to a new folder and change download to TRUE
    DOWNLOAD = True
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    _trainingSet = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=DOWNLOAD,
                                                transform=transform)
    _testingSet = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=DOWNLOAD,
                                               transform=transform)
    return _trainingSet, _testingSet


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainSet, testSet = getCIFAR10Dataset()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        logging.warning('The code is suggested to run in CUDA. No CUDA detected')
        device = 'cpu'

    patchOnly, patchAndTrigger = getTransformations(32, 8, 6)
    m = torchvision.models.mobilenet_v2(num_classes=10)
    anonM=torch.jit.load(r'CIFARMODELmobile')
    m.load_state_dict(anonM.state_dict())


    trans = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    patch = torch.zeros(3, 32, 32, device=device)

    report = train('result', trainSet, testSet, m, patch,
                   transformation=[patchOnly, patchAndTrigger],
                   norm=trans,
                   target=[-2, 9], inProgressShow=True, trainAccCheck=True, valiAccCheck=True,
                   rounds=10,
                   batchSize=128,device=device,autoRatio=0.5)

