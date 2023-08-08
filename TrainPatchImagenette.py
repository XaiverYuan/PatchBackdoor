import torch, torchvision
import logging, os
from PatchTrainer import train
from PatchApply import getTransformations

ROOT = r"C:\Users\96585\Desktop\imagenette2-320"


def getCIFAR10Dataset():
    """
    Imagenette dataset
    :return: return the dataset for using
    """

    # FIXME: otherwise, change the ROOT to a new folder and change download to TRUE
    DOWNLOAD = False
    trans = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.ToTensor()])
    _trainingSet = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, 'train'),
                                                    transform=trans)
    _testingSet = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, 'val'),
                                                   transform=trans)
    return _trainingSet, _testingSet


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    trainSet, testSet = getCIFAR10Dataset()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        logging.warning('The code is suggested to run in CUDA. No CUDA detected')
        device = 'cpu'
    side = 40
    size = 40
    patchOnly, patchAndTrigger = getTransformations(224, side, size)
    m = torchvision.models.resnet50(num_classes=10)
    anonM = torch.load(r'resnet.model')['model']
    m.load_state_dict(anonM)
    trans = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    print("SIDE=%d,SIZE=%d" % (side, size))
    patch = torch.rand(3, 224, 224, device=device)
    train('resnet', trainSet, testSet, m, patch,
          transformation=[patchOnly, patchAndTrigger],
          norm=trans,
          target=[-2, 9],
          rounds=10,
          valiAccCheck=True,
          inProgressShow=True,
          batchSize=16, lr=0.001, autoRatio=0.5)
