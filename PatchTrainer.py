# I hope python is a statically typed language
from typing import Callable, List, Optional, Dict, Union, Tuple
# needed for progress bar and time recording
import tqdm
import datetime
# pytorch
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
# save information when error
import traceback, atexit


def codeBook(address: str) -> str:
    """
    get the code from the address and save them into dict with its key as its address and value as its code
    :param address: the address book
    :return: the code book [address,whole code]
    """
    with open(address, 'r', encoding='utf-8') as f:
        answer = f.read()
    return answer


def test(dataloader: Union[DataLoader, Tuple[Dataset, int]],
         models: Union[List[torch.nn.Module], torch.nn.Module],
         patch: torch.Tensor,
         transformation: List[Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]]],
         norm: Optional[torchvision.transforms.Normalize],
         target: List[int],
         device: str = 'cuda', silence: bool = False) -> List[float]:
    """
    evaluate the given data loader with given data
    the patch is a parameter that parsed into the transformation like this:
    newX=transformation[i](x,patch)
    then we apply normalization:
    normedX=norm(x)
    :param dataloader: where the xs and ys comes from
                       it could be a dataloader, or a dataset with batchSize
    :param models: the model(s) we are about to evaluate
                   if it is a single model, then fine
                   if it is a list of models, then the length of it must match the length of transformation and target!
    :param patch: the patch we are about to evaluate 3D (C,H,W)
    :param transformation: a list of transformation that will be applied before the x was passed into the model.
        if None, then a dummy one (lambda x:x) would be used.
    :param norm: the normalization before calling the model
    :param target: a list of target class, None or negative numbers means use the original y.
    :param device: the device during evaluating
    :param silence: Do we show the process of testing?
    :return: a list of percentage of data equals to the target class
    """

    if norm is None:
        print("WARNING: No Normalization is passed in!")
        norm = lambda dummyX: dummyX

    if not issubclass(type(dataloader), DataLoader):
        # if a Dataset is passed in
        dataloader = DataLoader(dataloader[0], dataloader[1])

    if type(models) is not list:
        # in this case, models are just one model, we parse it to be a list of models
        models = [models for _ in range(len(target))]

    # keep track of original state of the models
    state = []
    for m in models:
        state.append(m.training)
        m.to(device)
        m.eval()
    del m

    if len(models) != len(transformation) or len(models) != len(target):
        raise Exception('The length of those should be same!')

    def realM(dummyX, dummyM):
        return dummyM(norm(dummyX))

    # if transformation is None, then simply return image (ignore the patch)
    transSize = len(transformation)
    for transI in range(len(transformation)):
        if transformation[transI] is None:
            transformation[transI] = lambda x, p: x
    del transI

    answer = [0 for _ in range(transSize)]

    if (silence):
        # if silence is on, then we do not need tqdm
        def myIter(dummyX):
            return dummyX
    else:
        # if silence is off, turn it on
        def myIter(dummyX):
            return tqdm.tqdm(iter(dummyX))
    del silence

    for x, y in myIter(dataloader):

        x = x.to(device)
        if (type(y) == torch.tensor):
            # if y is an integer, then we can not change the device of it
            y = y.to(device)

        for transI in range(transSize):
            currX = transformation[transI](x.clone(), patch)
            if (target[transI] == None or target[transI] < 0):
                # users could also use target[i]<0 to represent they want original label as the target
                answer[transI] += (torch.argmax(realM(currX, models[transI]), dim=1) ==
                                   y.to(device)).float().sum().item()
            else:
                answer[transI] += (torch.argmax(realM(currX, models[transI]), dim=1) ==
                                   torch.tensor(target[transI], device=device).repeat(y.shape)).float().sum().item()
    del x, y, transI, currX

    # dataset is assumed to have len
    dataSize = len(dataloader.dataset)
    for transI in range(transSize):
        answer[transI] /= dataSize

    # switch model states back
    for i in range(len(models)):
        m = models[i]
        if state[i]:
            m.train()
        else:
            m.eval()

    return answer


def train(name: str,
          trainLoader: Union[Dataset, DataLoader], valiLoader: Union[Dataset, DataLoader],
          models: Union[List[torch.nn.Module], torch.nn.Module], patch: torch.tensor,
          transformation: List[Callable[[torch.tensor, torch.tensor], torch.tensor]],
          norm: Optional[torchvision.transforms.Normalize],
          target: List[int],
          ratio: Optional[List[float]] = None, autoRatio: Optional[float] = None,
          batchSize: int = 64, lr: float = 0.001, rounds: int = 20,
          device: str = 'cuda',
          schedulerGamma: float = 0.8, schedulerMileStone: Optional[List[int]] = None,
          trainAccCheck: Union[bool, int] = False,
          valiAccCheck: Union[bool, int] = False,
          inProgressShow: bool = False,
          peak: bool = False, autosave: bool = True) -> Dict:
    """
    train a patch under a list of transformation and a list of target class
    :param name: the address of saving, also the name of the running
    Basic Inputs
    :param trainLoader: the Dataloader/Dataset for training
    :param valiLoader: the Dataloader/Dataset for testing set
    :param models: the model we are about to evaluate or a list of models we are about to evaulate.
                  In second case, please make sure the length of model, transfomation and target are same.
    :param patch: the patch we are about to train
    :param transformation: a list of transformations we want to apply to picture and patch
                           NOTE: for all the transformation in the list, it should be a function, it takes in pictures
                                 and the patch, and apply the patch on the pictures somehow.
                           EXAMPLE: x=transformation[0](x,patch)
    :param norm: the normalize function before we transfer x into models. If none, an identity function will be passed.
    :param target: the target class we try to approach to
                   NOTE: to refer the original label, use -1
                         to train without the original label, use -2
    :param ratio: the ratio to control the loss between different transformations
                  NOTE: default which is None which means same for everyone
                  EXAMPLE: [1,1.5] means the loss of the second transformation will be multiplied by 1.5
    :param autoRatio: if it is not None, the ratio of loss will be balanced.
                      For example, if autoRatio = 1, and the training loss is 30000 vs 20000,
                      then the ratio would be [1:1.5] on next round. if ratio is also set, they will be multiplied.
                      if auto ratio =0.8, in that case, the ratio would be[1:1+(1.5-1)*0.8]
    training setting
    :param batchSize: the batchSize for both Dataloader if Dataset is passed.
    :param lr: the learning rate for training
    :param rounds: how many epoch you want to train the patch
    :param device: the device for training
    scheduler setting
    :param schedulerGamma: the factor multiplied when scheduler milestone was achieved
    :param schedulerMileStone: the time we are about to change the milestone
    testing setting.
    If it is an int then it means after how many rounds we calculate something.
    :param trainAccCheck: do we check accuracy on training set.
    :param valiAccCheck: do we check accuracy on validation set
    :param inProgressShow: do we show the process during training
    other setting
    :param peak: true means save the picture after each transformation once
    :param autosave: save the result when it is stopped or finished
    :return: a dict of data, read by readData,
    """

    patch.requires_grad = True
    patch.to(device)

    if type(models) is not list:
        # in this case, models are just one model, we parse it to be a list of models
        models = [models for _ in range(len(target))]
    for m in models:
        m.eval()
        m.to(device)

    if len(models) != len(transformation) or len(models) != len(target):
        raise Exception('The length of those should be same!')

    optimizer = torch.optim.Adam([patch], lr=lr)

    if schedulerMileStone is None:
        schedulerMileStone = [20, 40, 60]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedulerMileStone, gamma=schedulerGamma)

    soft = torch.nn.Softmax(dim=1)

    # This has to be "sum" since when calculating loss when target[i] is -2, we are doing it manually.
    lossCal = torch.nn.CrossEntropyLoss(reduction="sum")

    if issubclass(type(trainLoader), Dataset):
        trainCount = len(trainLoader)
        trainLoader = DataLoader(trainLoader, batchSize, shuffle=True, num_workers=8)
    else:
        trainCount = len(trainLoader.dataset)

    if issubclass(type(valiLoader), Dataset):
        valiCount = len(valiLoader)
        valiLoader = DataLoader(valiLoader, batchSize, shuffle=True, num_workers=8)
    else:
        valiCount = len(valiLoader.dataset)

    if norm is None:
        norm = lambda x: x

    # recording loss and accuracy
    transL = len(transformation)
    if ratio is None:
        ratio = [1 for _ in range(transL)]
    # record the ratio user want us to multiply, while ratio is the real ratio we are going to use next round
    baseRatio = ratio.copy()
    trainLossData = []
    trainAcc = []
    valiAcc = []
    timeData = []
    startTime = str(datetime.datetime.now())

    @atexit.register
    def save():
        endTime = str(datetime.datetime.now())
        report = {'patch': patch, 'model': [m.state_dict() for m in models],
                  'lr': lr, 'rounds': rounds, 'transCount': transL,
                  'train count': trainCount, 'vali count': valiCount,
                  'train loss': trainLossData,
                  'train acc': trainAcc, 'vali acc': valiAcc,
                  'start time': startTime, 'end time': endTime}
        # record the error reason
        stack = traceback.extract_stack()
        report['codebook'] = {}
        for i in range(0, len(stack) - 1):
            if 'CodeCAP' in stack[i].filename:
                report['codebook'][stack[i].filename] = codeBook(stack[i].filename)
        if autosave:
            torch.save(report, name + ".report")

    report = {'time': startTime,
              'info': 'The program has started running. Please wait until the first round finish to make sure it runs correctly.'
                      ' If program exit on error after the first round complete, information will be saved here.'}
    if autosave:
        torch.save(report, name + ".report")

    if (peak):
        x = trainLoader.dataset[0][0]
        x = x.unsqueeze(0)
        x = x.to(device)
        from torchvision.utils import save_image
        save_image(x, name + "base.png")
        for transI in range(transL):
            save_image(transformation[transI](x, patch), name + str(transI) + ".png")

    def realTest(dataLoader, dummyPatch):
        return test(dataLoader, models, dummyPatch.clone(), transformation, norm, target, device, silence=True)

    def realM(dummyX, dummyM):
        return dummyM(norm(dummyX))

    print("START:" + str(realTest(valiLoader, patch)))
    tqdmIter = tqdm.tqdm(range(rounds))
    for roundI in tqdmIter:
        # start training
        trainLoss = [0 for _ in range(transL)]
        valiLoss = [0 for _ in range(transL)]
        for x, y in trainLoader:
            x = x.to(device)
            y = y.to(device)
            loss = 0
            for transI in range(transL):
                currX = x.clone()
                transedX = transformation[transI](currX, patch)
                currO = realM(transedX, models[transI])
                if (target[transI] == -1):
                    # target[i]=-1 means they want to train with given Y
                    currLoss = lossCal(currO, y)
                elif (target[transI] == -2):
                    o = realM(x.clone(), models[transI])
                    # give a minimum bound on value to avoid nan
                    currLoss = -(soft(o) * torch.log(torch.clamp(soft(currO), min=2 ** -149))).sum()
                    del o
                elif (target[transI] >= 0):
                    # otherwise, use the given target
                    currLoss = lossCal(currO, torch.stack([torch.tensor(target[transI], device=device)] * x.shape[0]))
                else:
                    raise ("Not implemented yet")
                loss += currLoss * ratio[transI]
                trainLoss[transI] += currLoss.detach().cpu().item()
            # Step
            optimizer.zero_grad()
            # certainly, loss could be backward
            loss.backward()
            optimizer.step()
            # make sure the patch is legal
            with torch.no_grad():
                patch[:] = torch.clamp(patch, 0, 1)
        del x, y, currX, currO,
        scheduler.step()
        if autoRatio is not None:
            minLoss = min(trainLoss)
            lossRatio = trainLoss.copy()
            for i in range(transL):
                lossRatio[i] /= minLoss
                lossRatio[i] = 1 + (lossRatio[i] - 1) * autoRatio
                ratio[i] = baseRatio[i] * lossRatio[i]

        if (trainAccCheck is True or (trainAccCheck > 0 and roundI % trainAccCheck == 0)):
            currTrainAcc = realTest(trainLoader, patch)
        else:
            currTrainAcc = [0 for _ in range(transL)]

        if (valiAccCheck is True or (valiAccCheck > 0 and roundI % valiAccCheck == 0)):
            currValiAcc = realTest(valiLoader, patch)
        else:
            currValiAcc = [0 for _ in range(transL)]

        if (inProgressShow):
            print("train loss: ")
            trainLossSum = 0
            for trainLossAnon in trainLoss:
                print("%.3f; " % (trainLossAnon), end="")
                trainLossSum += trainLossAnon
            print("sumLoss: %.3f" % trainLossSum)
            del trainLossAnon, trainLossSum

            print("train Top1: ")
            for trainAccAnon in currTrainAcc:
                print("%.2f;  " % (trainAccAnon), end="")
            del trainAccAnon
            print()

            print("vali Top1: ")
            for valiAccAnon in currValiAcc:
                print("%.2f;  " % (valiAccAnon), end="")
            del valiAccAnon
            print()

            if autoRatio is not None:
                print("raio:")
                for ratioAnon in ratio:
                    print("%.2f;  " % (ratioAnon), end="")
                print()
        trainLossData.append(trainLoss)
        trainAcc.append(currTrainAcc)
        valiAcc.append(currValiAcc)
        timeData.append(tqdmIter.format_dict['elapsed'])

    print("END:" + str(realTest(valiLoader, patch)))
    # we assume Dataset has length
    report = {'patch': patch, 'model': 'models', 'lr': lr, 'rounds': rounds,
              'transCount': transL, 'train count': trainCount, 'vali count': valiCount,
              'train loss': trainLossData,
              'train acc': trainAcc, 'vali acc': valiAcc,
              'time': timeData}
    stack = traceback.extract_stack()
    report['codebook'] = {}
    for i in range(0, len(stack) - 1):
        report['codebook'][stack[i].filename] = codeBook(stack[i].filename)
    if autosave:
        torch.save(report, name + ".report")
    return report
