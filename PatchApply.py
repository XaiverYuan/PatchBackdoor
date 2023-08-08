import torch, torchvision
from typing import Callable, Union, Tuple



def patchOnlyProtocol(pic: torch.Tensor, patch: torch.Tensor, resize: torchvision.transforms.Resize,
                      side: int) -> torch.Tensor:
    """
    apply the patch to the picture
    !!This is only a protocol,
    you should create a new function call who calls this and use that as the patchOnly function
    :param pic: the picture, expected 4D(B,C,H,W)
    :param patch: the patch, expected 3D(C,H,W)
    :param resize: the resize function
    :param side: the side of the patch
    :return: the picture
    """
    newX = torch.zeros(pic.shape, device=pic.device)
    newX[:] = patch
    newX[:, :, side:, side:] = resize(pic)
    return newX


def patchAndTriggerProtocol(pic: torch.Tensor, patch: torch.Tensor,
                            patchOnly: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], trigger: torch.Tensor,
                            x: int, y: int):
    """
    apply the patch first then apply the trigger
    !!This is only a protocol,
    you should create a new function call who calls this and use that as the patchAndTrigger function

    :param pic: the picture, expected 4D(B,C,H,W)
    :param patch: the patch, expected 3D(C,H,W)\
    :param patchOnly: the apply patch function NOT THE PROTOCOL
    :param trigger: the trigger
    :param x: the top left corner x for trigger
    :param y: the top right corner y for trigger
    :return: the picture
    """
    patchedX = patchOnly(pic, patch)
    patchedX[:, :, x:x + trigger.shape[1], y:y + trigger.shape[2]] = trigger
    return patchedX


def getTransformations(picSize: int, patchSide: int, trigger: Union[int, torch.Tensor], device: str = "cuda") \
        -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    a wrapper to get two tranformation needed for training
    !! everything should be a square!
    :param picSize: the side of image
    :param patchSide: the side of the patch
    :param trigger: the trigger size or the trigger itself
    :return: those two functions
    """
    resize = torchvision.transforms.Resize((picSize - patchSide, picSize - patchSide))

    def patchOnly(pic: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return patchOnlyProtocol(pic, patch, resize, patchSide)

    if type(trigger) == int:
        trigger = torch.ones(3, trigger, trigger, device=device)

    def patchAndTrigger(pic: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return patchAndTriggerProtocol(pic, patch, patchOnly, trigger, patchSide, patchSide)

    return patchOnly, patchAndTrigger
