# PatchBackdoor
The PatchBackdoor codebase is associated with the paper titled '**PatchBackdoor: Backdoor Attack against Deep Neural Networks without Model Modification**'.
### Prerequisite
* tqdm==4.63.0 
* torch==1.11.0
* torchvision==0.12.0

we did not test on other versions of torch/torchvision.
However, since we are not using some picky functions from torch.
Close versions should work too, which means, in most case, as far as you can run 
```python
import torch
a=torch.rand((3,224,224),device='cuda')
```
Then most likely, your environment could run our code.

### Example Usage
* simply running 
```bash
python TrainPatchCifar10.py
```
should work. it will keep printing the data of training. An example of data is shown below:
```
train loss: 
38614.067; 55173.612; sumLoss: 93787.679
train Top1: 
0.82;  0.87;  
vali Top1: 
0.78;  0.87;  
raio:
1.00;  1.21;    
```
First column is clean accuracy and its ratio, second column is attack success rate and its ratio.
For more about ratio please refer to comment in PatchTrainer.

Then it will generate a file called 'result.report', 
by torch.load that file, you can see the patch, the trained model, and testing data and so on.

### Data preparation (Optional)
Download Imagenette from https://github.com/fastai/imagenette. Or precisely https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
### Content 
* Transformations 
* Training and Testing
* Examples
#### Transformations(PatchApply.py)
* The code is stored in PatchApply.py
* Then we are going to explain every single function in PatchApply.py
##### PatchOnlyProtocol
* This function basically resize *pic* with given *resize* and put it into right bottom, and cover it on the patch.
In another word, the right bottom part of the patch is useless.


# This is not finished yet, will be mostly finished before 2023/8/8!