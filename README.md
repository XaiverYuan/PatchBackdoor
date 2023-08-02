# PatchBackdoor
PatchBackdoor is a code base associated with paper PatchBackdoor. 

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

### Code Understanding
* PatchTrainer provides function calls to train and test.
* PatchApply provides function calls to apply the patch
* TrainPatchCifar10 gives an easy demo of it.

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

### Advance Usage
If you want to replace or change the setting, please change read the comment in PatchTrainer and PatchApply