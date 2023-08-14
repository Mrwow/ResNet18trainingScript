# ResNet18trainingScript

### Output
This is python script is a command line for ResNet18 training. It will output 
- plots for training accuracy and testing accuracy along epoches
- confusion matrix for the last epoch
- prediction results together with image name and raw label into a csv
- ResNet18 weight from best epoch and last epoch

### usage
`-g` or `--gpug` : number for gpu:1, 2, 3

`d`  or `--data` : the data folder. There should be a test and train folder. Each class image should be orgnized into a single folder for test and train separately.

`-e` or `--epoch`: num of training epoch

`-s` or `--bsz`: batch size in training

`-r` or `--resz`: resizeing image

`-o` or `--out` : out name for acc and model

`-c` or `--nc` : class number for making confusion matrix plot

`-n` or `--names`: class name tuple for making confusion matrix plot


```
python3 resnet18_train.py -g 1 -d ./customdatafolder/ -o test_aug -c 2 -n case control -e 100 -s 50 -r 256

```

