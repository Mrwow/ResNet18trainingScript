# ResNet18trainingScript

### Output
This is python script is a command line for ResNet18 training. It will output 
- plots for training accuracy and testing accuracy along epoches
- confusion matrix for the last epoch
- prediction results together with image name and raw label into a csv
- ResNet18 weight from best epoch and last epoch

### usage
`-g` : number for gpu:1, 2, 3



```
python3 resnet18_train.py -g 1 -d ./customdatafolder/ -o test_aug -c 2 -n case control -e 100 -s 50 -r 256

```

