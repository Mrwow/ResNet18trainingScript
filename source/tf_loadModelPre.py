import torch
from torchvision import transforms, utils, datasets, models
import torch.nn as nn
import seaborn as sn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import time
import pandas as pd
import argparse

def plot_confusion_matrix(matrix, nc, out, normalize=True, save_dir='', names=()):
    '''


    '''

    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
    print(array)
    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=2.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and len(names) == nc  # apply names to ticklabels

    sn.heatmap(array, annot=nc < 30, annot_kws={"size": 20}, cmap='Blues', fmt='.2f', square=True,
               xticklabels=names if labels else "auto",
               yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))

    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    out_name = out + '_confusion_matrix.png'
    fig.savefig(Path(save_dir) / out_name, dpi=250)
    plt.close()

def foldImgPre(dir, gpu, weight, names, out):
    nc = len(names)
    gpu = 'cuda:' + str(gpu)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    # image loader
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_fold = datasets.ImageFolder(dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(img_fold, batch_size=10, shuffle=False)

    # define model
    model = models.resnet18(pretrained=False)
    num_fc_in = model.fc.in_features
    model.fc = nn.Linear(num_fc_in, nc)
    weights = torch.load(weight, map_location=device)
    model.load_state_dict(weights)
    model.to(device)


    model.eval()
    correct_test = 0
    y_true = []
    y_pred = []
    y_pred_score = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            # Record the correct predictions for training data
            predict_score, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum()
            y_true.append(labels)
            y_pred.append(predicted)

            for output in outputs:
                predict_score = torch.nn.functional.softmax(output, dim=0)
                predict_score = predict_score.tolist()
                y_pred_score.append(predict_score)
    print(y_true)
    print(y_pred)
    print(y_pred_score)
    # GPU tensor to cpu tensor
    y_true = torch.cat(y_true, 0)
    y_true = y_true.cpu()
    y_pred_class = torch.cat(y_pred, 0)
    y_pred_class = y_pred_class.cpu()
    # Test confusion matrix in each epoch
    cm_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_class)
    print(cm_matrix)
    plot_confusion_matrix(cm_matrix.transpose(), nc=nc, save_dir=dir, names=names, out=out)

    # output the (img name, prediction label, prediction possibility, true label) into csv file
    ids_list = img_fold.imgs
    ids = [os.path.basename(i[0]) for i in ids_list]
    df = {
        'id': ids,
        'true_label': y_true,
        'prediction': y_pred_class,
        'Prediction score': y_pred_score,
    }
    df = pd.DataFrame(df)
    csv_out = out + "_prediction.csv"
    out_path  = os.path.join(dir,csv_out)
    df.to_csv(out_path, index=False)


parser = argparse.ArgumentParser(description="Define the data folder, gpu and out name")
parser.add_argument('--gpu', '-g', help='number for gpu:1, 2, 3')
parser.add_argument('--data', '-d', help='the data folder', required=True)
parser.add_argument('--out', '-o', help='out name for acc and model', required=True)
parser.add_argument('--names', '-n', help='class name tuple', nargs='+', required=True)
parser.add_argument('--weight', '-w', help='num of epoch', required=True)
args = parser.parse_args()


if __name__ == '__main__':
    # model_weight = "/Users/ZhouTang/Downloads/zzlab/1_Project/Meijing/pad/wheat_seed_pad_Aug.pth"
    # dir = "/Users/ZhouTang/Downloads/zzlab/1_Project/Meijing/split/test"
    # foldImgPre(dir=dir, gpu=1, names=[0,1], weight=model_weight, out="wheatSeedVal")
    foldImgPre(dir=args.data, gpu=args.gpu, weight=args.weight, names=args.names, out=args.out)
