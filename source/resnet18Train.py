import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets, models

from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import pandas as pd
import os
import seaborn as sn

def plot_confusion_matrix(matrix, nc, out, normalize=True, save_dir='', names=()):

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

def main(data_dir, out, gpu, nc, names, nps, bsz, resz):
    # Load the data
    transform = transforms.Compose([transforms.Resize((resz, resz)), transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=transform)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_fc_in = model.fc.in_features
    model.fc = nn.Linear(num_fc_in, int(nc))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gpu = 'cuda:' + str(gpu)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = nps
    # Define the lists to store the results of loss and accuracy
    train_acc_epoch = []
    test_acc_epoch = []
    test_acc_0_epoch = []
    test_acc_1_epoch = []
    best_acc = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('*' * 10)
        print(f'epoch {epoch + 1}')
        correct_train = 0
        iter_train = 0
        iter_loss = 0.0
        # training mode
        model.train()
        start = time.time()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            #  print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            # print(loss)
            iter_loss += float(loss)
            # backward
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum()
            iter_train += 1

        # # Record the training loss
        # train_loss_epoch.append(iter_loss / iter_train)
        # Record the training accuracy
        train_acc_epoch.append((correct_train / len(train_data)))

        # Testing
        # evaluation mode
        model.eval()
        correct_test = 0
        y_true = []
        y_pred = []
        y_pred_score = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Calculate the loss
                # Record the correct predictions for training data
                predict_score, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum()
                y_true.append(labels)
                y_pred.append(predicted)

                # print("+" * 10)
                # print(outputs)
                # print(outputs[0])
                for output in outputs:
                    predict_score = torch.nn.functional.softmax(output, dim=0)
                    predict_score = predict_score.tolist()
                    y_pred_score.append(predict_score)
                # print(y_pred_score)
                # print(len(y_pred_score))

        # Record the Testing accuracy
        test_acc_epoch.append((correct_test / len(test_data)))
        stop = time.time()
        print(
            'Epoch {}/{}, Training Accuracy: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                .format(epoch + 1, num_epochs, train_acc_epoch[-1], test_acc_epoch[-1], stop - start))

        # GPU tensor to cpu tensor
        y_true = torch.cat(y_true, 0)
        y_true = y_true.cpu()
        y_pred_class = torch.cat(y_pred, 0)
        y_pred_class = y_pred_class.cpu()

        # Test confusion matrix in each epoch
        cm_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_class)
        print(cm_matrix)

        each_class_acc = cm_matrix.diagonal() / cm_matrix.sum(axis=1)
        each_class_ratio = cm_matrix.sum(axis=1) / cm_matrix.sum()
        acc_test_epoch = cm_matrix.diagonal().sum() / cm_matrix.sum()
        print("Overall accuracy is: {:.3f}".format(acc_test_epoch))
        print("Each class accuracy:")
        print(each_class_acc)
        print(y_true)
        print(y_pred_class)
        print(y_pred_score)
        if acc_test_epoch > best_acc:
            best_acc = acc_test_epoch
            if train_acc_epoch > 0.99:
                best_modelname = out + '_best.pth'
                torch.save(model.state_dict(), best_modelname)


    print('Finished Training')
    # plot confusion matrix of final epoch
    plot_confusion_matrix(cm_matrix.transpose(), nc=nc, save_dir="./", names=names, out=out)

    # plot training and testing accuracy along epoch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_acc_epoch, label="train acc")
    plt.plot(test_acc_epoch, label="test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(out + ".jpg")

    # save final epoch model weight
    modelname = out + 'last_eph.pth'
    torch.save(model.state_dict(), modelname)

    # output the (img name, prediction label, prediction possibility, true label) into csv file
    ids_list = test_data.imgs
    ids = [os.path.basename(i[0]) for i in ids_list]

    df = {
        'id': ids,
        'true_label': y_true,
        'prediction': y_pred_class,
        'Prediction score': y_pred_score,
    }
    df = pd.DataFrame(df)
    out_path = out + '_prediction.csv'
    df.to_csv(out_path, index=False)


parser = argparse.ArgumentParser(description="Define the data folder, gpu and out name")
parser.add_argument('--gpu', '-g', help='number for gpu:1, 2, 3')
parser.add_argument('--data', '-d', help='the data folder', required=True)
parser.add_argument('--out', '-o', help='out name for acc and model', required=True)
parser.add_argument('--nc', '-c', type=int, help='class number', required=True)
parser.add_argument('--names', '-n', help='class name tuple', nargs='+', required=True)
parser.add_argument('--epoch', '-e', type=int, help='num of epoch', required=True)
parser.add_argument('--bsz', '-s', type=int, help='batch size', required=True)
parser.add_argument('--resz', '-r', type=int, help='resize of image', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(0)
    # data_dir = "./data_09"
    main(args.data, args.out, args.gpu, args.nc, args.names, args.epoch, args.bsz)
