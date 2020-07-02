# training
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from utils import evaluation


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters())
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training hajimeruyo~
        for i, (inputs,labels) in enumerate(train):
            inputs = inputs.to(device,dtype=torch.long)
            labels = labels.to(device,dtype=torch.float)
            optimizer.zero_grad()  # 由於loss.backward()的gradient會累加，所以每次餵完一個batch後需要歸零
            outputs = model(inputs)  # 將input餵給模型
            outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
            loss = criterion(outputs, labels)  # 計算此時模型的training loss
            loss.backward()  # 算loss的gradient
            optimizer.step()  # 更新訓練模型的參數
            correct = evaluation(outputs, labels)  # 計算此時模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # 這段做validation
        model.eval()  # 將model的模式設為eval，這樣model的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # device為"cuda"，將inputs轉成torch.cuda.LongTensor
                labels = labels.to(device,
                                   dtype=torch.float)  # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
                outputs = model(inputs)  # 將input餵給模型
                outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
                loss = criterion(outputs, labels)  # 計算此時模型的validation loss
                correct = evaluation(outputs, labels)  # 計算此時模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果validation的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
        print('-----------------------------------------------')
        model.train()  # 將model的模式設為train，這樣optimizer就可以更新model的參數（因為剛剛轉成eval模式）
