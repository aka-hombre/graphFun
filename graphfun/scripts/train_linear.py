import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#-----
#    Local modules
#-----
from graphfun.data.data_manager import DataManager, GraphDataSet
from graphfun.data.graph_grappler import get_graphs
from graphfun.models import get_model

print(6*'*'+'starting program'+'*'*6)

#-----
#   Hyperparameters
#-----
cfg = dict()
cfg['numEpoch'] = 100
cfg['learning_rate'] = .0001
cfg['batchSize'] = 1024
cfg['model'] = 'linear'

#-----
#   Device specific training
#-----

cfg['num_workers'] = 2  # if -c 8 OR 12 id -c 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#-----
#   Loading Data & Preprocessing
#-----
p_path = "data/metadata/graphs_manifest.parquet"
d_path = "data/graph_data/V10/"

data = DataManager(parquet_path=p_path, data_dir=d_path).to_full_dataframe()
G, labels = get_graphs(data, return_adj=True)

#-----  Spliting train and test
G_train, G_test, y_train, y_test = train_test_split(G, 
                                                    labels, 
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=11)
train_set = GraphDataSet(G_train, y_train)
test_set = GraphDataSet(G_test, y_test)

#-----  DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=cfg['batchSize'],
    shuffle=True,
    num_workers=cfg['num_workers'],
    pin_memory=(device.type == "cuda")          # If device cuda, then true  
)

test_loader = DataLoader(
    test_set,
    batch_size=cfg['batchSize'],
    shuffle=False,
    num_workers=cfg['num_workers'],
    pin_memory= (device.type == "cuda") 
)


#----- 
#   Model Making
#-----
myModel = get_model(cfg['model'], in_dimension=100, classes=2)
myModel = myModel.to(device)

myLoss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])


print(f"Number of paramerters of Linear model: {sum(p.numel() for p in myModel.parameters() if p.requires_grad)}")


N_train = len(train_loader.dataset) 
N_test = len(test_loader.dataset)
nbr_miniBatch_train = len(train_loader) # number of mini-batches
nbr_miniBatch_test = len(test_loader)
average_loss = []
average_test_loss = []

df = pd.DataFrame(index=range(cfg['numEpoch']),
                  columns=('epoch', 'loss_train', 
                           'loss_test','accuracy_train',
                           'accuracy_test'))

log_str=''

for epoch in range(cfg['numEpoch']):
    # a new epoch begins
    
    log_str+='\n-- epoch '+str(epoch) 
    print('-- epoch '+str(epoch))
    running_loss_train = 0.0
    accuracy_train = 0.0
    myModel.train()
    
    for X,y in train_loader:
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        # 2) compute the score and loss
        score = myModel(X)
        loss = myLoss(score, y)
        # 3) estimate the gradient and update parameters
        loss.backward()
        optimizer.step()
        # 4) estimate the overall loss over the all training set
        running_loss_train += loss.detach().cpu().numpy()
        accuracy_train += (score.argmax(dim=1) == y).sum().cpu().numpy()
        # end epoch

    myModel.eval()  
    running_loss_test = 0.0
    accuracy_test = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            # 1) compute score and loss
            scores = myModel(X)
            loss = myLoss(scores, y)

            # 2) estimate the overall loss over the all test set
            running_loss_test += loss.detach().cpu().numpy()
            accuracy_test += (scores.argmax(dim=1) == y).cpu().numpy()
    # End epoch

    #-----
    #   Stats
    #-----
    loss_train = running_loss_train/nbr_miniBatch_train
    loss_test = running_loss_test/nbr_miniBatch_test
    accuracy_train /= N_train
    accuracy_test /= N_test
    print('    loss     (train, test): {:.4f},  {:.4f}'.format(loss_train, loss_test))
    print('    accuracy (train, test): {:.4f},  {:.4f}'.format(accuracy_train, accuracy_test))
    log_str+='    loss     (train, test): {:.4f},  {:.4f}'.format(loss_train, loss_test)+'\n'
    log_str+='    accuracy (train, test): {:.4f},  {:.4f}'.format(accuracy_train, accuracy_test)+'\n'
    df.loc[epoch] = [epoch, loss_train, loss_test, accuracy_train, accuracy_test]
    
    # end epoch

#-----
#   Plotting train and loss
#-----

plt.figure(figsize=(8, 6))
plt.semilogy(range(cfg['numEpoch']), df['loss_train'], label=r'$\ell_{train}$')
plt.semilogy(range(cfg['numEpoch']), df['loss_test'], label=r'$\ell_{test}$')
plt.title("Loss vs Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True, which='both', axis='y')   
plt.grid(True, which='major', axis='x')
plt.savefig('graphfun/outputs/linear_output/loss.png', dpi=300)

#-----
#   Writing data to .csv and making log
#-----

df.to_csv('graphfun/outputs/linear_output/stats.csv')

with open('graphfun/outputs/linear_output/log.txt', 'w', encoding='utf-8') as f:
    f.write(log_str)


#-----
#   Confusion Matrix
#-----

all_preds = []
all_targets = []

myModel.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        
        scores = myModel(X)
        preds = scores.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())


label_gr = ['Not Planar','Planar']

fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(all_targets,all_preds, normalize='true')

disp = ConfusionMatrixDisplay(cm, display_labels=label_gr)
disp.plot(ax= ax, cmap='Blues', values_format='.2f')
fig.subplots_adjust(bottom=0.8) 
plt.tight_layout()
plt.title("Confusion Matrix")

fig.savefig('graphfun/outputs/linear_output/cm.png')

