import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#   Local modules
from graphfun.data.data_manager import DataManager, GraphDataSet
from graphfun.data.graph_grappler import get_graphs


cfg = dict()
cfg['numEpoch'] = 10
cfg['learning_rate'] = .0001
cfg['batchSize'] = 11

p_path = "data/metadata/graphs_manifest.parquet"
d_path = "data/graph_data/V10/"


shard1 = DataManager(p_path, d_path).pull_shard(000)
print(shard1.head())
G, labels = get_graphs(shard1, return_adj=True)

print(G.dtype)

G_train, G_test, y_train, y_test = train_test_split(G, 
                                                    labels, 
                                                    test_size=0.2,
                                                    stratify=labels,
                                                    random_state=11)
train_set = GraphDataSet(G_train, y_train)
test_set = GraphDataSet(G_test, y_test)

train_loader = DataLoader(
    train_set,
    batch_size=cfg['batchSize'],
    shuffle=True   
)

test_loader = DataLoader(
    test_set,
    batch_size=cfg['batchSize'],
    shuffle=False  
)

myModel = torch.nn.Linear(100, 2)

myLoss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])

print(f"Number of paramerters of Linear model: {sum(p.numel() for p in myModel.parameters() if p.requires_grad)}")

nbr_miniBatch = len(train_loader)
average_loss = []
average_test_loss = []

for epoch in range(cfg['numEpoch']):
    # a new epoch begins
    print('-- epoch '+str(epoch))
    running_loss = 0.0
    running_test_loss = 0.0
    for X,y in train_loader:
        # (X,y) is a mini-batch:
        # X size Nx1x10x28 (N: size mini-batch, 1: only one color, 28x28: widthxheigh)
        # y size N
        # 1) initialize the gradient "Grad loss" to zero
        optimizer.zero_grad()
        # 2) compute the score and loss
        N,nX,nY = X.size()
        score = myModel(X.view(N,-1))
        loss = myLoss(score, y)
        # 3) estimate the gradient (back propagation -> explain next week!)
        loss.backward()
        # 4) update the parameters
        optimizer.step()
        # 5) estimate the overall loss over the all training set
        running_loss += loss.detach().numpy()
        # end epoch
    print(' average loss: '+str(running_loss/nbr_miniBatch)) # normalize  by the number of mini-batch
    average_loss.append(running_loss/nbr_miniBatch)

    myModel.eval()  
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            N,nX,nY = X.size()
            scores = myModel(X.view(N,-1))
            tloss = myLoss(scores, y)
            running_test_loss += tloss.detach().numpy()
            preds = scores.argmax(dim=1)

            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    average_test_loss.append(running_test_loss / len(test_loader))
    
    # end epoch

print(set(all_targets))
plt.semilogy(range(cfg['numEpoch']), average_loss, label=r'$\ell_{train}$')
plt.semilogy(range(cfg['numEpoch']), average_test_loss, label=r'$\ell_{test}$')
plt.title("Loss vs Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.savefig('graphfun/tests/loss.png')

label_gr = ['Not Planar','Planar']

fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(all_targets,all_preds, normalize='true')

disp = ConfusionMatrixDisplay(cm, display_labels=label_gr)
disp.plot(ax= ax, cmap='Blues', values_format='.2f')
fig.subplots_adjust(bottom=0.8) 
plt.tight_layout()
plt.title("Confusion Matrix")

fig.savefig('graphfun/tests/cm.png')