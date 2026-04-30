import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = np.load('graphfun/outputs/MLP_output/preds_targets.npz')
all_preds = data['preds']
all_targets = data['targets']
label_gr = ['Not Planar','Planar']

fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(all_targets,all_preds, normalize='true')

disp = ConfusionMatrixDisplay(cm, display_labels=label_gr)
disp.plot(ax= ax, cmap='Blues', values_format='.2f')

plt.tight_layout()
plt.title("Confusion Matrix")

fig.savefig('graphfun/outputs/MLP_output/cm.png')