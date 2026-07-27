import sklearn.metrics as metrics
from numpy import mean

def evaluate(model, test_loader, device):
    model.eval()
    
    y_true = []
    y_pred = []

    for wig, lab in test_loader:
        wig = wig.to(device)
        lab = lab.long()
        
        preds = model(wig)
        pred = preds.argmax(dim=1)


        y_true.extend(lab.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    accuracy = mean([ y_pred[i] == y_true[i] for i in range(len(y_pred)) ])
    cm = metrics.confusion_matrix(y_true, y_pred)

    return accuracy, cm