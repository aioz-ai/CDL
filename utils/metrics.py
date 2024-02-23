import torch

def RMSE(y_preds, y_trues):
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_preds - y_trues)**2))

def MSE(y_preds, y_trues):
    with torch.no_grad():
        return torch.mean((y_preds - y_trues)**2)