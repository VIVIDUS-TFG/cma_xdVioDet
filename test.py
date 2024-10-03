from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch
import csv
import os

def test(dataloader, model, gt):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()

        for i, inputs in enumerate(dataloader):
            inputs = inputs.cuda()

            logits = model(inputs)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)

        return pr_auc

def test_single_video(dataloader, model, args):
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = torch.zeros(0).to(device)

        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)

            logits = model(inputs)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value > 0.42 else 0 for pred_value in pred]
    return pred_binary

def save_results(results, filename):
    np.save(filename, results)
    

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec