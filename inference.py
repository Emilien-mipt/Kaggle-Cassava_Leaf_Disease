import tqdm
import numpy as np

import torch

def inference(model, model_state, test_loader, device):
    model.to(device)
    model.load_state_dict(torch.load(model_state)['model'])
    model.eval()
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    pred_probs = []
    for i, (images) in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        pred_probs.append(y_preds.softmax(1).to('cpu').numpy())
    probs = np.concatenate(pred_probs)
    return probs