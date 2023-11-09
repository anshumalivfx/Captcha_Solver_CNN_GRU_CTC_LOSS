import tqdm 
import torch
import config


def train_fun(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm.tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
        
    return fin_loss / len(data_loader)


def eval_fun(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_pred = []
    
    
    with torch.no_grad():
        tk = tqdm.tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_pred.append(batch_preds)
        
        return fin_pred, fin_loss / len(data_loader)
