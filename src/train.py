import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
from model import CaptchaModel
import engine

def decode_prediction(preds, encoder: preprocessing.LabelEncoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_pred = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k = k - 1
            if k == -1:
                temp.append("°")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_pred.append(tp)
    return cap_pred

            
        

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]
    
    
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(targets_flat)
    
    targets_enc = [label_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1
    train_images, test_images, train_targets, test_targets, _, test_orig_targets = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )
    
    
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    
    model = CaptchaModel(number_of_characters=len(label_enc.classes_))
    model.to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=5,
        verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fun(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fun(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_prediction(vp, label_enc)
            valid_cap_preds.extend(current_preds)
        print(list(zip(test_orig_targets, valid_cap_preds))[6:11])
        print(f"EPOCH {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")
    # Save model
    torch.save(model.state_dict(), config.MODEL_PATH)


if __name__ == '__main__':
    run_training()
    # 75 values: