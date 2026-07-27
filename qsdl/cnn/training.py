from math import inf
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

def CNN_training(model, train_loader, val_loader, device, n_epochs=40, criterion=None, optimizer=None, debug=True, name="model"):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    model = model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = inf
    best_i = 0
    best_weights = None
    for epoch in range(1, n_epochs + 1):

        # ---- TRAIN ----
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for wig, lab in train_loader:

            wig = wig.to(device)
            lab = lab.long()
            lab = lab.to(device)

            optimizer.zero_grad()
            pred = model(wig)
            loss = criterion(pred, lab)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * wig.size(0)
            train_correct += (pred.argmax(dim=1) == lab).sum().item()
            train_total += wig.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ---- VAL ----
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():           
            for wig, lab in val_loader:
                wig = wig.to(device)
                lab = lab.long()
                lab = lab.to(device)
                
                pred = model(wig)
                loss = criterion(pred, lab)
                val_loss += loss.item() * wig.size(0)
                val_correct += (pred.argmax(dim=1) == lab).sum().item()
                val_total += wig.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_i = epoch
            best_weights = deepcopy(model.state_dict())

    
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if debug:
            print(
                f"Epoka {epoch}/{n_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )
        
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"models/{name}.pt")

    return (history, {"best_val_loss": best_val_loss, "epoch_i": best_i})