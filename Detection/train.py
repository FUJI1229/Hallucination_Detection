import time
import os
import copy
import torch
import torch.optim as optim
from torch import nn
from validate import validate

def train(args, all_hidden_list, all_label_list, train_loader, 
          model, device, idx_layer, val_hidden_list, val_label_list):

    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-3)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    patience = 100
    counter = 0
    best_score = 0.0
    best_th = 0.0
    model_best = copy.deepcopy(model)
    
    for epoch in range(1, args.epoch + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            indices = batch['idx']
            labels = batch['label'].to(device)
            batch_hiddens = [all_hidden_list[idx][0] if isinstance(all_hidden_list[idx], list) else all_hidden_list[idx] for idx in indices]

            optimizer.zero_grad()
            
            output = model(
                hidden_states=batch_hiddens,
                labels=labels
            )

            loss = output.get('loss') 

            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        end_time = time.time()
        epoch_duration = end_time - start_time

        # --- Validation & Early Stopping ---
        if epoch % args.val_interval == 0 or epoch == args.epoch:
            # 5. Validate: 
            # Increasing sampling at evaluation time (e.g., to 8) can stabilize AUC.
            val_auc, th, val_loss = validate(
                args, model, val_hidden_list, val_label_list,
                idx_layer, device, show_progress=False
            )
            
            # Step scheduler with validation loss.
            scheduler.step(val_loss)
            status = f"[Epoch {epoch:03d}] Time: {epoch_duration:.2f}s | Loss: {avg_loss:.4f} | Val_Loss: {val_loss:.4f} | Val_AUC: {val_auc:.4f} | Best: {best_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            if val_auc > best_score:
                best_score = val_auc
                best_th = th
                model_best = copy.deepcopy(model)
                counter = 0
                print(f"{status} -> NEW BEST")
            else:
                counter += 1
                print(f"{status} (Patience: {counter}/{patience})")

            if counter >= patience:
                print(f"!! Early stopping at epoch {epoch} !!")
                break

    # Save best checkpoint
    save_path = f"{args.save_dir}/{args.model_name}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model_best.state_dict(),
               f"{save_path}/{args.data_name}_{args.pooling_method}_{idx_layer}_best.pth")

    return model, model_best, val_auc, best_score, best_th
