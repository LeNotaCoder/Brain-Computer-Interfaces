import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


class Trainer:
    def __init__(self, model, device, lr=0.0001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = 100. * correct / total
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        return total_loss / len(loader), acc, kappa, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        pbar = tqdm(range(epochs), desc='Training')
        
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_kappa, _, _ = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            pbar.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'val_acc': f'{val_acc:.2f}%',
                'val_kappa': f'{val_kappa:.4f}'
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self.history
