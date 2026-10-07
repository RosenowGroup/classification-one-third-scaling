import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def filter_classes(X, Y, target_classes=[0, 1, 2]):
    """
    Filters X and Y tensors to only include samples from target_classes.
    Also remaps labels to 0, 1, 2... for proper CrossEntropy calculation.
    """
    print(f"Original dataset size: {X.shape[0]}")
    
    # 1. Create a mask for rows where Y is in our target list
    mask = torch.isin(Y, torch.tensor(target_classes))
    
    # 2. Apply mask to both X and Y
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    # 3. Remap labels
    mapping = {old_label: new_label for new_label, old_label in enumerate(target_classes)}
    Y_remapped = torch.tensor([mapping[y.item()] for y in Y_filtered], dtype=torch.long)
    
    print(f"Filtered dataset size: {X_filtered.shape[0]}")
    print(f"Unique labels in filtered set: {torch.unique(Y_filtered).tolist()}")
    
    return X_filtered, Y_remapped

def evaluate_model(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Update running loss 
            # criterion(outputs, labels) returns the mean loss per batch by default.
            # We multiply by batch size to get the total loss for the batch.
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
    
    model.train()
    
    # Calculate averages
    epoch_loss = running_loss / total
    epoch_acc = (correct.double() / total).item()
    
    return epoch_acc, epoch_loss

def train_linear_head(X, Y, epochs=1, test_split=0.1, lr=0.1, bs=1, seed=0):
    torch.manual_seed(seed)
    
    # 1. Setup Data
    num_samples = X.shape[0]
    input_dim = X.shape[1]
    num_classes = len(torch.unique(Y))
    
    indices = torch.randperm(num_samples)
    test_size = int(num_samples * test_split)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    train_ds = TensorDataset(X[train_indices], Y[train_indices])
    test_ds = TensorDataset(X[test_indices], Y[test_indices])
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024) 

    # 2. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(input_dim, num_classes, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # SGD without momentum
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 3. Training Loop
    steps = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    
    running_corrects = 0
    total_steps = 0

    print(f"Starting training on {device} with LR={lr}, Seed={seed}...")
    model.train()
    
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_steps += 1
            
            if total_steps % 50 == 0:
                normalized_step = total_steps / input_dim
                steps.append(normalized_step)
                
                train_acc = running_corrects.double() / 50
                train_accuracies.append(train_acc.item())
                running_corrects = 0
                
                test_acc, test_loss = evaluate_model(model, test_loader, device)
                test_accuracies.append(test_acc)
                test_losses.append(test_loss)
                
                if total_steps % 5000 == 0:
                    print(f"Step/{input_dim}: {normalized_step:.2f} | Test Acc: {test_acc:.4f}")

    data = {"steps": steps, "test_acc": test_accuracies, "train_acc": train_accuracies, "test_loss": test_losses}
    return model, data

def generate_teacher_labels(teacher_model, X, batch_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating teacher labels on {device}...")
    
    teacher_model.eval()
    teacher_model.to(device)
    
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    y_teacher_list = []
    
    with torch.no_grad():
        for i, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
            logits = teacher_model(inputs)
            preds = torch.argmax(logits, dim=1)
            y_teacher_list.append(preds.cpu())
            
            if (i + 1) % 500 == 0:
                print(f"Processed {(i + 1) * batch_size} samples...")

    Y_teacher = torch.cat(y_teacher_list, dim=0)
    print(f"Done! Generated {Y_teacher.shape[0]} teacher labels.")
    
    return Y_teacher

def data_title_gen(lr, seed, true_labels, num_examples, bs, use_whitening=True):
    title = f"lr={lr:.2e}_seed={seed}_examples={num_examples}_bs={bs}"
    if not use_whitening:
        title += "_unwhitened"
    if true_labels:
        title += "_TrueLabels"
    else:
        title += "_TeacherLabels"
    return title

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003], dest="lr")
    parser.add_argument('--seeds', type=int, nargs="+", default=[0, 1, 2], dest="seeds")
    parser.add_argument('--num_examples', type=int, default=200000, dest="num_examples")
    parser.add_argument('--test_split', type=float, default=0.05, dest="test_split")
    parser.add_argument('--use_whitening', action='store_true', help="Apply whitening to the features")
    parser.add_argument('--data_file', type=str, default="cifar5m_vit_features_plain.pt")
    parser.add_argument('--out_dir', type=str, default="data")
    parser.add_argument('--bs', type=int, default=1, dest="bs")
    parser.add_argument('--teacher_bs', type=int, default=10, dest="teacher_bs")
    parser.add_argument('--teacher_lr', type=float, default=0.0005, dest="teacher_lr")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    data = torch.load(args.data_file)
    
    target_list = [0, 1, 7] 
    X_sub, Y_sub = filter_classes(data["features"], data["labels"], target_classes=target_list)

    print("Centering data...")
    mean_vec = torch.mean(X_sub, dim=0)
    X_centered = X_sub - mean_vec

    if args.use_whitening:
        print("Computing Covariance Matrix and Whitening...")
        n_samples = X_centered.shape[0]
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (n_samples - 1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        eigenvalues = eigenvalues.flip(dims=(0,))
        eigenvectors = eigenvectors.flip(dims=(1,))
        
        epsilon = 1e-5
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
        
        W = torch.matmul(eigenvectors, torch.matmul(D_inv_sqrt, eigenvectors.T))
        X_sub = torch.matmul(X_centered, W)
    else:
        print("Skipping whitening (using centered data).")
        X_sub = X_centered

    # Train Teacher
    print("Training Teacher Model...")
    model_teacher, _ = train_linear_head(X_sub[:args.num_examples], Y_sub[:args.num_examples], test_split=args.test_split, lr=args.teacher_lr, bs=args.teacher_bs)
    Y_teacher = generate_teacher_labels(model_teacher, X_sub)

    # Main Grid Training
    for true_labels in [True, False]:
        for lr in args.lr:
            print(f"\n--- NEXT LEARNING RATE: {lr} | True Labels: {true_labels} ---")
            for seed in args.seeds:
                Y_target = Y_sub if true_labels else Y_teacher
                model, history = train_linear_head(
                    X_sub[:args.num_examples], 
                    Y_target[:args.num_examples], 
                    test_split=args.test_split, 
                    lr=lr, 
                    bs=args.bs, 
                    seed=seed
                )
                
                filename = "cifar5m_vit_data_" + data_title_gen(lr, seed, true_labels, args.num_examples, args.bs, use_whitening=args.use_whitening) + ".npz"
                filepath = os.path.join(args.out_dir, filename)
                np.savez_compressed(filepath, **history)
                print(f"Saved {filepath}")