import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_and_preprocess_features(cifar5m_dataset, save_path="cifar5m_vit_features.pt", batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ViT and Processor
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    # ViTModel returns the raw hidden states and pooled output (second to last layer features)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()

    dataloader = DataLoader(cifar5m_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []

    print("Extracting features from ViT... (This will take a while for 5M images)")
    with torch.no_grad():
        for images, labels in dataloader:
            # ViT processor expects a list of images or a specific tensor format
            inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            
            # The pooler_output represents the [CLS] token representation
            features = outputs.pooler_output.cpu()
            
            all_features.append(features)
            all_labels.append(labels)

    # Concatenate everything into massive tensors (~15.3 GB for 5M items)
    X = torch.cat(all_features, dim=0)
    Y = torch.cat(all_labels, dim=0)
    
    # print("Computing mean and covariance...")
    # # 1. Subtract mean
    # mean_vec = torch.mean(X, dim=0)
    # X_centered = X - mean_vec

    # # 2. Compute Covariance Matrix (768 x 768)
    # n_samples = X_centered.shape[0]
    # cov_matrix = torch.matmul(X_centered.T, X_centered) / (n_samples - 1)

    # # 3. Eigendecomposition
    # # eigh is optimized for symmetric matrices like covariance
    # eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # # Sort eigenvalues in descending order for the plot
    # eigenvalues = eigenvalues.flip(dims=(0,))
    # eigenvectors = eigenvectors.flip(dims=(1,))

    # print("Plotting Eigenspectrum...")
    # plt.figure(figsize=(10, 6))
    # plt.plot(eigenvalues.numpy(), marker='.', linestyle='none')
    # plt.yscale('log')
    # plt.title("Eigenspectrum of ViT Features (CIFAR-5M)")
    # plt.xlabel("Principal Component Index")
    # plt.ylabel("Eigenvalue Magnitude (Log Scale)")
    # plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.savefig("eigenspectrum_before_preprocessing.png")
    # plt.close()

    # print("Normalizing Covariance (ZCA Whitening)...")
    # # Add a small epsilon to prevent division by zero for near-zero eigenvalues
    # epsilon = 1e-5
    # D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    
    # # Whitening matrix W = U * D^(-1/2) * U^T
    # W = torch.matmul(eigenvectors, torch.matmul(D_inv_sqrt, eigenvectors.T))
    
    # # Apply preprocessing to features
    # X_preprocessed = torch.matmul(X_centered, W)

    print(f"Saving preprocessed features to {save_path}...")
    torch.save({'features': X, 'labels': Y}, save_path)
    print("Done!")

    return X, Y

import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os

class CIFAR5M(Dataset):
    def __init__(self, data_dir="cifar5m_data"):
        """
        Loads the 6 .npz files from the CIFAR-5M dataset.
        """
        images, labels = [], []
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if not files:
            raise FileNotFoundError(f"No .npz files found in '{data_dir}'. Please download them first.")
            
        for file in files:
            print(f"Loading {file} into memory...")
            data = np.load(file)
            # The npz files contain arrays under the keys 'X' (images) and 'Y' (labels)
            images.append(data['X']) 
            labels.append(data['Y'])
            
        self.images = np.concatenate(images, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        
        # Convert from NHWC (N, 32, 32, 3) to PyTorch's NCHW (N, 3, 32, 32)
        if self.images.ndim == 4 and self.images.shape[-1] == 3:
            self.images = self.images.transpose(0, 3, 1, 2)
            
        # Convert to float tensors. The ViT processor expects inputs in the [0, 1] range.
        self.images = torch.tensor(self.images, dtype=torch.float32) / 255.0
        
        # Some versions of CIFAR-5M have labels as floats or one-hots; ensure 1D longs for CrossEntropy
        if self.labels.ndim > 1:
            self.labels = np.argmax(self.labels, axis=1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == '__main__':
    # 1. Instantiate the dataset
    dataset = CIFAR5M(data_dir="cifar5m_data")

    # 2. Call the preprocessing function
    X_preprocessed, Y = extract_and_preprocess_features(
        cifar5m_dataset=dataset, 
        save_path="cifar5m_vit_features.pt", 
        batch_size=512 
    )