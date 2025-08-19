"""
Simple test script for shared Kronecker factors in PSGD.
"""

import torch
import torch.nn as nn
from psgd_torch import KronWhitenShared

# ADJUST THIS TO CHANGE TRAINING STEPS
train_steps = 100  # <-- Change this to see more training iterations

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerLikeModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(32, hidden_dim, bias=False)
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for _ in range(num_layers)
        ])
        self.ffn_up = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
            for _ in range(num_layers)
        ])
        self.ffn_down = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, 10, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        for i in range(len(self.attention_layers)):
            attn = torch.relu(self.attention_layers[i](x))
            ffn = torch.relu(self.ffn_up[i](attn))
            ffn = self.ffn_down[i](ffn)
            x = x + ffn
        return self.output(x)

def main():
    torch.manual_seed(42)
    
    # Create models
    model_shared = TransformerLikeModel(hidden_dim=64, num_layers=4).to(device)
    model_individual = TransformerLikeModel(hidden_dim=64, num_layers=4).to(device)
    model_individual.load_state_dict(model_shared.state_dict())
    
    # Create optimizers
    print("\nInitializing optimizers...")
    opt_shared = KronWhitenShared(
        list(model_shared.parameters()),
        share_factors=True,
        lr_params=0.003,
        # lr_preconditioner=0.1,
        momentum=0.95,
        dQ="QEQ"
    )
    
    opt_individual = KronWhitenShared(
        list(model_individual.parameters()),
        share_factors=False,
        lr_params=0.003,
        # lr_preconditioner=0.1,
        momentum=0.95,
        dQ="QEQ"
    )
    
    # Training data
    x = torch.randn(32, 32, device=device)
    y = torch.randn(32, 10, device=device)
    
    # Training loop
    print(f"\nTraining for {train_steps} steps...")
    print("-" * 60)
    print(f"{'Step':<10} {'Shared Loss':<20} {'Individual Loss':<20}")
    print("-" * 60)
    
    for i in range(train_steps):
        # Train shared model
        def closure_shared():
            return nn.MSELoss()(model_shared(x), y)
        loss_shared = opt_shared.step(closure_shared)
        
        # Train individual model
        def closure_individual():
            return nn.MSELoss()(model_individual(x), y)
        loss_individual = opt_individual.step(closure_individual)
        
        # Print every 10 steps or at the end
        if i == 0 or (i + 1) % 10 == 0 or i == train_steps - 1:
            print(f"{i+1:<10} {loss_shared.item():<20.6f} {loss_individual.item():<20.6f}")
    
    print("-" * 60)

if __name__ == "__main__":
    main()
