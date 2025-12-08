import torch
from torch import nn
import random
import numpy as np

class NeuralNetworkCUDA(nn.Module):
    '''
    CUDA-Optimized Convolutional Neural Network for Hex
    Uses convolutions to detect spatial patterns (bridges, connections, etc.)
    '''
    def __init__(self, board_size=11, device=None):
        super().__init__()
        self.board_size = board_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            # Input: (batch, 2, 11, 11)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # (batch, 32, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (batch, 64, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (batch, 128, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (batch, 64, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ).to(self.device)
        
        # Dense layers for policy head
        conv_output_size = 64 * board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, board_size * board_size)
        ).to(self.device)
        
        # Collect all layers for parameter operations
        self.all_layers = []
        for module in [self.conv_layers, self.policy_head]:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    self.all_layers.append(layer)
    
    def forward(self, x):
        '''
        Forward pass supporting batches. Returns logits.
        Args:
            x: (batch, h, w, 2) Tensor or (h, w, 2) Tensor
        Returns:
            logits: (batch, h*w)
        '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Reshape from (batch, h, w, 2) to (batch, 2, h, w) for Conv2d
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional feature extraction
        features = self.conv_layers(x)
        
        # Policy head (move probabilities)
        logits = self.policy_head(features)
        
        return logits
    
    def setParameters(self, net):
        '''Copy parameters from another network'''
        self.load_state_dict(net.state_dict())
    
    def randomiseParameters(self, strength=0.1):
        '''Randomise the network's parameters'''
        with torch.no_grad():
            for layer in self.all_layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data = strength * torch.randn_like(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data = strength * torch.randn_like(layer.bias)
    
    def printParameters(self):
        print("\n === Network Parameters ===")
        for idx, layer in enumerate(self.all_layers):
            print(f"\nLayer {idx}: {layer.__class__.__name__}")
            if hasattr(layer, 'weight'):
                print(f"Weight shape: {layer.weight.shape}")
                print(f"Weight stats: mean={layer.weight.data.mean():.4f}, std={layer.weight.data.std():.4f}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                print(f"Bias shape: {layer.bias.shape}")
                print(f"Bias stats: mean={layer.bias.data.mean():.4f}, std={layer.bias.data.std():.4f}")
    
    @staticmethod
    def deepcopy(n):
        new_nn = NeuralNetworkCUDA(n.board_size, n.device)
        new_nn.setParameters(n)
        return new_nn
    
    @staticmethod
    def mutate(n, strength=0.1):
        new = NeuralNetworkCUDA.deepcopy(n)
        with torch.no_grad():
            for layer in new.all_layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data += strength * torch.randn_like(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data += strength * torch.randn_like(layer.bias)
        return new
    
    @staticmethod
    def crossover(n1: "NeuralNetworkCUDA", n2: "NeuralNetworkCUDA"):
        newDevice = 'cuda' if n1.device == 'cuda' and n2.device == 'cuda' else 'cpu'
        child = NeuralNetworkCUDA(n1.board_size, device=newDevice)
        
        with torch.no_grad():
            for child_layer, p1_layer, p2_layer in zip(child.all_layers, n1.all_layers, n2.all_layers):
                if hasattr(child_layer, 'weight'):
                    # Uniform crossover at parameter level
                    mask = torch.rand_like(p1_layer.weight) < 0.5
                    child_layer.weight.data = torch.where(mask, p1_layer.weight.data, p2_layer.weight.data)
                
                if hasattr(child_layer, 'bias') and child_layer.bias is not None:
                    mask = torch.rand_like(p1_layer.bias) < 0.5
                    child_layer.bias.data = torch.where(mask, p1_layer.bias.data, p2_layer.bias.data)
        
        return child
    
    def save(self, filepath):
        cpu_state = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save({
            'state_dict': cpu_state,
            'board_size': self.board_size
        }, filepath)
    
    @staticmethod
    def load(filepath, device=None, default_board_size=11):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(filepath, map_location="cpu")

        # Case 1: checkpoint is {'state_dict': ..., 'board_size': ...}
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            board_size = checkpoint.get('board_size', default_board_size)
        else:
            # Case 2: plain state_dict or completely different thing
            state_dict = checkpoint
            board_size = default_board_size

        network = NeuralNetworkCUDA(board_size, device=device)

        try:
            network.load_state_dict(state_dict)
        except RuntimeError as e:
            # State dict does not match this architecture – just randomise
            print(f"Warning: could not load state_dict from {filepath}: {e}")
            print("Using randomly initialised network instead.")
            network.randomiseParameters()

        network.to(device)
        return network



if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Test the network
    net = NeuralNetworkCUDA(board_size=11)
    net.printParameters()
    
    # Test forward pass
    test_board = torch.randn(1, 11, 11, 2)
    output = net(test_board)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test mutation
    mutated = NeuralNetworkCUDA.mutate(net, strength=0.1)
    print("\nMutation test: Created mutated copy")
    
    # Test crossover
    net2 = NeuralNetworkCUDA(board_size=11)
    child = NeuralNetworkCUDA.crossover(net, net2)
    print("Crossover test: Created child network")