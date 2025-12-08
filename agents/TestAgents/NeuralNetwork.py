import torch;
from torch import nn;

class NeuralNetwork(nn.Module):
    def __init__(self, initial_weight_strength=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(11 * 11 * 2,  11 * 11),
            nn.Softmax(dim=-1)
        )

        self.network[0].weight.data = initial_weight_strength * torch.randn_like(self.network[0].weight)
        self.network[0].bias.data = initial_weight_strength * torch.randn_like(self.network[0].bias)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.flatten(x)
        return self.network(x)
    


    def setParameters(self, net):
        self.network[0].weight.data = net[0].weight.data.clone()
        self.network[0].bias.data = net[0].bias.data.clone()

    @staticmethod
    def deepcopy(n):
        nn = NeuralNetwork()
        nn.setParameters(n.network)
        return nn
    
    @staticmethod
    def mutate(n, strength=1):
        '''
        Takes the input neural network and returns a new one with weights/biases offset by
        normal noise
        '''
        new = NeuralNetwork.deepcopy(n)

        new.network[0].weight.data += strength * torch.randn_like(new.network[0].weight)
        new.network[0].bias.data += strength * torch.randn_like(new.network[0].bias)

        return new
    
    @staticmethod
    def crossover(n1, n2):
        '''
        Creates a child network with the average attributes of both parents
        '''
        child = NeuralNetwork()
        child.network[0].weight.data = (n1.network[0].weight.data + n2.network[0].weight.data) / 2  
        child.network[0].bias.data = (n1.network[0].bias.data + n2.network[0].bias.data) / 2

        return child
    
    def save(self, filepath):
        torch.save({
            'state_dict' : self.state_dict(),
        }, filepath)

    @staticmethod
    def load(filepath):
        checkpoint = torch.load(filepath)
        network = NeuralNetwork()
        network.load_state_dict(checkpoint['state_dict'])
        return network