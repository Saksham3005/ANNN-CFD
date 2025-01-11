import torch 
import torch.nn as nn
from torchsummary import summary

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4000, 3800)   
        self.fc2 = nn.Linear(3800, 3600)  
        self.fc3 = nn.Linear(3600, 3400)  
        self.fc4 = nn.Linear(3400, 3200)  
        self.fc5 = nn.Linear(3200, 3000)  
        self.fc6 = nn.Linear(3000, 3000)
        self.fc7 = nn.Linear(3000, 3000)
        self.fc8 = nn.Linear(3000, 3000)
        self.fc9 = nn.Linear(3000, 3000)
        self.fc10 = nn.Linear(3000, 3000)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = torch.tanh(self.fc4(x)) 
        x = torch.tanh(self.fc5(x)) 
        x = torch.tanh(self.fc6(x)) 
        x = torch.tanh(self.fc7(x)) 
        x = torch.tanh(self.fc8(x)) 
        x = torch.tanh(self.fc9(x))
        x = self.fc10(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size=5000, hidden_size=3000, num_layers=4, batch_first=True):
        super(LSTMNet, self).__init__()
        
        # Initial dense layer to reduce dimensionality
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=0.2  # Adding dropout for regularization
        )
        
        # Final output layer
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # Reshape input for LSTM if it's not already in the right shape
        # Expected shape: (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            # Assuming input is (batch_size, features)
            # Reshape to (batch_size, 1, features) for single time step
            x = x.unsqueeze(1)
            
        # Initial dimensionality reduction
        x = torch.tanh(self.input_layer(x))
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last time step
        x = lstm_out[:, -1, :]
        
        # Final output layer
        x = self.fc_out(x)
        
        return x
    
if __name__ == "__main__":
    model = LSTMNet().to('cuda')
    input = torch.randn(2, 5000).to('cuda')
    summary(model, (5000,),batch_size=2)    
