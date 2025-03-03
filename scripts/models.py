from imports import*

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class CNN_LSTM(nn.Module):
    def __init__(self, conv_kernel_size, hidden_dim, fc_hidden_dims):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=conv_kernel_size)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, 2*hidden_dim, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(2*hidden_dim, 4*hidden_dim, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(4*hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dims[0])
        self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
        self.fc3 = nn.Linear(fc_hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose to [batch_size, channels, sequence_length]
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        
        x = x.transpose(1, 2)  # Transpose back to [batch_size, sequence_length, channels]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        x = x.transpose(1, 2)  # Transpose to [batch_size, channels, sequence_length] for pooling
        x = self.global_avg_pool(x).squeeze(-1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = self.sigmoid(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, fc_hidden_dims):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.fc1 = nn.Linear(in_features=158, out_features=fc_hidden_dims[0])
        self.fc2 = nn.Linear(in_features=fc_hidden_dims[0], out_features=fc_hidden_dims[1]) 
        self.fc3 = nn.Linear(in_features=fc_hidden_dims[1], out_features=fc_hidden_dims[2]) 
        self.fc4 = nn.Linear(in_features=fc_hidden_dims[2], out_features=fc_hidden_dims[3]) 
        self.fc5 = nn.Linear(in_features=fc_hidden_dims[3], out_features=1) 
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = x.transpose(1, 2)  # Transpose to [batch_size, channels, sequence_length]
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.maxpool(x) # Pooling on the sequence length
        x = x.mean(dim=1) # Average on the number of channels
        x = torch.reshape(x, (-1,158)) # 158 is the length of the sequence
        
        x = torch.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        
        x = torch.relu(self.fc3(x))

        x = self.fc4(x)

        x = self.fc5(x)
        
        x = self.sigmoid(x)

        return x
    
class KAN(nn.Module):
    def __init__(self, fc_hidden_dims, grid, order):
        super(KAN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
        self.maxpool = nn.MaxPool1d(3, stride=2)
        self.fc1 = KANLayer(in_dim=158, out_dim=fc_hidden_dims, num=grid, k=order)
        self.fc2 = KANLayer(in_dim=fc_hidden_dims, out_dim=1, num=grid, k=order)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):

        x = x.transpose(1, 2)  # Transpose to [batch_size, channels, sequence_length]
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.maxpool(x)# Pooling on the sequence length
        x = x.mean(dim=1) # Average on the number of channels
        x = torch.reshape(x, (-1,158)) # 158 is the length of the sequence
        
        x = self.fc1(x) 
        
        x = self.dropout(x[0])
        x = self.fc2(x)
        
        return x[0]


