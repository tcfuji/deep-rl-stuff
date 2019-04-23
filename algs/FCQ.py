import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32),
                activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dims, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        x = torch.tensor(state, device=self.device, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def to_float(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def to_long(self, variable):
        variable = torch.from_numpy(variable).long().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = self.to_float(states)
        actions = self.to_long(actions)
        new_states = self.to_float(new_states)
        rewards = self.to_float(rewards)
        is_terminals = self.to_float(is_terminals)
        return states, actions, new_states, rewards, is_terminals
