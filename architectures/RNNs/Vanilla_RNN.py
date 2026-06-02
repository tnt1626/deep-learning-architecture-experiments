import torch
import torch.nn as nn


class PureVanillaRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PureVanillaRNNCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.linear_x(x_t) + self.linear_h(h_prev))
        return h_t
    

class VanillaRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VanillaRNNCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, hidden_dim)

        self.combined_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        h_t = torch.tanh(self.combined_linear(combined))
        return h_t