import torch
import torch.nn as nn


class PureVanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PureVanillaRNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.linear_x = nn.Linear(input_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, hidden_size)
        self.linear_y = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x_t, a_prev):
        a_t = torch.tanh(self.linear_x(x_t) + self.linear_a(a_prev))
        y_hat_t = self.softmax(self.linear_y(a_t))
        return a_t, y_hat_t
    

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.combined_linear = nn.Linear(input_size + hidden_size, hidden_size) 
        self.linear_y = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x_t, a_prev):
        combined = torch.cat((x_t, a_prev), dim=1)
        a_t = torch.tanh(self.combined_linear(combined))
        y_hat_t = self.softmax(self.linear_y(a_t))
        return a_t, y_hat_t
    

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):  
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = VanillaRNNCell(input_size, hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        a_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            a_t, y_t = self.rnn_cell(x_t, a_t)
            outputs.append(y_t)

        predictions = torch.stack(outputs, dim=1)

        return a_t, predictions