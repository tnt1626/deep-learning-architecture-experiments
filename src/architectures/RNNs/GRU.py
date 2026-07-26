import torch
import torch.nn as nn

class PinteeGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PinteeGRUCell, self).__init__()
        self.linear_rz = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.linear_a = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, a_prev):
        combined = torch.cat((x_t, a_prev), dim=1)
        r_gate, z_gate = self.linear_rz(combined).chunk(2, dim=1)
        r_t = torch.sigmoid(r_gate)
        z_t = torch.sigmoid(z_gate)
        
        candidate_input = torch.cat((x_t, r_t * a_prev), dim=1)
        h_candidate = torch.tanh(self.linear_a(candidate_input))

        h_t = (1 - z_t) * h_candidate + z_t * a_prev

        return h_t
    

class PinteeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = PinteeGRUCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        hidden_states = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.gru_cell(x_t, h_t)
            hidden_states.append(h_t)
        
        output = torch.stack(hidden_states, dim=1)

        return output, h_t

