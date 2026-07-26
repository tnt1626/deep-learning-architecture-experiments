import torch
import torch.nn as nn

class PurePinteeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PurePinteeLSTMCell, self).__init__()

        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_u = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, a_prev, c_prev):
        combined = torch.cat((a_prev, x_t), dim=1)
        ft = torch.sigmoid(self.W_f(combined))
        ut = torch.sigmoid(self.W_u(combined))
        cct = torch.tanh(self.W_C(combined))
        ct = ft * c_prev + ut * cct
        ot = torch.sigmoid(self.W_o(combined))
        at = ot * torch.tanh(ct)
        return at, ct
    
class PinteeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PinteeLSTMCell, self).__init__()
        self.W_gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, a_prev, c_prev):
        combined = torch.cat((a_prev, x_t), dim=1)

        gates = self.W_gates(combined)
        f_gate, u_gate, c_gate, o_gate = gates.chunk(4, dim=1)

        ft = torch.sigmoid(f_gate)   
        ut = torch.sigmoid(u_gate)     
        cct = torch.tanh(c_gate)      
        ot = torch.sigmoid(o_gate)     
        
        ct = ft * c_prev + ut * cct
        at = ot * torch.tanh(ct)
        
        return at, ct
    
class PinteeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = PinteeLSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        a_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_states = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            a_t, c_t = self.lstm_cell(x_t, a_t, c_t)
            hidden_states.append(a_t)

        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states, (a_t, c_t)