import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from LSTMCell import LSTMCell


class ParsingNetwork(nn.Module):
    def __init__(self, ninp, nhid, nslots=5, nlookback=1, resolution=0.1, dropout=0.4, hard=False):
        super(ParsingNetwork, self).__init__()

        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = 1
        self.nslots = nslots
        self.nlookback = nlookback
        self.resolution = resolution
        self.hard = hard

        self.drop = nn.Dropout(dropout)

        # Attention layers
        '''self.gate = nn.Sequential(nn.Dropout(dropout),
                                  nn.Conv1d(ninp, nhid, (nlookback + 1)),
                                  nn.BatchNorm1d(nhid),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Conv1d(nhid, 2, 1, groups=2),
                                  nn.Sigmoid())
        '''
        self.rnn = nn.Sequential(
            nn.Dropout(dropout),
            nn.LSTM(ninp, nhid, self.nlayers, batch_first = False, dropout = 0, bidirectional = True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.postnn = nn.Sequential(
            nn.Linear(nhid * 2, 1),
            nn.Sigmoid()
        )
        self.postnn_next = nn.Sequential(
            nn.Linear(nhid * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, parser_state):
        hidden, cell, cum_gate = parser_state
        ntimestep = emb.size(0)
        #print 'ntimestep = ' + str(ntimestep)
        bsz = emb.size(1)

        #print 'ninp value: ' + str(self.ninp)
        #print 'emb size: ' + str(emb.size())

        if emb.is_cuda and not hidden.is_cuda:
            hidden = hidden.cuda()
        if emb.is_cuda and not cell.is_cuda:
            cell = cell.cuda()

        intermediate, hc = self.rnn(emb, (hidden, cell))  # ntimestep, batchsize, vector_length
        hidden_n, cell_n = hc

        g = self.postnn(intermediate)
        g_next = self.postnn_next(intermediate)

        g = g[:, :, 0]
        g_next = g_next[:, :, 0]
        g = g.transpose(0, 1)
        g_next = g_next.transpose(0, 1)

        #print 'g size: ' + str(g.size())
        #print 'cum_gate size: ' + str(cum_gate.size())

        cum_gate = torch.cat([cum_gate, g], dim=1)
        #print 'cum_gate size: ' + str(cum_gate.size())
        gate_hat = torch.stack([cum_gate[:, i:i + ntimestep] for i in range(self.nslots, 0, -1)],
                               dim=2)  # bsz, ntimestep, nslots

        #print 'gate_hat size: ' + str(gate_hat.size())
        
        if self.hard:
            memory_gate = (F.hardtanh((g[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
        else:
            memory_gate = F.sigmoid(
                (g[:, :, None] - gate_hat) / self.resolution * 10 + 5)  # bsz, ntimestep, nslots
        memory_gate = torch.cumprod(memory_gate, dim=2)  # bsz, ntimestep, nlookback+1
        memory_gate = torch.unbind(memory_gate, dim=1)

        if self.hard:
            memory_gate_next = (F.hardtanh((g_next[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
        else:
            memory_gate_next = F.sigmoid(
                (gnext[:, :, None] - gate_hat) / self.resolution * 10 + 5)  # bsz, ntimestep, nslots
        memory_gate_next = torch.cumprod(memory_gate_next, dim=2)  # bsz, ntimestep, nlookback+1
        memory_gate_next = torch.unbind(memory_gate_next, dim=1)

        return (memory_gate, memory_gate_next), g, (hidden_n, cell_n, cum_gate[:, -self.nslots:])

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        self.ones = Variable(weight.new(bsz, 1).zero_() + 1)
        return  Variable(weight.new(2, bsz, self.nhid).zero_()), \
                Variable(weight.new(2, bsz, self.nhid).zero_()), \
                Variable(weight.new(bsz, self.nslots).zero_() + numpy.inf)