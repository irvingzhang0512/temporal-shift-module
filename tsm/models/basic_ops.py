import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    consensus_type = None
    dim = None
    shape = None

    def __init__(self, consensus_type, dim=1):
        SegmentConsensus.consensus_type = consensus_type
        SegmentConsensus.dim = dim
        SegmentConsensus.shape = None

    @staticmethod
    def forward(ctx, input_tensor):
        SegmentConsensus.shape = input_tensor.size()
        if SegmentConsensus.consensus_type == 'avg':
            output = input_tensor.mean(dim=SegmentConsensus.dim, keepdim=True)
        elif SegmentConsensus.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if SegmentConsensus.consensus_type == 'avg':
            grad_in = grad_output.expand(
                SegmentConsensus.shape) / \
                float(SegmentConsensus.shape[SegmentConsensus.dim])
        elif SegmentConsensus.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if \
            consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim).apply(input)
