import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import one_hot


class QIndep(nn.Module):
    def __init__(self, num_coef, num_designs):
        super(QIndep, self).__init__()
        assert type(num_coef) == int
        assert type(num_designs) == int
        assert num_coef > 0
        assert num_designs > 0
        self.num_coef = num_coef
        self.num_designs = num_designs
        # TODO: The functions we're asking the networks to learn are
        # not very complicated, so we can likely get away with
        # networks with fewer parameters.
        self.net = nn.Sequential(BatchLinear(num_designs, 1, 50),
                                 nn.ELU(),
                                 BatchLinear(num_designs, 50, num_coef),
                                 nn.Sigmoid())

    @staticmethod
    def encode(targets):
        return targets.float()

    def forward(self, inputs):
        assert inputs.shape[-1] == 1
        # TODO: There's probably a better approach than clamping --
        # parameterize loss by logits?
        eps = 1e-6
        return self.net(inputs).clamp(eps, 1-eps)

    # Compute (vectorized over y/m pair and designs) q(m|y;d).
    # m: targets
    # y: inputs
    # d: design
    def logprobs(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == targets.shape[1]
        N = inputs.shape[1]
        assert inputs.shape == (self.num_designs, N, 1)
        assert targets.shape == (self.num_designs, N, self.num_coef)
        probs = self.forward(inputs)
        return torch.sum(targets*torch.log(probs) + (1-targets)*torch.log(1-probs), -1)

    def marginal_probs(self, inputs):
        return self.forward(inputs)


# e.g. bits2long(torch.tensor([[[0,0,0], [0,0,1]],
#                              [[0,1,0], [1,1,1]]]))
# =>
# tensor([[0, 1],
#         [2, 7]])
#
def bits2long(t):
    batch_dims = t.shape[0:-1]
    width = t.shape[-1]
    powers_of_two = torch.tensor([2**i for i in range(width-1, -1, -1)])
    out = torch.sum(t * powers_of_two, -1)
    assert out.shape == batch_dims
    return out


# e.g. int2bits(3,4) => [0,0,1,1]
def int2bits(i, width):
    assert i < 2**width
    return [int(b) for b in ('{:0'+str(width)+'b}').format(i)]


# All of the target values (as bit vectors) that satisfy \theta_coef == 1.
#
# e.g.
# target_values_for_marginal(0, 3)
# =>
# tensor([[1, 0, 0],
#         [1, 0, 1],
#         [1, 1, 0],
#         [1, 1, 1]])
#
# target_values_for_marginal(1, 3)
# =>
# tensor([[0, 1, 0],
#         [0, 1, 1],
#         [1, 1, 0],
#         [1, 1, 1]])
#
def target_values_for_marginal(coef, num_coef):
    values = [bits for bits in (int2bits(i, num_coef) for i in range(2**num_coef)) if bits[coef] == 1]
    return torch.tensor(values)


# e.g.
# bits2onehot(torch.tensor([[[0,0,0], [0,0,1]]
#                           [[0,1,0], [1,1,1]]]))
# =>
# tensor([[[1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0]],
#
#         [[0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1]]])
#
def bits2onehot(t):
    width = t.shape[-1]
    return one_hot(bits2long(t), 2**width)


class QFull(nn.Module):
    def __init__(self, num_coef, num_designs):
        super(QFull, self).__init__()
        assert type(num_coef) == int
        assert type(num_designs) == int
        assert num_coef > 0
        assert num_designs > 0
        self.num_coef = num_coef
        self.num_designs = num_designs
        self.net = nn.Sequential(BatchLinear(num_designs, 1, 50),
                                 nn.ELU(),
                                 BatchLinear(num_designs, 50, 2**num_coef),
                                 nn.LogSoftmax(dim=-1))

    # Encode raw targets in the format output by the network.
    @staticmethod
    def encode(targets):
        return bits2onehot(targets).float()

    def forward(self, inputs):
        assert len(inputs.shape) == 3
        N = inputs.shape[1]
        assert inputs.shape == (self.num_designs, N, 1)
        return self.net(inputs)

    def logprobs(self, inputs, targets_enc):
        assert len(targets_enc.shape) == 3
        assert inputs.shape[1] == targets_enc.shape[1]
        N = inputs.shape[1]
        assert targets_enc.shape == (self.num_designs, N, 2 ** self.num_coef)
        logprobs = self.forward(inputs)
        assert logprobs.shape == (self.num_designs, N, 2 ** self.num_coef)
        return torch.sum(logprobs * targets_enc, -1)

    def marginal_probs(self, inputs):
        logprobs = self.forward(inputs)
        # e.g. For num_coefs == 3, `cols` will be:
        # tensor([[4, 5, 6, 7],
        #         [2, 3, 6, 7],
        #         [1, 3, 5, 7]])
        cols = torch.stack([bits2long(target_values_for_marginal(i, self.num_coef)) for i in range(self.num_coef)])
        return torch.sum(torch.exp(logprobs[..., cols]), -1)


# Note that if (a degenerate configuration of) this is used to compare
# the performance of vectorizing over designs vs. using separate nets
# then I ought to revert `forward` to using `matmul` and comment out
# the addition of the bias for the separate nets case. This is because
# `nn.Linear` uses `addmm` internally, which is faster the adding the
# bias separately, hence commenting out makes for a fairer test. (The
# existence of `baddbmm` does't change the proceeding.)
class BatchLinear(nn.Module):
    def __init__(self, batch_size, in_features, out_features):
        super(BatchLinear, self).__init__()
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.empty(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(batch_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Apply the init. from nn.Linear to each sub-network.
        for i in range(self.batch_size):
            init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias[i], -bound, bound)

    # Given an input of shape `(batch_size, N, in_features)`, this
    # returns a tensor with shape `(batch_size, N, out_features)`.
    def forward(self, inp):
        # return torch.matmul(inp, self.weight) + self.bias
        return torch.baddbmm(self.bias, inp, self.weight)


def main():
    pass


if __name__ == '__main__':
    main()
