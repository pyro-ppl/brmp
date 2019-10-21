import torch
import numpy as np
from matplotlib import pyplot as plt

from brmp.design import RealValued, Categorical
from brmp.priors import Prior
from brmp.family import Normal, HalfNormal
from brmp.oed import SequentialOED


def get_float_input(msg):
    try:
        return float(input(msg))
    except ValueError:
        return get_float_input(msg)


# Callback used to compute the data required to make a picture of the
# training data seen by the networks during OED.
def collect_plot_data(q_net, inputs, targets, design_space, target_coefs):
    # inputs (D, N, 1)
    # targets (D, N, num_coefs) replicated across designs

    num_designs = inputs.shape[0]
    num_coefs = targets.shape[-1]

    # Build test inputs at which to sample the network output.
    tis = []
    for i in range(num_designs):
        imin = inputs[i].min()  # Test each design over the range spanned by the inputs.
        imax = inputs[i].max()
        ti = torch.linspace(imin, imax, 50).unsqueeze(0)
        tis.append(ti)
    test_inputs = torch.cat(tis, 0).unsqueeze(-1)

    # Compute the marginal prob of each coef on the test inputs.
    test_out = q_net.marginal_probs(test_inputs).detach()

    # Assemble a nested array of the relevant data for plotting.
    out = []
    for i in range(num_designs):
        row = []
        for j in range(num_coefs):
            neg_cases = inputs[i, targets[:, j] == 0]
            pos_cases = inputs[i, targets[:, j] == 1]
            row.append((pos_cases.numpy(),
                        neg_cases.numpy(),
                        test_inputs[i].numpy(),
                        test_out[i, :, j].numpy(),
                        design_space[i],
                        target_coefs[j]))
        out.append(row)

    return out


# Here's a sketch of how to make plots showing the QFull's output on
# all 2**M outputs.

# for k in range(2**num_coefs):

#     pos_cases = inputs_d[bits2long(targets_d) == k]
#     neg_cases = inputs_d[bits2long(targets_d) != k]

#     # Vis. the function implemented by the net.
#     imin = inputs_d.min()
#     imax = inputs_d.max()
#     test_in = torch.arange(imin, imax, (imax-imin)/50.).reshape(-1, 1)
#     test_out = torch.exp(q_net.logprobs(
#         test_in,
#         torch.tensor(int2bits(k,num_coefs)).expand(test_in.shape[0],-1)).detach())

#     plot_data[j][k] = (pos_cases.numpy(), neg_cases.numpy(), test_in.numpy(), test_out.numpy(), design)


def make_training_data_plot(plot_data):
    plt.figure(figsize=(12, 12))
    for j, row in enumerate(plot_data):
        for k, (pos_cases, neg_cases, test_in, test_out, design, coef) in enumerate(row):
            plt.subplot(len(plot_data), len(row), (j*len(row) + k)+1)
            if j == 0:
                plt.title('coef={}'.format(coef))
            if k == 0:
                plt.ylabel('q(m|y;d={})'.format(design))
            plt.xlabel('y')
            plt.scatter(neg_cases, np.random.normal(0, 0.01, neg_cases.shape),
                        marker='.', alpha=0.15, label='coef. not nr. zero')
            plt.scatter(pos_cases, np.random.normal(1, 0.01, pos_cases.shape),
                        marker='.', alpha=0.15, label='coef. nr. zero')
            plt.ylim((-0.1, 1.1))
            plt.plot(test_in, test_out, color='gray', label='q(m|y;d)')
            # plt.legend()
    plt.show()


def main():

    oed = SequentialOED(
        'y ~ 1 + a + b',
        [RealValued('y'),
         Categorical('a', ['a1', 'a2']),
         Categorical('b', ['b1', 'b2'])],
        priors=[
            Prior(('b',),            Normal(0., 1.)),
            Prior(('resp', 'sigma'), HalfNormal(.2)),
        ])

    for _ in range(1000):
        design, dstar, eigs, fit, plot_data = oed.next_trial(callback=collect_plot_data, verbose=True)
        print(fit.marginals())
        print('EIGs:')
        print(eigs)
        print('Next trial: {}'.format(design))
        make_training_data_plot(plot_data)
        result = get_float_input('Enter result: ')
        oed.add_result(design, result)


if __name__ == '__main__':
    main()
