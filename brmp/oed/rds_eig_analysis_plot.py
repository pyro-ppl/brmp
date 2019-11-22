import json
import sys

import numpy as np

from matplotlib import pyplot as plt


def main(fn, selected_trials_fn, eigs_fn):

    otherix = 1  # The position of the trial with which to compare the best trial.

    plt.figure(figsize=(10, 12))

    for step in range(20):

        plt.subplot(5, 4, step+1)
        plot_one(fn, selected_trials_fn, eigs_fn, step, otherix)

        if step % 4 == 0:
            plt.ylabel('eig - {}th best trial'.format(otherix))
        if step >= 16:
            plt.xlabel('eig - best trial')

    plt.tight_layout()
    plt.show()
    # plt.savefig('fig4.pdf')


def plot_one(fn, selected_trials_fn, eigs_fn, step, otherix):

    # Sanity check that the selected trials and eigs file come from
    # the same run.
    i0 = selected_trials_fn.split('.')[0].split('_')[-1]
    i1 = eigs_fn.split('.')[0].split('_')[-1]
    assert i0 == i1, 'input file look like they come from different trials'
    i = i0

    # df = pd.read_csv(selected_trials_fn)
    # df['p'] = pd.Categorical(df['p'])
    # df['z'] = pd.Categorical(df['z'])

    # print(df)

    # EIGS from call to `next_trial` at each step during the actual run.
    with open(eigs_fn) as f:
        eigs = json.load(f)
    step_eigs = sorted(eigs[step], reverse=True, key=lambda pair: pair[1])
    # print(step_eigs)

    # assert df.iloc[step]['x'] == step_eigs[0][0]['x']
    # assert df.iloc[step]['z'] == step_eigs[0][0]['z']

    # Multiple runs of `next_trial` at `step`, produced by `analyse_step.py`.
    with open('results/eigs_{}_step_{}.json'.format(i, step)) as f:
        all_eigs = json.load(f)

    # eigs for trial selected during actual run.
    eigs_winner = all_eigs['{}_{}'.format(step_eigs[0][0]['x'], step_eigs[0][0]['z'])]
    # eigs for trial judged second best during actual run.
    eigs_other = all_eigs['{}_{}'.format(step_eigs[otherix][0]['x'], step_eigs[otherix][0]['z'])]

    # print(eigs0)
    # print(eigs1)

    plt.plot([-1, 1], [-1, 1], c='lightgray')
    plt.scatter(eigs_winner, eigs_other, marker='.')
    plt.scatter([step_eigs[0][1]], [step_eigs[otherix][1]], marker='.')

    lim_lo = np.min([np.min(eigs_winner), np.min(eigs_other), step_eigs[0][1], step_eigs[otherix][1]])
    lim_hi = np.max([np.max(eigs_winner), np.max(eigs_other), step_eigs[0][1], step_eigs[otherix][1]])

    lim_range = lim_hi - lim_lo
    lim_margin = lim_range * 0.05

    lim_lo = lim_lo - lim_margin
    lim_hi = lim_hi + lim_margin

    plt.xlim(lim_lo, lim_hi)
    plt.ylim(lim_lo, lim_hi)

    trial_winner = '({},{})'.format(step_eigs[0][0]['x'], step_eigs[0][0]['z'])
    trial_other = '({},{})'.format(step_eigs[otherix][0]['x'], step_eigs[otherix][0]['z'])

    plt.title('step={}\n{} vs. {}'.format(step, trial_winner, trial_other), fontsize=10)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
