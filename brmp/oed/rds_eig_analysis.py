import os
import sys
from collections import defaultdict
import json
from functools import partial

import torch.optim as optim
import pandas as pd

from brmp.oed import SequentialOED
from brmp.design import metadata_from_df
from brmp.priors import Prior
from brmp.family import Normal, HalfNormal
from brmp.numpyro_backend import backend as numpyro

# Take a "selected trials" CSV and a step number as input.

# Re-run `next_trial` on the data-so-far at that point multiple times
# and collect the resulting EIG estimates for further analysis.


def main(fn, selected_trials_fn, eigs_fn, step):

    # fn:                 file name of full data
    # selected_trials_fn: file name of selected trials CSV in a particular run
    # eigs_fn:            the EIGs for each trial considered at each step of the particular run
    # step:               the step to analyse. i.e. the step for which `next_trial`
    #                     should be repeated for later analysis.

    # Sanity check that the selected trials and eigs file come from
    # the same run.
    i0 = selected_trials_fn.split('.')[0].split('_')[-1]
    i1 = eigs_fn.split('.')[0].split('_')[-1]
    assert i0 == i1, 'input file look like they come from different trials'
    i = i0

    df = pd.read_csv(fn)
    df['p'] = pd.Categorical(df['p'])
    df['z'] = pd.Categorical(df['z'])
    df_metadata = metadata_from_df(df)

    assert type(step) == int and step >= 0 and step < len(df)

    formula_str = 'y ~ 1 + x + z + (1 + x || p)'
    priors = [Prior(('b',), Normal(0., 5.)),
              Prior(('sd',), HalfNormal(2.)),
              Prior(('resp', 'sigma'), HalfNormal(.5))]
    target_coef = 'b_z[b]'

    oed = SequentialOED(
        formula_str,
        df_metadata.columns,
        priors=priors,
        target_coefs=[target_coef],
        num_samples=6000,
        num_epochs=300,
        backend=numpyro,
        use_cuda=bool(os.environ.get('OED_USE_CUDA', 0)))
    selected_trials = pd.read_csv(selected_trials_fn)
    selected_trials['p'] = pd.Categorical(selected_trials['p'])
    selected_trials['z'] = pd.Categorical(selected_trials['z'])

    oed.data_so_far = selected_trials[0:step]

    participant = 'p{}'.format(step // 2)

    run_so_far_df = oed.data_so_far[oed.data_so_far['p'] == participant]
    assert len(run_so_far_df) <= 1
    run_so_far = (set() if len(run_so_far_df) == 0
                  else set([(run_so_far_df.iloc[0]['x'],
                             run_so_far_df.iloc[0]['z'])]))

    participant_col = 'p'
    design_cols = ['x', 'z']

    participant_rows = df[df[participant_col] == participant]
    actual_trials = set(zip(*[participant_rows[col] for col in design_cols]))
    not_yet_run = actual_trials - run_so_far
    next_design_space = [dict(zip(design_cols, d), **{participant_col: participant})
                         for d in not_yet_run]

    print(oed.data_so_far)
    print(next_design_space)

    all_eigs = defaultdict(list)
    for _ in range(20):
        optimizer = partial(optim.SGD, momentum=0.9, lr=0.0001, weight_decay=0.1)
        _, _, eigs, _, _ = oed.next_trial(design_space=next_design_space,
                                          interval_method='adapt',
                                          q_net='independent',
                                          optimizer=optimizer,
                                          verbose=True)
        for trial, eig in eigs:
            all_eigs['{}_{}'.format(trial['x'], trial['z'])].append(eig)
    print(all_eigs)

    with open('results/eigs_{}_step_{}.json'.format(i, step), 'w') as f:
        json.dump(all_eigs, f)


if __name__ == '__main__':
    # e.g. python rds_eig_analysis.py rds.csv results/selected_trials_oed_2_0.csv results/eigs_0.json
    for step in range(20):
        main(sys.argv[1], sys.argv[2], sys.argv[3], step)
