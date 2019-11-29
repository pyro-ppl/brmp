import argparse
import operator
from functools import reduce, partial
import json
import os
from pprint import pprint

from scipy.stats import gaussian_kde
import pandas as pd
import torch.optim as optim

from brmp.numpyro_backend import backend as numpyro
from brmp.design import metadata_from_df
from brmp.oed import SequentialOED
from brmp.priors import Prior
from brmp.family import Normal, HalfNormal

from brmp.oed import possible_values  # TODO: Perhaps this ought to be moved to design.py?
from brmp.oed.example import collect_plot_data  # , make_training_data_plot

# An attempt at an implementation of the "Real Data Simulation" idea.

# We assume that the "real data" has one column which indicates the
# "participant", another containing the participants' response, and
# one or more columns that describe the design (or trial, condition,
# etc.).

# The OED selects trails from those actually run for each participant.
# Whether this makes sense will be situation specific. If all
# participants faced all possible trials then this is fine. It's also
# probably fine if each participant faced a randomly selected subset
# of all possible trials. (OED may have done better if given the
# freedom to select from all trials, but without the actual results of
# doing so we can't run the simulation.) OTOH, if trials were selected
# by some other means then it's less clear whether the simulation
# tells us anything interesting.

# Q: In what ways should this be relaxed?
#
# * Allow additional columns that aren't part of the design space?
#   e.g. Maybe there's a column that records the outside temperature
#   at the moment a response was collected. (That this can't already
#   be handled is a restriction is coming from `brmp.oed`. It assumes
#   that all columns mentioned in the model formula are categorical,
#   and that they combine to form the design space. Without thinking
#   more carefully, I'm not sure what is involved in relaxing this.)
#
# * Would it be useful to relax (in some as yet unspecified way) the
#   assumption that there is a single "participant" column?
#


def run_simulation(df, M, formula_str, priors,
                   target_coefs, response_col, participant_col, design_cols,
                   interval_method, num_samples, num_epochs, opt_method, lr, weight_decay, use_oed=True):

    df_metadata = metadata_from_df(df)
    participants = possible_values(df_metadata.column(participant_col))

    # Ensure we have enough data to run the number of requested trials
    # per participant.
    for participant in participants:
        participant_rows = df[df[participant_col] == participant]
        assert M <= len(participant_rows), 'too few rows for participant "{}" with M={}'.format(participant, M)

    print('==============================')
    print('Real data:')
    print(df.head())
    print('------------------------------')
    print('Participants: {}'.format(participants))

    # Begin simulation.
    # ----------------------------------------

    # Set-up a new OED. We are required to give a formula and to describe
    # the shape of the data we'll collect.
    oed = SequentialOED(
        formula_str,
        df_metadata.columns,
        priors=priors,
        target_coefs=target_coefs,
        num_samples=num_samples,
        num_epochs=num_epochs,
        backend=numpyro,
        use_cuda=bool(os.environ.get('OED_USE_CUDA', 0)))

    opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
    if opt_method == 'SGD':
        opt_kwargs['momentum'] = 0.9
    optimizer = partial(optim.__dict__[opt_method], **opt_kwargs)

    all_eigs = []

    for participant in participants:

        # Determine the trials that were actually run for this
        # participant.
        participant_rows = df[df[participant_col] == participant]
        actual_trials = set(zip(*[participant_rows[col] for col in design_cols]))

        # Track the designs/trials run by OED for the current participant.
        run_so_far = set()

        print('==============================')
        print('Participant: {}'.format(participant))

        for i in range(M):

            not_yet_run = actual_trials - run_so_far
            next_design_space = [dict(zip(design_cols, d), **{participant_col: participant})
                                 for d in not_yet_run]

            if use_oed:
                next_trial, dstar, eigs, fit, plot_data = oed.next_trial(
                    design_space=next_design_space,
                    callback=collect_plot_data,
                    interval_method=interval_method,
                    q_net='independent',
                    optimizer=optimizer,
                    verbose=True)
                all_eigs.append(eigs)

                pprint(sorted(eigs, key=lambda pair: pair[1], reverse=True))
                # make_training_data_plot(plot_data)

            else:
                next_trial = oed.random_trial(design_space=next_design_space)

            # Look up this trial in the real data, and extract the response given.
            ix = reduce(operator.and_, (df[col] == next_trial[col] for col in design_cols + [participant_col]))
            rows = df[ix]
            assert len(rows) == 1
            response = rows[response_col].tolist()[0]

            run_so_far.add(tuple(next_trial[col] for col in design_cols))
            oed.add_result(next_trial, response)

            print('------------------------------')
            print('OED selected trial: {}'.format(next_trial))
            print('Real response was: {}'.format(response))
            print('Data so far:')
            print(oed.data_so_far)

    return oed, all_eigs


def kde(fit, coef):
    return gaussian_kde(fit.get_scalar_param(coef))


def main(args):

    M = args.m

    df = pd.read_csv('rds.csv', index_col=0)
    df['p'] = pd.Categorical(df['p'])
    df['z'] = pd.Categorical(df['z'])

    formula_str = 'y ~ 1 + x + z + (1 + x || p)'
    priors = [Prior(('b',), Normal(0., 5.)),
              Prior(('sd',), HalfNormal(2.)),
              Prior(('resp', 'sigma'), HalfNormal(.5))]
    target_coef = 'b_z[b]'

    oed, eigs = run_simulation(
        df,
        M,
        formula_str,
        priors,
        response_col='y',
        participant_col='p',
        design_cols=['x', 'z'],
        target_coefs=[target_coef],
        interval_method=args.interval_method,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        opt_method=args.opt_method,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_oed=not args.rand)

    # Compute the Bayes factor. (We avoid defining the model
    # using `selected_trials` since initially that data frame
    # will not include e.g. all categorical levels present in
    # the full data frame, therefore a different model will be
    # fit.)

    # Compute prior density.
    num_bf_samples = 2000
    prior_fit = oed.model.run_algo('prior', oed.model.encode(df),
                                   num_samples=num_bf_samples, seed=None)
    prior_density = kde(prior_fit, target_coef)(0)

    oed_fit = oed.model.run_algo('nuts', oed.model.encode(oed.data_so_far),
                                 iter=num_bf_samples, warmup=num_bf_samples // 2,
                                 num_chains=1, seed=None)
    oed_density = kde(oed_fit, target_coef)(0)
    oed_bayes_factor, = oed_density / prior_density

    try:
        with open('results/results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    args_dict = vars(args)
    if args.rand:
        # All the other args. are ignored in this case, so reflect that in `name`.
        name = 'rand_{}'.format(M)
    else:
        name = '_'.join(str(args_dict[k]) for k in sorted(args_dict.keys()))

    if name not in results:
        results[name] = []
    i = len(results[name])
    results[name].append(oed_bayes_factor)
    with open('results/results.json', 'w') as f:
        json.dump(results, f)
    with open('results/selected_trials__{}__{}.csv'.format(name, i), 'w') as f:
        oed.data_so_far.to_csv(f)
    with open('results/eigs__{}__{}.json'.format(name, i), 'w') as f:
        json.dump(eigs, f)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real data simulation')
    parser.add_argument('-m', type=int, required=True,
                        help='number of trials to perform with each participant')
    parser.add_argument('--interval-method', type=str, choices=['fixed', 'quantile', 'adapt'], default='fixed',
                        help='method used to select OED target interval')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='number of posterior samples to take at each OED step')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of epochs of optimisation to perform at each OED step')
    parser.add_argument('--opt-method', type=str, default='Adam',
                        help='optimisation method')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--rand', action='store_true', default=False,
                        help='select trials uniformly at random rather than using OED')
    args = parser.parse_args()
    print(args)
    main(args)
