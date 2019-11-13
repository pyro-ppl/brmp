import operator
from itertools import product
from functools import reduce
import json
import sys

from scipy.stats import gaussian_kde
import pandas as pd

from brmp import defm
from brmp.numpyro_backend import backend as numpyro
from brmp.design import metadata_from_df, makedata
from brmp.oed import SequentialOED
from brmp.priors import Prior
from brmp.family import Normal, HalfNormal
from brmp.fit import Fit
from brmp.backend import data_from_numpy

from brmp.oed import possible_values  # TODO: Perhaps this ought to be moved to design.py?
from brmp.oed.example import collect_plot_data  # , make_training_data_plot

# An attempt at an implementation of the "Real Data Simulation" idea.

# We assume that the "real data" has one column which indicates the
# "participant", another containing the participants' response, and
# one or more columns that describe the design (or trial, condition,
# etc.). We further assume that each participant has taken part in all
# possible trials. (The set of all possible trials is assumed to be
# the product of the sets of possible values taken by each of the
# design columns.)


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
# * Allow the design space to be something other than the full product
#   of the levels/possible values of the design columns. (See
#  `all_designs` below.)

def run_simulation(df, M, formula_str, priors,
                   target_coef, response_col, participant_col, design_cols,
                   use_oed=True, fixed_target_interval=True):

    df_metadata = metadata_from_df(df)
    participants = possible_values(df_metadata.column(participant_col))
    all_designs = set(product(*[possible_values(df_metadata.column(col)) for col in design_cols]))

    # We assume that the data frame contains exactly one response for each
    # participant/trial pair:
    actual_trials = zip(*[df[col] for col in [participant_col] + design_cols])
    expected_trials = [(p,) + t for p, t in product(participants, all_designs)]
    assert sorted(actual_trials) == sorted(expected_trials), \
        "Data frame doesn't include responses for the expected trials."

    # M is the number of trials to run with each participant in the
    # simulation.
    N = len(all_designs)
    assert M <= N

    print('==============================')
    print('Real data:')
    print(df.head())
    print('------------------------------')
    print('Participants: {}'.format(participants))
    print('All designs:  {}'.format(all_designs))
    print('Will simulate running {} of {} designs per participant.'.format(M, N))

    # Begin simulation.
    # ----------------------------------------

    # Set-up a new OED. We are required to give a formula and to describe
    # the shape of the data we'll collect.
    oed = SequentialOED(
        formula_str,
        df_metadata.columns,
        priors=priors,
        target_coefs=[target_coef],
        num_samples=2000,
        backend=numpyro)

    for participant in participants:

        # Track the designs/trials run with `participant` so far.
        run_so_far = set()

        print('==============================')
        print('Participant: {}'.format(participant))

        for i in range(M):

            not_yet_run = all_designs - run_so_far
            next_design_space = [dict(zip(design_cols, d), **{participant_col: participant})
                                 for d in not_yet_run]

            if use_oed:
                next_trial, dstar, eigs, fit, plot_data = oed.next_trial(
                    design_space=next_design_space,
                    callback=collect_plot_data,
                    fixed_target_interval=fixed_target_interval,
                    verbose=True)
                print(eigs)
            else:
                next_trial = oed.random_trial(design_space=next_design_space)

            # make_training_data_plot(plot_data)

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

    return oed


def kde(fit, coef):
    return gaussian_kde(fit.get_scalar_param(coef))


def main(name, M):

    df = pd.read_csv('rds.csv', index_col=0)
    df['p'] = pd.Categorical(df['p'])
    df['z'] = pd.Categorical(df['z'])

    formula_str = 'y ~ 1 + x + z + (1 + x || p)'
    priors = [Prior(('b',), Normal(0., 5.)),
              Prior(('sd',), HalfNormal(2.)),
              Prior(('resp', 'sigma'), HalfNormal(.5))]
    target_coef = 'b_z[b]'

    conditions = dict(
        oed=dict(use_oed=True, fixed_target_interval=True),
        oed_alt=dict(use_oed=True, fixed_target_interval=False),
        rand=dict(use_oed=False))
    kwargs = conditions[name]

    oed = run_simulation(
        df,
        M,
        formula_str,
        priors,
        target_coef='b_z[b]',
        response_col='y',
        participant_col='p',
        design_cols=['x', 'z'],
        **kwargs)

    # Compute the Bayes factor. (We avoid defining the model
    # using `selected_trials` since initially that data frame
    # will not include e.g. all categorical levels present in
    # the full data frame, therefore a different model will be
    # fit.)

    # Compute prior density.
    num_bf_samples = 2000
    model = defm(formula_str, df, priors=priors).generate(numpyro)
    prior_fit = model.prior(num_samples=num_bf_samples)
    prior_density = kde(prior_fit, target_coef)(0)

    # TODO: Make it easier to build models from metadata.
    dsf = data_from_numpy(oed.backend,
                          makedata(oed.formula, oed.data_so_far, oed.metadata, oed.contrasts))
    samples = oed.backend.nuts(dsf, oed.model, iter=num_bf_samples,
                               warmup=num_bf_samples // 2, num_chains=1, seed=None)
    oed_fit = Fit(oed.formula, oed.metadata, oed.contrasts, dsf,
                  oed.model_desc, oed.model, samples, oed.backend)
    oed_density = kde(oed_fit, target_coef)(0)
    oed_bayes_factor, = oed_density / prior_density

    try:
        with open('results/results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    if name not in results:
        results[name] = []
    i = len(results[name])
    results[name].append((M, oed_bayes_factor))
    with open('results/results.json', 'w') as f:
        json.dump(results, f)
    with open('results/selected_trials_{}_{}_{}.csv'.format(name, M, i), 'w') as f:
        oed.data_so_far.to_csv(f)

    print(results)


if __name__ == '__main__':
    name = sys.argv[1]
    M = int(sys.argv[2])
    main(name, M)
