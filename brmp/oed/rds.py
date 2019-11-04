import operator
from itertools import product
from functools import reduce
from collections import defaultdict

from scipy.stats import gaussian_kde
import pandas as pd

from brmp import defm
from brmp.numpyro_backend import backend as numpyro
from brmp.design import metadata_from_df
from brmp.oed import SequentialOED
from brmp.priors import Prior
from brmp.family import Normal

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
                   target_coef, response_col, participant_col, design_cols):

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

            next_trial, dstar, eigs, fit, plot_data = oed.next_trial(
                design_space=next_design_space,
                callback=collect_plot_data,
                verbose=True)
            print(eigs)

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

    return oed.data_so_far


def kde(fit, coef):
    return gaussian_kde(fit.get_scalar_param(coef))


def main():

    # These data were generated from a model something like `y ~ 1 + x
    # + (1 + x || p)`. Note that `z` does not appear.
    df = pd.read_csv('rds.csv', index_col=0)
    df['p'] = pd.Categorical(df['p'])
    df['z'] = pd.Categorical(df['z'])

    formula_str = 'y ~ 1 + x + z + (1 + x || p)'
    priors = [Prior(('b',), Normal(0., 5.))]
    target_coef = 'b_z[b]'

    # Compute the Bayes factor on the full data using Savage-Dickey.
    # (> 1 supports nested model, < 1 supports full model.) We use KDE
    # to estimate densities, but log splines seem to be preferred in
    # the literature.
    print('------------------------------')
    print('Computing Bayes factor on full data...')

    model = defm(formula_str, df, priors=priors).generate(numpyro)
    prior_fit = model.prior(num_samples=2000)
    posterior_fit = model.nuts(iter=2000)
    prior_density = kde(prior_fit, target_coef)(0)
    posterior_density = kde(posterior_fit, target_coef)(0)
    full_bayes_factor, = posterior_density / prior_density
    print('Bayes factor on full data: {}'.format(full_bayes_factor))

    selected_trials = run_simulation(
        df,
        2,  # Number of trials per participant.
        formula_str,
        priors,
        target_coef='b_z[b]',
        response_col='y',
        participant_col='p',
        design_cols=['x', 'z'])

    # Compute the Bayes factor on the trials selected by OED.
    model = defm(formula_str, selected_trials, priors=priors).generate(numpyro)
    oed_fit = model.nuts(iter=2000)
    oed_density = kde(oed_fit, target_coef)(0)
    oed_bayes_factor, = oed_density / prior_density
    print('Bayes factor on OED selected trials: {}'.format(oed_bayes_factor))


def run_many():
    # Here we run multiple simulations for varying M and collect the
    # final Bayes factors.

    df = pd.read_csv('rds.csv', index_col=0)
    df['p'] = pd.Categorical(df['p'])
    df['z'] = pd.Categorical(df['z'])

    formula_str = 'y ~ 1 + x + z + (1 + x || p)'
    priors = [Prior(('b',), Normal(0., 5.))]
    target_coef = 'b_z[b]'

    # Compute prior density.
    model = defm(formula_str, df, priors=priors).generate(numpyro)
    prior_fit = model.prior(num_samples=2000)
    prior_density = kde(prior_fit, target_coef)(0)

    results = defaultdict(list)

    for _ in range(1):  # Repeat simulation multiple times for each M.
        for M in range(1, 12+1):
            selected_trials = run_simulation(
                df,
                M,
                formula_str,
                priors,
                target_coef='b_z[b]',
                response_col='y',
                participant_col='p',
                design_cols=['x', 'z'])

            # Compute the Bayes factor.
            model = defm(formula_str, selected_trials, priors=priors).generate(numpyro)
            oed_fit = model.nuts(iter=2000)
            oed_density = kde(oed_fit, target_coef)(0)
            oed_bayes_factor, = oed_density / prior_density
            results[M].append(oed_bayes_factor)
            print(results)


if __name__ == '__main__':
    main()
