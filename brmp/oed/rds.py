import operator
from itertools import product
from functools import reduce

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

# In this data `y` is the response, `p` the participant, and `x` and
# `z` the design/condition. The data were generated from a model
# something like `y ~ 1 + x + (1 + x || p)`. Note that `z` does not
# appear.

df = pd.read_csv('rds.csv', index_col=0)
df['p'] = pd.Categorical(df['p'])
df['z'] = pd.Categorical(df['z'])


formula_str = 'y ~ 1 + x + z + (1 + x || p)'
priors = [Prior(('b',), Normal(0., 5.))]
target_coef = 'b_z[b]'
response_col = 'y'
participant_col = 'p'
design_cols = ['x', 'z']


df_metadata = metadata_from_df(df)
participants = possible_values(df_metadata.column(participant_col))
all_designs = set(product(*[possible_values(df_metadata.column(col)) for col in design_cols]))


# We assume that the data frame contains exactly one response for each
# participant/trial pair:
actual_trials = zip(*[df[col] for col in [participant_col] + design_cols])
expected_trials = [(p,) + t for p, t in product(participants, all_designs)]
assert sorted(actual_trials) == sorted(expected_trials), \
    "Data frame doesn't include responses for the expected trials."

N = len(all_designs)
M = 2  # The number of trials to run with each participant in the simulation.
assert M <= N

print('==============================')
print('Real data:')
print(df.head())
print('------------------------------')
print('Participants: {}'.format(participants))
print('All designs:  {}'.format(all_designs))
print('Will simulate running {} of {} designs per participant.'.format(M, N))


# Compute the Bayes factor on all data using Savage-Dickey.

def target_coef_zero_density(fit):
    kde = gaussian_kde(fit.get_scalar_param(target_coef))
    return kde(0)[0]


print('------------------------------')
print('Computing Bayes factor...')

model = defm(formula_str, df, priors=priors).generate(numpyro)
prior_fit = model.prior(num_samples=2000)
post_fit = model.nuts(iter=2000)

# Bayes factor:
# (> 1 supports nested model, < 1 supports full model)
prior_density = target_coef_zero_density(prior_fit)
fullbf = target_coef_zero_density(post_fit) / prior_density
print('Bayes factor on full data: {}'.format(fullbf))


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

bfs = []
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

        # We might want separate out evaluation from OED, since we
        # might want to carefully evaluate (i.e. use lots of samples)
        # the use of fewer samples for OED. Perhaps I ought to
        # eventually perform evaluation as a post-processing step.
        # (This could remain as a guide though.)

        # Expected to be ~1 initially.
        bf = target_coef_zero_density(fit) / prior_density
        print('Bayes factor on data-so-far: {}'.format(bf))
        bfs.append(bf)

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

# Compute final Bayes factor.
fit = defm(formula_str, oed.data_so_far, priors=priors).generate(numpyro).nuts(iter=2000)
bf = target_coef_zero_density(fit) / prior_density
print('Bayes factor on data-so-far: {}'.format(bf))
bfs.append(bf)
print(bfs)
