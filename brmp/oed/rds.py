from itertools import product

import pandas as pd

from brmp.numpyro_backend import backend as numpyro
from brmp.design import RealValued, Categorical
from brmp.oed import SequentialOED

# A first attempt at an implementation of the "Real Data Simulation"
# idea.

# I'm assuming that the "real data" we have exactly fits the
# participants/questions set-up. That is, the data can be view as a
# series of participants' responses to questions, and that each of
# these (participant, question, response) occupies a single column.

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
#   assumption that there is one each of "participant" and "question"
#   columns comprising the design space?
#

df = pd.DataFrame({
    'response': [float(i) for i in range(9)],
    'question': pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']),
    'participant': pd.Categorical(['x', 'x', 'x', 'y', 'y', 'y', 'z', 'z', 'z']),
})

participants = list(df.participant.cat.categories)
questions = list(df.question.cat.categories)

# We assume that the data frame contains exactly one response for each
# participant/question pair:
actual_trials = zip(df['participant'], df['question'])
expected_trials = product(participants, questions)
assert sorted(actual_trials) == sorted(expected_trials), \
    "Data frame doesn't include responses for the expected trials."

N = len(questions)
M = 2  # The number of questions to ask each participant in the simulation.
assert M <= N

print('==============================')
print('Real data:')
print(df)
print('------------------------------')
print('Participants: {}'.format(participants))  # e.g. ['x', 'y', 'z']
print('Questions:    {}'.format(questions))     # e.g. ['a', 'b', 'c']
print('Will simulate asking each participant {} of {} questions.'.format(M, N))


# Begin simulation.
# ----------------------------------------

# Set-up a new OED. We are required to give a formula and to describe
# the shape of the data we'll collect.
oed = SequentialOED(
    'response ~ 1 + question + (1 + question | participant)',
    [RealValued('response'),
     Categorical('question', questions),
     Categorical('participant', participants),
     ],
    backend=numpyro)


for participant in participants:

    # Track the questions put to `participant` so far.
    asked_so_far = set()

    print('==============================')
    print('Participant: {}'.format(participant))

    for i in range(M):

        not_yet_asked = set(questions) - asked_so_far

        next_trial, dstar, eigs, fit, cbresult = oed.next_trial(
            # Here we restrict the design space to the product:
            # {participant} x not_yet_asked
            design_space=[(q, participant) for q in not_yet_asked],
            verbose=False)

        # Look up this trial in the real data, and extract the response given.
        next_question, next_participant = next_trial
        rows = df[(df['question'] == next_question) & (df['participant'] == next_participant)]
        assert len(rows) == 1
        response = rows['response'].tolist()[0]

        asked_so_far.add(next_question)
        oed.add_result(next_trial, response)

        print('------------------------------')
        print('OED selected question: {}'.format(next_question))
        print('Real response was: {}'.format(response))
        print('Data so far:')
        print(oed.data_so_far)
