import itertools
import time

import pandas as pd
from pandas.api.types import is_categorical_dtype

import torch
import torch.optim as optim

from brmp import makedesc
from brmp.formula import parse, OrderedSet, unique
from brmp.design import Metadata, makedata, metadata_from_cols, RealValued, Categorical, code_lengths
from brmp.family import Normal
from brmp.backend import data_from_numpy
from brmp.pyro_backend import backend as pyro_backend
from brmp.fit import Fit

from brmp.oed.nets import QIndep, QFull # noqa: 401


# Provides a convenient interface for performing sequential OED.

# Also acts as a single data structure in which to store the
# components of the model definition. (Formula, family, priors, etc.)
# The constructor takes care of the boiler plate required to set-up a
# brmp model.

# Also holds the data-so-far. This data is used when computing the
# next trial. The data-so-far can be extended with the result of an
# experiment using the `add_result` method. Note that `data_so_far` is
# the only mutable state held by instances of SequentialOED. The only
# method that modifies this is `add_result`.

# There are also methods/properties for obtaining information about
# the current sequence:

# oed.design_space()
# oed.data_so_far

def null(*args):
    return None


class SequentialOED:
    def __init__(self, formula_str, cols, family=Normal, priors=[],
                 contrasts={}, target_coefs=[], backend=pyro_backend):
        formula = parse(formula_str)
        metadata = metadata_from_cols(cols)
        model_desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))
        model = backend.gen(model_desc)
        data_so_far = empty_df_from_cols(cols)
        dscols = design_space_cols(formula, metadata)

        assert type(target_coefs) == list
        if len(target_coefs) == 0:
            target_coefs = model_desc.population.coefs
        else:
            # TODO: Move `unique` to utils or similar.
            target_coefs = unique(target_coefs)
            assert set(target_coefs).issubset(set(model_desc.population.coefs)), 'unknown target coefficient given'

        num_coefs = len(target_coefs)

        # TODO: Prefix non-public stuff with underscores?
        self.formula = formula
        self.contrasts = contrasts
        self.metadata = metadata
        self.model_desc = model_desc
        self.model = model
        self.data_so_far = data_so_far
        self.num_coefs = num_coefs
        self.target_coefs = target_coefs
        self.dscols = dscols

        self.backend = backend
        self.num_samples = 1000

    def next_trial(self, callback=None, verbose=False, **kwargs):

        if callback is None:
            callback = null

        design_space = self.design_space(**kwargs)
        design_space_df = design_space_to_df(self.dscols, design_space)

        # Code the data-so-far data frame into design matrices.
        dsf = data_from_numpy(self.backend,
                              makedata(self.formula, self.data_so_far, self.metadata, self.contrasts))

        # Draw samples from current distribution over parameters.
        if len(self.data_so_far) == 0:
            samples = self.backend.prior(dsf, self.model, num_samples=self.num_samples)
        else:
            samples = self.backend.nuts(dsf, self.model, iter=self.num_samples,
                                        warmup=self.num_samples // 2, num_chains=1)
        fit = Fit(self.formula, self.metadata, self.contrasts, dsf, self.model_desc, self.model, samples, self.backend)

        # Values sampled for (population-level) target coefs. (numpy array.)
        latent_samples = [fit.get_scalar_param('b_{}'.format(tc)) for tc in self.target_coefs]

        # Draw samples from p(y|theta;d)
        y_samples = fit.fitted('sample', design_space_df)  # numpy array.

        # All ANN work is done using PyTorch, so convert samples from
        # numpy to torch ready for what follows.
        latent_samples = torch.stack([torch.tensor(col) for col in latent_samples], 1)
        y_samples = torch.tensor(y_samples)
        assert latent_samples.shape == (self.num_samples, self.num_coefs)
        assert y_samples.shape == (self.num_samples, len(design_space))

        # Compute the targets. (These are used by all designs.)
        eps = 0.5
        targets = ((-eps < latent_samples) & (latent_samples < eps)).long()
        assert targets.shape == (self.num_samples, self.num_coefs)

        inputs = y_samples.t().unsqueeze(-1)
        assert inputs.shape == (len(design_space), self.num_samples, 1)

        # Estimate EIGs
        Q = QFull  # QIndep
        vectorize = True
        est_eig_fn = est_eig_vec if vectorize else est_eig
        eigs, cbvals, elapsed = est_eig_fn(Q, targets, inputs, design_space, self.target_coefs, callback, verbose)
        if verbose:
            print('Elapsed: {}'.format(elapsed))

        dstar = argmax(eigs)
        return design_space[dstar], dstar, list(zip(design_space, eigs)), fit, cbvals

    def add_result(self, design, result):
        self.data_so_far = extend_df_with_result(self.formula, self.metadata, self.data_so_far, design, result)

    def design_space(self, **kwargs):
        return design_space(self.dscols, self.metadata, **kwargs)


def argmax(lst):
    return torch.argmax(torch.tensor(lst)).item()


# Estimate the EIG for each design.
def est_eig(Q, targets, inputs, design_space, target_coefs, callback, verbose):
    num_coefs = targets.shape[1]
    eigs = []
    cbvals = []
    elapsed = 0.0
    targets_enc = Q.encode(targets).unsqueeze(0)
    for i, design in enumerate(design_space):
        inputs_i = inputs[i].unsqueeze(0)

        # Construct and optimised the network.
        q_net = Q(num_coefs, num_designs=1)
        t0 = time.time()
        optimise(q_net, inputs_i, targets_enc, verbose)

        eig = torch.mean(q_net.logprobs(inputs_i, targets_enc)).item()
        eigs.append(eig)
        elapsed += (time.time() - t0)

        cbvals.append(callback(q_net, inputs_i, targets, [design], target_coefs)[0])

    return eigs, cbvals, elapsed


# Estimate the EIG for each design. (Vectorized over designs.)
def est_eig_vec(Q, targets, inputs, design_space, target_coefs, callback, verbose):
    num_coefs = targets.shape[1]
    # Encode targets, and replicate for each design.
    targets_enc = Q.encode(targets).unsqueeze(0).expand(len(design_space), -1, -1)
    q_net = Q(num_coefs, len(design_space))
    t0 = time.time()
    optimise(q_net, inputs, targets_enc, verbose)
    eigs = torch.mean(q_net.logprobs(inputs, targets_enc), -1)
    elapsed = time.time() - t0
    cbvals = callback(q_net, inputs, targets, design_space, target_coefs)
    return eigs.tolist(), cbvals, elapsed


def optimise(net, inputs, targets, verbose):

    # TODO: Mini-batches. (On shuffled inputs/outputs.)
    # TODO: Note: using some weight decay probably makes sense here.

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        loss = -torch.sum(torch.mean(net.logprobs(inputs, targets), -1))
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 and verbose:
            print('{:5d} | {:.6f}'.format(i+1, loss.item()))

    if verbose:
        print('--------------------')


def empty_df_from_cols(cols):
    def emptydfcol(col):
        if type(col) == Categorical:
            return pd.Categorical([])
        elif type(col) == RealValued:
            return []
        else:
            raise Exception('encountered unsupported column type')
    return pd.DataFrame({col.name: emptydfcol(col) for col in cols})


# Extract the names of the columns/factors appearing on RHS of the
# model formula.

# TODO: This is similar to `allfactors` in formula.py -- consolidate?

def design_space_cols(formula, meta):
    cols = OrderedSet()
    for t in formula.terms:
        cols = cols.union(t.factors)
    for group in formula.groups:
        cols = cols.union(OrderedSet(*group.columns))
        for t in group.terms:
            cols = cols.union(t.factors)
    assert all(type(meta.column(c) == Categorical) for c in cols)
    return list(cols)


# This defaults to using the full Cartesian product of the columns,
# but allows individual columns to be restricted to a subset of their
# values.
def design_space(names, metadata, **levels_lookup):
    assert type(names) == list
    assert all(type(name) == str for name in names)
    assert type(metadata) == Metadata

    def levels(name):
        col = metadata.column(name)
        assert type(col) == Categorical
        if name in levels_lookup:
            vals = levels_lookup[name]
            assert set(vals).issubset(set(col.levels)), 'one of more invalid levels given for "{}"'.format(name)
            return vals
        else:
            return col.levels

    all_possible_vals = list(itertools.product(*[levels(name) for name in names]))
    return all_possible_vals


def design_space_to_df(dscols, design_space):
    return pd.DataFrame(dict((name, pd.Categorical(col))
                             for name, col in zip(dscols, list(zip(*design_space)))))


# TODO: Does it *really* take this much work to add a row to a df?
def extend_df_with_result(formula, meta, data_so_far, design, result):
    assert type(design) == tuple
    assert type(result) == float
    cols = design_space_cols(formula, meta)
    assert len(design) == len(cols)
    # This assumes that `design` is ordered following
    # `design_space_cols`.
    row = dict(zip(cols, design))
    row[formula.response] = result
    return df_append_row(data_so_far, row)


def df_append_row(df, row):
    assert type(row) == dict
    row_df = pd.DataFrame({k: pd.Categorical([v]) if is_categorical_dtype(df[k]) else [v]
                           for k, v in row.items()})
    out = df.append(row_df, sort=False)
    # Simply appending a new df produces a new df in which
    # (sometimes!?!) a column that was categorical in the two inputs
    # is not categorical in the output. This tweaks the result to
    # account for that. I don't know why this is happening.
    for k in row.keys():
        if is_categorical_dtype(df[k]) and not is_categorical_dtype(out[k]):
            out[k] = pd.Categorical(out[k])
    return out
