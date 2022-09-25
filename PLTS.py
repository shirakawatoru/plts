import pandas as pd 
import numpy as np

import scipy.optimize
from scipy.stats import norm

import random
import matplotlib.pyplot as plt

import dataclasses

from dataclasses import dataclass
from dataclasses import field
from typing import List

def sigmoid(x): return (1+np.exp(-x))**(-1)

from dataclasses import dataclass
from dataclasses import field
from typing import List

@dataclass
class Featurizer:
    n_features:int = 1
    deflate_index: List[int] = field(default=None, compare=False)

    def __post_init__(self):
        if self.deflate_index is None:
            self.deflate_index = list(range(self.n_features))

        self.inflate_index = np.repeat(-1, self.n_features)
        
        for i, k in enumerate(self.deflate_index):
            self.inflate_index[k] = i
    
    def inflate(self, x:np.ndarray):
        if x.ndim == 1:
            return np.hstack([x, 0])[...,self.inflate_index]
        elif x.ndim == 2:
            return np.hstack([x, np.zeros((len(x),1))])[...,self.inflate_index]

    def inflate_matrix(self, x:np.ndarray):
        if x.ndim == 2:
            return np.append(np.append(x, np.zeros((1,len(x))), axis=0), np.zeros((len(x)+1, 1)), axis=1)[self.inflate_index,:][:,self.inflate_index]

    def deflate(self, x:np.ndarray):
        return x[...,self.deflate_index]

@dataclass
class IdentityFeaturizer(Featurizer):
    def inflate(self, x:np.ndarray):
        return x

    def deflate(self, x:np.ndarray):
        return x

@dataclass
class IdentityFeaturizer(Featurizer):
    def inflate(self, x:np.ndarray):
        return x

    def deflate(self, x:np.ndarray):
        return x

@dataclass(frozen=True)
class Prior:
    m: np.ndarray
    S_inv: np.ndarray

    def value(self, w:np.ndarray):
        return 0.5 * (w - self.m) @ self.S_inv @ (w - self.m)

    def grad(self, w:np.ndarray):
        return self.S_inv @ (w - self.m)

    def hess(self, w:np.ndarray):
        return self.S_inv

@dataclass(frozen=True)
class NullPrior(Prior):
    def value(self, w:np.ndarray):
        return 0

    def grad(self, w:np.ndarray):
        return np.zeros(len(self.m))

    def hess(self, w:np.ndarray):
        return np.eye(len(self.m))

@dataclass(frozen=True)
class Likelihood:
    def value(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        #return - np.log(y[r==1]).sum() - np.log(1 - y[r==0]).sum()
        return - np.sum(r * np.log(y) + (1 - r) * np.log(1 - y))

    def grad(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return np.sum((y-r)[:,np.newaxis] * x, axis=0)

    def hess(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return np.tensordot(((y*(1-y))[:,np.newaxis] * x).T, x, axes=1)

@dataclass(frozen=True)
class NullLikelihood():
    def value(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return 0

    def grad(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return np.zeros(len(x[0]))

    def hess(self, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return np.eye(len(x[0]))


@dataclass(frozen=True)
class Posterior:
    p: Prior
    l: Likelihood

    def value(self, w:np.ndarray, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return self.p.value(w) + self.l.value(r, x, y) 

    def grad(self, w:np.ndarray, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return self.p.grad(w) + self.l.grad(r, x, y) 

    def hess(self, w:np.ndarray, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return self.p.hess(w) + self.l.hess(r, x, y) 

    def hess_inv(self, w:np.ndarray, r:np.ndarray, x:np.ndarray, y:np.ndarray):
        return np.linalg.inv(self.hess(w, r, x, y))

@dataclass
class Model:
    n_features: int = 5
    n_features_w: int = 5
    n_features_u: int = 1
    n_features_v: int = 5
    n_features_z: int = 1

    # before hyperparameters
    Sigma_w: np.ndarray = np.eye(n_features_w) * 1.0
    Sigma_u: np.ndarray = np.eye(n_features_u) * 10.0
    Sigma_v: np.ndarray = np.eye(n_features_v) * 10.0
    Sigma_z: np.ndarray = np.eye(n_features_z) * 5.0

    eta: callable = None

    def __post_init__(self):
        self.fw = IdentityFeaturizer(n_features = self.n_features)
        self.fu = Featurizer(n_features = self.n_features, deflate_index=[0])
        self.fv = IdentityFeaturizer(n_features = self.n_features)
        self.fz = Featurizer(n_features = self.n_features, deflate_index=[0])

        self.Sigma_u_inv = np.linalg.inv(self.Sigma_u)
        self.Sigma_v_inv = np.linalg.inv(self.Sigma_v)
        self.Sigma_z_inv = np.linalg.inv(self.Sigma_z)

    def get_param(self, w, u, v, z):
        return self.fw.inflate(w) + self.fu.inflate(u) + self.fv.inflate(v) + self.fz.inflate(z)

    def get_y(self, w, u, v, z, x, c):
        if np.ndim(x) == 1:
            ret = self.get_param(w, u, v, z) @ x
        else:
            ret = self.get_param(w, u, v, z) * x
            if ret.ndim == 2: 
                ret = ret.sum(axis=1)
        
        if self.eta is not None:
            ret += self.eta(c)

        return sigmoid(ret)


@dataclass(frozen=True)
class Data:
    i:np.ndarray
    a:np.ndarray
    r:np.ndarray
    x:np.ndarray
    c:np.ndarray

    def __post_init__(self):
        n_obs = len(self.i)

    def units(self):
        return set(self.i)    

    def actions(self):
        return set(self.a)

    def ia_pairs(self):
        return set(zip(self.i, self.a))

@dataclass
class DataGeneratingProcess:
    model: Model

    n_units:int = 100
    n_actions:int = 50

    def __post_init__(self):
        self.w = np.hstack([-2, np.random.normal(0, 1.0, self.model.n_features_w-1)])
        self.u = np.random.multivariate_normal(np.zeros(self.model.n_features_u), self.model.Sigma_u, self.n_units)
        self.v = np.random.multivariate_normal(np.zeros(self.model.n_features_v), self.model.Sigma_v, self.n_actions)
        self.z = np.random.multivariate_normal(np.zeros(self.model.n_features_z), self.model.Sigma_z, (self.n_units, self.n_actions))

    def generate(self, n_obs):
        i = np.random.randint(0, self.n_units, n_obs)
        x = np.random.normal(0, 1, (n_obs, self.model.n_features))

        a = np.zeros(n_obs, dtype=int)
        c = np.zeros((n_obs, self.n_actions), dtype=int)

        for k in range(n_obs):
            n_a  = np.random.randint(3,8)
            candidates = np.random.randint(0, self.n_actions, n_a)
            for _a in candidates:
                c[k,_a] = 1
            a[k] = np.random.choice(candidates)

        x[:,0]=1

        y = self.model.get_y(self.w, self.u[i], self.v[a], self.z[i,a], x, c)
        r = np.random.binomial(1, y, n_obs)

        return Data(i, a, r, x, c)
    
    def generate_r(self, i, a, x, c):
        l = None if np.ndim(i) == 0 else len(i)
        return np.random.binomial(1, self.model.get_y(self.w, self.u[i], self.v[a], self.z[i,a], x, c), l)
    
    def calculate_y(self, i, a, x, c):
        return self.model.get_y(self.w, self.u[i], self.v[a], self.z[i,a], x, c)

class Parameter:
    pass

@dataclass
class Parameter:
    model: Model

    units: set = field(default_factory=set)
    actions: set = field(default_factory=set)
    ia_pairs: set = field(default_factory=set)

    m: dict = field(default_factory=dict)
    S: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ia_pairs_ext = self.get_extended_ia_pairs(self.ia_pairs)

        for i, a in self.ia_pairs_ext:
            if (i, a) not in self.m:
                self.reset_hparam(i, a)

        self.sampled_param = {}
        self.sampled_param_ia = {}
        
    def update(self, par:Parameter):
        assert(par.ia_pairs_ext <= self.ia_pairs_ext)

        for k in par.ia_pairs_ext:
            self.m[k] = par.m[k]
            self.S[k] = par.S[k]

    def reset_hparam(self, i:int, a:int):
        '''Reset hyperparameters with initial priors for target_ia_pairs. If target_ia_pairs is None then reset all hyperparameters.'''
        def default_m(i, a):
            if   (i==-1) and (a==-1): return np.zeros(self.model.n_features_w)
            elif             (a==-1): return np.zeros(self.model.n_features_u)
            elif (i==-1)            : return np.zeros(self.model.n_features_v)
            else                    : return np.zeros(self.model.n_features_z)

        def default_S(i, a):
            if   (i==-1) and (a==-1): return self.model.Sigma_w.copy()
            elif             (a==-1): return self.model.Sigma_u.copy()
            elif (i==-1)            : return self.model.Sigma_v.copy()
            else                    : return self.model.Sigma_z.copy()

        self.m[i,a] = default_m(i, a)
        self.S[i,a] = default_S(i, a)
    
    def get_extended_ia_pairs(self, ia_pairs:set):
        return ia_pairs | {(-1, a) for i, a in ia_pairs} | {(i, -1) for i, a in ia_pairs} |{(-1, -1)}

    def extend(self, ia_pairs:set, sample_param=False):
        ia_pairs_diff = ia_pairs - self.ia_pairs
        ia_pairs_ext = self.get_extended_ia_pairs(ia_pairs)
        for i, a in (ia_pairs_ext - self.ia_pairs_ext):
            self.reset_hparam(i, a)
        if sample_param:
            self.sample_param(ia_pairs_diff)
        
        self.units        |= {i for i, _ in ia_pairs}
        self.actions      |= {a for _, a in ia_pairs}
        self.ia_pairs     |= ia_pairs
        self.ia_pairs_ext |= ia_pairs_ext

    def sample_param(self, ia_pairs:set = None):
        if ia_pairs is None: ia_pairs = self.ia_pairs
        ia_pairs_ext = self.get_extended_ia_pairs(ia_pairs)

        for i, a in ia_pairs_ext:
            self.sampled_param[i,a] = np.random.multivariate_normal(self.m[i,a], self.S[i,a])

        for i, a in ia_pairs:
            self.sampled_param_ia[i,a] = self.model.get_param(
                self.sampled_param[-1,-1], self.sampled_param[i,-1], self.sampled_param[-1,a], self.sampled_param[i,a]
                )

@dataclass
class PackedParameterVector:
    model: Model
    val: np.ndarray

    n_units:int
    n_actions:int
    n_ia_pairs:int

    def __post_init__(self):
        self.o = np.cumsum(np.hstack([
            0, self.model.n_features_w, 
            np.repeat(self.model.n_features_u, self.n_units), 
            np.repeat(self.model.n_features_v, self.n_actions), 
            np.repeat(self.model.n_features_z, self.n_ia_pairs)
            ]))

        self.k = np.cumsum([0, \
            self.model.n_features_w, \
            self.model.n_features_u * self.n_units, \
            self.model.n_features_v * self.n_actions, \
            self.model.n_features_v * self.n_ia_pairs \
            ])

    def __getattr__(self, name:str):
        if name == 'w':
            return self.val[0:self.model.n_features_w]
        elif name == 'u':
            return self.val[self.k[1]: self.k[2]].reshape(self.n_units, self.model.n_features_u)
        elif name == 'v':
            return self.val[self.k[2]: self.k[3]].reshape(self.n_actions, self.model.n_features_v)
        elif name == 'z':
            return self.val[self.k[3]: self.k[4]].reshape(self.n_ia_pairs, self.model.n_features_z)
        else:
            super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == 'w':
            self.val[0:model.n_features_w] = value
        else:
            super().__setattr__(name, value)

    def __getitem__(self, n):
        return self.val[self.o[n]:self.o[n+1]]

    def __setitem__(self, n, val):
        self.val[self.o[n]:self.o[n+1]] = val


@dataclass
class PackedParameterCalculator:
    model: Model
    parameter: Parameter
    data: Data

    units:list
    actions:list
    ia_pairs:list

    def __post_init__(self):
        self.i_dic = {i:n for n, i in enumerate(self.units)}
        self.ui = [self.i_dic[k] for k in self.data.i]

        self.a_dic = {i:n for n, i in enumerate(self.actions)}
        self.vi = [self.a_dic[k] for k in self.data.a]

        self.ia_dic = {i:n for n, i in enumerate(self.ia_pairs)}
        self.zi = [[self.ia_dic[k] for k in zip(self.data.i, self.data.a)]]

        S_inv = {k:np.linalg.inv(self.parameter.S[k]) for k in self.parameter.get_extended_ia_pairs(self.data.ia_pairs())}

        self.likelihood = Likelihood()

        # posteriors
        #self.qs = [Posterior(Prior(self.parameter.m[-1,-1], S_inv[-1,-1]),                          self.likelihood)] + \
        self.qs = [Posterior(NullPrior(self.parameter.m[-1,-1], S_inv[-1,-1]),                      NullLikelihood())] + \
                  [Posterior(Prior(self.parameter.m[ i,-1], S_inv[ i,-1] + self.model.Sigma_u_inv), self.likelihood) for i    in self.units] + \
                  [Posterior(Prior(self.parameter.m[-1, a], S_inv[-1, a] + self.model.Sigma_v_inv), self.likelihood) for    a in self.actions] + \
                  [Posterior(Prior(self.parameter.m[ i, a], S_inv[ i, a] + self.model.Sigma_z_inv), self.likelihood) for i, a in self.ia_pairs]

        self.w = PackedParameterVector(
            model = self.model,
            val = np.hstack([q.p.m for q in self.qs]),
            n_units   = len(self.units),
            n_actions = len(self.actions),
            n_ia_pairs  = len(self.ia_pairs),
            )

        self.update_y(self.w.val)

    def r_x_y(self, n):
        if n < 1:
            return self.data.r, self.model.fw.deflate(self.data.x), self.y
        n -= 1

        if n < len(self.units):
            i = self.units[n]
            idx = (self.data.i == i)
            return self.data.r[idx], self.model.fu.deflate(self.data.x[idx]), self.y[idx]
        n -= len(self.units)

        if n < len(self.actions):
            a = self.actions[n]
            idx = (self.data.a == a)
            return self.data.r[idx], self.model.fv.deflate(self.data.x[idx]), self.y[idx]
        n -= len(self.actions)

        if n < len(self.ia_pairs):
            i, a = self.ia_pairs[n]
            idx = ((self.data.a == a) & (self.data.i == i))
            return self.data.r[idx], self.model.fz.deflate(self.data.x[idx]), self.y[idx]

        assert(False)

    def loss(self, val):
        return self.qval(val)

    def grad(self, val):
        p = dataclasses.replace(self.w, val=val)
        return np.hstack([q.grad(p[n], *self.r_x_y(n)) for n, q in enumerate(self.qs)])

    def grad_p(self, val):
        p = dataclasses.replace(self.w, val=val)
        return np.hstack([q.p.grad(p[n]) for n, q in enumerate(self.qs)])

    def grad_l(self, val):
        self.update_y(val)
        return np.hstack([q.l.grad(*self.r_x_y(n)) for n, q in enumerate(self.qs)])

    def hess_inv(self, val):
        p = dataclasses.replace(self.w, val=val)
        k = np.cumsum([1,len(self.units), len(self.actions)])
        h = [q.hess_inv(p[n], *self.r_x_y(n)) for n, q in enumerate(self.qs)]
        return h[0], h[1:k[1]], h[k[1]:k[2]], h[k[2]:]

    def w_u_v_z(self, val): 
        p = dataclasses.replace(self.w, val=val)
        return p.w, p.u, p.v, p.z

    def expand_param(self, w, u, v, z):
        return w, u[self.ui], v[self.vi], z[self.zi]

    def update_y(self, w):
        self.y = self.model.get_y(*self.expand_param(*self.w_u_v_z(w)), self.data.x, self.data.c)

    def qval(self, val):
        '''value of posterior distribution density at w'''
        return self.pval(val) + self.lval(val)

    def pval(self, val):
        '''value of prior distribution density at w'''
        p = dataclasses.replace(self.w, val=val)
        return np.sum([q.p.value(p[n]) for n, q in enumerate(self.qs)])

    def lval(self, val):
        '''value of likelihood distribution density at w'''
        self.update_y(val)
        return self.likelihood.value(self.data.r, self.data.x, self.y)
    
    def unpack_param(self, w):
        m_w1, m_u1, m_v1, m_z1 = self.w_u_v_z(w)
        S_w1, S_u1, S_v1, S_z1 = self.hess_inv(w)

        m = {(-1,-1):m_w1}
        for i   in self.units:    m[ i,-1] = m_u1[self.i_dic[i]]
        for a   in self.actions:  m[-1, a] = m_v1[self.a_dic[a]]
        for i,a in self.ia_pairs: m[ i, a] = m_z1[self.ia_dic[i,a]]

        S = {(-1,-1):S_w1}
        for i   in self.units:    S[ i,-1] = S_u1[self.i_dic[i]]
        for a   in self.actions:  S[-1, a] = S_v1[self.a_dic[a]]
        for i,a in self.ia_pairs: S[ i, a] = S_z1[self.ia_dic[i,a]]

        return Parameter(
            model = self.model, 
            units = set(self.units), 
            actions = set(self.actions), 
            ia_pairs = set(self.ia_pairs), 
            m = m, 
            S = S
            )

@dataclass
class BatchUpdater:
    model: Model
    parameter: Parameter
    data: Data

    def update(self):
        self.ppc = PackedParameterCalculator(
            model = self.model, 
            parameter = self.parameter, 
            data = self.data,
            units = list(self.data.units()),
            actions = list(self.data.actions()),
            ia_pairs = list(self.data.ia_pairs()),
            )

        result = scipy.optimize.minimize(
            self.ppc.loss, 
            self.ppc.w.val, 
            method = 'L-BFGS-B',
            jac = self.ppc.grad,
            callback = self.ppc.update_y,
        )

        return self.ppc.unpack_param(result.x)
