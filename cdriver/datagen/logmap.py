import numpy as np
from tqdm import tqdm
from types import SimpleNamespace


class LogMap:
    def __init__(self, r: np.ndarray, A: np.ndarray, x0: np.ndarray):
        self.r = r
        self.A = A
        self.x0 = x0

    def bounder(self, x: np.ndarray) -> np.ndarray:
        """Applies boundary condition on data"""
        z = np.array(x)
        while not np.all(np.logical_and(z <= 1, z >= 0)):
            z = np.abs(z)
            inds = np.where(z > 1)[0]
            for i in inds:
                z[i] = 1 - (z[i] - 1)
        return z

    def step(self, x : np.ndarray) -> np.ndarray:
        return self.bounder(self.r * x * (np.ones(len(x)) - self.A @ x))

    def gen_dataset(self, n : int) -> np.ndarray:
        x = [self.x0]
        for i in range(n - 1):
            x.append(self.step(x[-1]))
        return np.array(x)


class LogmapExpRunner:
    def __init__(self, nvars : int, baseA : np.ndarray, r_interval: list):
        self.nvars = nvars
        self.baseA = baseA
        self.r_interval = r_interval

    def sample_params(self, A=None, r=None, x0=None, seed=None):
        np.random.seed(seed)
        rint = self.r_interval
        if r is None:
            r = rint[0] + (rint[1] - rint[0]) * np.random.rand(self.nvars)

        if A is None:
            A = self.baseA * (0.4 * np.random.rand(self.nvars ** 2).reshape([self.nvars, self.nvars]) + 0.1)
        np.fill_diagonal(A, 1)

        if x0 is None:
            x0 = np.random.rand(self.nvars)
        self.A = A
        self.r = r
        self.x0 = x0
        return r, A, x0

    def gen_experiment(self, n, A=None, r=None, x0=None, seed=None):
        r, A, x0 = self.sample_params(r=r, A=A, x0=x0, seed=seed)
        data = LogMap(r=r, A=A, x0=x0).gen_dataset(n)
        self.data = data
        return data, {'r': r, 'A': A, 'x0': x0}


def gen_logmapdata(param_dict: dict) -> tuple[list[np.ndarray], list[dict]]:
    """Generate logmap dataset

    :param dict param_dict: dict containing the parameters for the logmap dataset
    :return: (datasets, parameters) tuple of data and parameters
    :rtype: tuple
    """
    dparams = SimpleNamespace(**param_dict)
    N = dparams.N  # number of realizations
    n = dparams.n  # Length of time series
    rint = dparams.rint  # interval to chose from the value of r parameter
    A0 = dparams.A0
    A = dparams.A

    dataset, params = zip(*[LogmapExpRunner(nvars=3,
                                            baseA=A0,
                                            r_interval=rint).gen_experiment(n=n,
                                                                            A=A,
                                                                            seed=i) for i in
                            tqdm(range(N), desc='Generating dataset')])
    return dataset, params
