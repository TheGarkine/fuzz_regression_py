from fuzzy_regression import *
import enum


class RegressionDataType(enum.IntEnum):
    Native = 0
    # e.g. CSV file


class Regression:
    def __init__(self, data):
        self.data = data
        self.solution = None

    @staticmethod
    def of(data, dtype=RegressionDataType.Native):
        if dtype == RegressionDataType.Native:
            return Regression(data)
        else:
            raise NotImplementedError('dtype not implemented')

    @property
    def symmetric(self):
        return SymmetricRegression(self)

    @property
    def asymmetric(self):
        return AsymmetricRegression(self)


class SymmetricRegression:
    def __init__(self, regression):
        self.regression = regression

    def linear_LP(self, *args, **kwargs):
        return RegressionResult(
            self.regression,
            solution=fuz_sym_lin_reg_LP(self.regression.data, *args, **kwargs),
            plot_fn=plot_sym_lin)

    def linear_QP(self, *args, **kwargs):
        return RegressionResult(
            self.regression,
            solution=fuz_sym_lin_reg_QP(self.regression.data, *args, **kwargs),
            plot_fn=plot_sym_lin)

    def linear_QP_expert_adv(self, *args, **kwargs):
        return RegressionResult(
            self.regression,
            solution=fuz_sym_lin_reg_QP_expert_adv(self.regression.data, *args, **kwargs),
            plot_fn=plot_sym_lin)

    def linear_QP_expert(self, *args, **kwargs):
        return RegressionResult(
            self.regression,
            solution=fuz_sym_lin_reg_QP_expert(self.regression.data, *args, **kwargs),
            plot_fn=plot_sym_lin)


class AsymmetricRegression:
    def __init__(self, regression):
        self.regression = regression

    def linear_QP(self, *args, **kwargs):
        return RegressionResult(
            self.regression,
            solution=fuz_asym_lin_reg_QP(self.regression.data, *args, **kwargs),
            plot_fn=plot_asym_lin)


class RegressionResult:
    def __init__(self, regression, solution, plot_fn=None):
        self.regression = regression
        self.regression.solution = solution
        self.plot_fn = plot_fn

    def plot(self, *args, **kwargs):
        data = self.regression.solution
        assert self.plot_fn
        return self.plot_fn(self.regression, *args, **kwargs)

    def __repr__(self):
        return str(self.regression.solution)
