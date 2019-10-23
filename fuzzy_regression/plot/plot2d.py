from fuzzy_regression.utils import SymLinearExpertSolution
import matplotlib.pyplot as plt


def prepare_plot(labels=None):
    if not labels:
        labels = ['x', 'y']

    plt.figure(figsize=(18, 10))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('x' if not len(labels) > 0 else labels[0], fontsize=18)
    plt.ylabel('y' if not len(labels) > 1 else labels[1], fontsize=18)


def plot(list_of_coordinates, a0, a1, u0, u1, l0=None, l1=None, e=None, h=None, labels=None):
    X, Y = zip(*list_of_coordinates)

    if not l0 or not l1:
        l0 = u0
        l1 = u1

    fuzzy_regressed_expectancy = [a1*x + a0 for x in X]

    border_lower = [a1*x + a0 - (l0 + l1*x) for x in X]
    border_upper = [a1*x + a0 + (u0 + u1*x) for x in X]

    prepare_plot(labels)

    plt.plot(X, Y, linestyle='none', marker='.', markersize=14)
    plt.plot(X, fuzzy_regressed_expectancy)
    plt.plot(X, border_lower)
    plt.plot(X, border_upper)

    if e:
        if h is None:
            h = 0

        border_lower_e = [a1*x + a0 - (1-h)*(l0 + l1*x) - e for x in X]
        border_upper_e = [a1*x + a0 + (1-h)*(u0 + u1*x) + e for x in X]

        plt.plot(X, border_lower_e, '--')
        plt.plot(X, border_upper_e, '--')

    return plt


def plot_sym_lin(regression, **kwargs):
    if isinstance(regression.solution, SymLinearExpertSolution):
        return plot(
            regression.data,

            regression.solution.a0,
            regression.solution.a1,

            regression.solution.c0,
            regression.solution.c1,

            e=regression.solution.e,

            **kwargs)
    else:
        return plot(
            regression.data,

            regression.solution.a0,
            regression.solution.a1,

            regression.solution.c0,
            regression.solution.c1,

            **kwargs)


def plot_asym_lin(regression, **kwargs):
    return plot(
        regression.data,

        regression.solution.a0,
        regression.solution.a1,

        regression.solution.u0,
        regression.solution.u1,

        regression.solution.l0,
        regression.solution.l1,

        **kwargs)
