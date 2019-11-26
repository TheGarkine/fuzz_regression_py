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


def plot(list_of_coordinates, a, u, l=None, e=None, h=None, labels=None):
    X, Y = zip(*list_of_coordinates)

    if l is None:
        l = u

    fuzzy_regressed_expectancy = [a[1]*x + a[0] for x in X]

    border_lower = [a[1]*x + a[0] - (l[0] + l[1]*x) for x in X]
    border_upper = [a[1]*x + a[0] + (u[0] + u[1]*x) for x in X]

    prepare_plot(labels)

    plt.plot(X, Y, linestyle='none', marker='.', markersize=14)
    plt.plot(X, fuzzy_regressed_expectancy)
    plt.plot(X, border_lower)
    plt.plot(X, border_upper)

    if e:
        if h is None:
            h = 0

        border_lower_e = [a[1]*x + a[0] - (1-h)*(l[0] + l[1]*x) - e for x in X]
        border_upper_e = [a[1]*x + a[0] + (1-h)*(u[0] + u[1]*x) + e for x in X]

        plt.plot(X, border_lower_e, '--')
        plt.plot(X, border_upper_e, '--')

    return plt


def plot_sym_lin(regression, **kwargs):
    if isinstance(regression.solution, SymLinearExpertSolution):
        return plot(
            regression.data,

            regression.solution.a,
            regression.solution.c,

            e=regression.solution.e,

            **kwargs)
    else:
        return plot(
            regression.data,

            regression.solution.a,
            regression.solution.c,

            **kwargs)


def plot_asym_lin(regression, **kwargs):
    return plot(
        regression.data,

        regression.solution.a,
        regression.solution.u,
        regression.solution.l,

        **kwargs)
