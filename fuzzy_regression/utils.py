import collections
import cvxopt
import numpy as np


SymLinearSolution = collections.namedtuple('SymLinearSolution', ['c0', 'a0', 'c1', 'a1'])
SymLinearExpertSolution = collections.namedtuple('SymLinearExpertSolution', ['c0', 'a0', 'c1', 'a1', 'e'])

AsymLinearSolution = collections.namedtuple('AsymLinearSolution', ['l0', 'u0', 'a0', 'l1', 'u1', 'a1'])


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # https://scaron.info/blog/quadratic-programming-in-python.html
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def lin_reg_QP(list_of_coordinates, h=0, k1=1, k2=1):
    n = len(list_of_coordinates[0])-1
    new_list = [(1, *c) for c in list_of_coordinates]

    # caching sum to access later
    sum_matrix = []
    for x in range(n+2):
        row = []
        for y in range(n+2):
            row.append(sum([i[x]*i[y] for i in new_list]))
        sum_matrix.append(row)

    Q_matrix = []
    for j in range(n+1):
        row = []
        for i in range((n+1)):
            row.append(k1*sum_matrix[j][i])
        Q_matrix.append(row)

    Q = np.array(Q_matrix) * 2

    p_vector = []
    for j in range(n+1):
        p_vector.append(-2*sum_matrix[n+1][j]*k1)
    p = np.array(p_vector)
    # no constraints
    G = np.array([[0. for _ in range(n+1)]])
    h = np.array([0.])

    return cvxopt_solve_qp(Q, p, G, h)
