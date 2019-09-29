from fuzzy_regression.utils import cvxopt_solve_qp
import numpy as np


def fuz_asym_lin_reg_QP(list_of_coordinates, h=0, k1=1, k2=1):
    n = len(list_of_coordinates[0])-1
    new_list = [(1, *c) for c in list_of_coordinates]

    k2 = k2/4

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
        # line l
        for i in range(n+1):
            row.append(k2*sum_matrix[j][i])  # l
            row.append(0.)  # u
            row.append(0.)  # a
        Q_matrix.append(row)

        row = []
        # line u
        for i in range(n+1):
            row.append(0.)  # l
            row.append(k2*sum_matrix[j][i])  # u
            row.append(0.)  # a
        Q_matrix.append(row)

        row = []
        # line a
        for i in range((n+1)):
            row.append(0.)  # l
            row.append(0.)  # u
            row.append(k1*sum_matrix[j][i])  # a
        Q_matrix.append(row)

    Q = np.array(Q_matrix) * 2  # times 2 for algorithm in quadprog

    p_vector = []
    for j in range(n+1):
        p_vector.append(0.)  # l
        p_vector.append(0.)  # u
        p_vector.append(-2*sum_matrix[n+1][j]*k1)  # a
    p = np.array(p_vector)

    # Gh
    G_buffer = []
    h_buffer = []

    for el in new_list:
        # lower then upper
        row = []
        for j in range(n+1):
            row.append(-(1-h)*el[j])
            row.append(0.)
            row.append(el[j])
        G_buffer.append(row)
        h_buffer.append(el[n+1])

        # higher then lower
        row = []
        for j in range(n+1):
            row.append(0.)
            row.append(-(1-h)*el[j])
            row.append(-el[j])
        G_buffer.append(row)
        h_buffer.append(-el[n+1])

    # u_j, l_j >= 0
    for j in range(n+1):
        row_u = []  # l
        row_l = []  # u
        for i in range(n+1):
            if not i == j:
                row_l.append(-1.)
                row_l.append(0.)
                row_l.append(0.)

                row_u.append(0)
                row_u.append(-1.)
                row_u.append(0)
            else:
                row_l.append(0.)
                row_l.append(0.)
                row_l.append(0.)

                row_u.append(0.)
                row_u.append(0.)
                row_u.append(0.)
        G_buffer.append(row_l)
        h_buffer.append(0.)
        G_buffer.append(row_u)
        h_buffer.append(0.)

    G = np.array(G_buffer)
    h = np.array(h_buffer)

    return cvxopt_solve_qp(Q, p, G, h)
