from fuzzy_regression.utils import cvxopt_solve_qp, AsymLinearSolution, AsymLinearExpertSolution, lin_reg_QP
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
            if i == j:
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

    res = cvxopt_solve_qp(Q, p, G, h)
    return AsymLinearSolution(l=res[::3], u=res[1::3], a=res[2::3])

def fuz_asym_lin_reg_QP_expert_adv(list_of_coordinates, h=0, k1=1, k2=1, k3=1, t=2):
    n = len(list_of_coordinates[0])-1
    p = len(list_of_coordinates)

    new_list = [(1, *c) for c in list_of_coordinates]

    # get factors from linear regression
    lin_reg = lin_reg_QP(list_of_coordinates)

    estimates = []
    sigma = 0
    for x in new_list:
        estimate = 0
        for j in range(n+1):
            estimate += x[j]*lin_reg[j]
        estimates.append(estimate)
        sigma += (x[n+1]-estimate)**2
    sigma /= (p-n-1)
    sigma = sigma**0.5

    R, S = set(), set()

    for j in range(len(new_list)):
        if estimates[j]-(sigma*t) <= new_list[j][n+1] <= estimates[j]+(sigma*t):
            R.add(new_list[j])
        else:
            S.add(new_list[j])

    sum_matrix = []
    # caching sum to access later
    for x in range(n+2):
        row = []
        for y in range(n+2):
            row.append(float(sum([i[x]*i[y]
                                    for num, i in enumerate(new_list)])))
        sum_matrix.append(row)

    Q_matrix = []
    for j in range(n+1):
        row = []
        # line l
        for i in range(n+1):
            row.append(k2*sum_matrix[j][i])  # l
            row.append(0.)  # u
            row.append(0.)  # a
        row.append(0.)  # e_l
        row.append(0.)  # e_u
        Q_matrix.append(row)

        row = []
        # line u
        for i in range(n+1):
            row.append(0.)  # l
            row.append(k2*sum_matrix[j][i])  # u
            row.append(0.)  # a
        row.append(0.)  # e_l
        row.append(0.)  # e_u
        Q_matrix.append(row)

        row = []
        # line a
        for i in range((n+1)):
            row.append(0.)  # l
            row.append(0.)  # u
            row.append(k1*sum_matrix[j][i])  # a
        row.append(0.)  # e_l
        row.append(0.)  # e_u
        Q_matrix.append(row)

    row_l = [0 for _ in range((n+1)*3)]
    row_l.append(k3)
    row_l.append(0)
    Q_matrix.append(row_l) # row for e_l

    row_u = [0 for _ in range((n+1)*3+1)]
    row_u.append(k3)
    Q_matrix.append(row_u) # row for e_u
    Q = np.array(Q_matrix) * 2  # times 2 for algorithm in quadprog

    p_vector = []
    for j in range(n+1):
        p_vector.append(0.)  # l
        p_vector.append(0.)  # u
        p_vector.append(-2*sum_matrix[n+1][j]*k1)  # a
    p_vector.append(0.)  # e_l
    p_vector.append(0.)  # e_u
    p = np.array(p_vector)

    # Gh
    G_buffer = []
    h_buffer = []

    for num, el in enumerate(new_list):
        if el in R:
            # lower then upper
            row = []
            for j in range(n+1):
                row.append(-(1-h)*el[j])  # l
                row.append(0.) # u
                row.append(el[j])  # a
            row.append(0.)  # e_l
            row.append(0.)  # e_u
            G_buffer.append(row)
            h_buffer.append(el[n+1])

            # higher then lower
            row = []
            for j in range(n+1):
                row.append(0.) # l
                row.append(-(1-h)*el[j])  # u
                row.append(-el[j])  # a
            row.append(0.)  # e_l
            row.append(0.)  # e_u
            G_buffer.append(row)
            h_buffer.append(-el[n+1])

        if el in S:
            # lower then upper + e
            row = []
            for j in range(n+1):
                row.append(-(1-h)*el[j]) # l
                row.append(0.) # u
                row.append(el[j])
            row.append(-1.) # e_l
            row.append(0.)  # e_u
            G_buffer.append(row)
            h_buffer.append(el[n+1])

            # higher then lower + e
            row = []
            for j in range(n+1):
                row.append(0.) # l
                row.append(-(1-h)*el[j]) # u
                row.append(-el[j])
            row.append(0.)  # e_l
            row.append(-1.) # e_u
            G_buffer.append(row)
            h_buffer.append(-el[n+1])

    # u_j, l_j >= 0
    for j in range(n+1):
        row_u = []  # l
        row_l = []  # u
        for i in range(n+1):
            if i == j:
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
        row_u.append(0.) # e_l
        row_u.append(0.) # e_u
        row_l.append(0.) # e_l
        row_l.append(0.) # e_u
        G_buffer.append(row_l)
        h_buffer.append(0.)
        G_buffer.append(row_u)
        h_buffer.append(0.)

    G = np.array(G_buffer)
    h = np.array(h_buffer)

    res = cvxopt_solve_qp(Q, p, G, h)
    return AsymLinearExpertSolution(l=res[:-2:3], u=res[1:-1:3], a=res[2::3],e_l=res[-2],e_u=res[-1])