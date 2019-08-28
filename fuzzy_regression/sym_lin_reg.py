import cvxopt
import numpy as np
from fuzzy_regression.utils import cvxopt_solve_qp, lin_reg_QP

def fuz_sym_lin_reg_LP(list_of_coordinates, h=0, tol=1e-12):
    """
    This function optimizes the given list of coordinates in a linear behaviour.
    The last column of the iteratable will be used the criterium analyzed.
    """
    
    n = (len(list_of_coordinates[0]))-1

    c = [len(list_of_coordinates), 0]  # c0, a0
    for i in range (n):
        c.append(sum(list(zip(*list_of_coordinates))[i]))  # c_j
        c.append(0.0)  # a_j

    # constraints
    A = []
    b = []
    # lower border must be smaller than value
    for el in list_of_coordinates:
        constraint = [-(1-h),1]
        for j in range(n):
            constraint.append(float(-el[j]*(1-h)))  # c_j
            constraint.append(float(el[j]))  # a_j
        A.append(constraint)
        b.append(float(el[n]))
        
    # upper border must be greater than value
    for el in list_of_coordinates:
        constraint = [-(1-h),-1]
        for j in range(n):
            constraint.append(float(-(1-h)*el[j]))  # c_j
            constraint.append(float(-el[j]))  #a_j
        A.append(constraint)
        b.append(float(-el[n]))

    bounds = []
    for i in range(n+1):
        bounds.append((0.0,None))  # c
        bounds.append((None,None))  # a

    c = cvxopt.matrix(c)
    A = cvxopt.matrix(A).T
    b = cvxopt.matrix(b)

    res = cvxopt.solvers.lp(c,A,b)
    return res["x"]

def fuz_sym_lin_reg_QP(list_of_coordinates, h=0, k1=1, k2=1):
    n = len(list_of_coordinates[0])-1
    new_list = [(1,*c) for c in list_of_coordinates]
    
    #caching sum to access later
    sum_matrix = []
    for x in range(n+2):
        row = []
        for y in range(n+2):
            row.append(sum([i[x]*i[y] for i in new_list]))
        sum_matrix.append(row)
        
    Q_matrix = []
    for j in range (n+1):
        row = []
        #line c
        for i in range (n+1):
            row.append(k2*sum_matrix[j][i])
            row.append(0.)  #a
        Q_matrix.append(row)
        
        row=[]
        # line a
        for i in range ((n+1)):
            row.append(0.)  #c
            row.append(k1*sum_matrix[j][i])
        Q_matrix.append(row)
    
    Q=np.array(Q_matrix) *2
    
    p_vector = []
    for j in range(n+1):
        p_vector.append(0.)  #c
        p_vector.append(-2*sum_matrix[n+1][j]*k1)
    p = np.array(p_vector)

    # Gh
    G_buffer = []
    h_buffer = []

    for el in new_list:
        #lower then upper
        row = []
        for j in range(n+1):
            row.append(-(1-h)*el[j])
            row.append(el[j])
        G_buffer.append(row)
        h_buffer.append(el[n+1])
        
        #higher then lower
        row=[]
        for j in range(n+1):
            row.append(-(1-h)*el[j])
            row.append(-el[j])
        G_buffer.append(row)
        h_buffer.append(-el[n+1])

    #c_j >= 0
    for j in range(n+1):
        row =[]
        for i in range(n+1):
            if i == j:
                row.append(-1.)
                row.append(0.)
            else:
                row.append(0.)
                row.append(0.)
        G_buffer.append(row)
        h_buffer.append(0.)

    G = np.array(G_buffer)
    h = np.array(h_buffer)

    return cvxopt_solve_qp(Q, p, G, h)

def fuz_sym_lin_reg_QP_expert_adv(list_of_coordinates, h=None, k1=1, k2=1, k3=1, t=2):
    n = len(list_of_coordinates[0])-1
    p = len(list_of_coordinates)
    
    new_list = [(1,*c) for c in list_of_coordinates]
    
    lin_reg = lin_reg_QP(list_of_coordinates) #get factors from linear regression
    
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
    #caching sum to access later
    if h is None:
        for x in range(n+2):
            row = []
            for y in range(n+2):
                row.append(float(sum([i[x]*i[y] for num,i in enumerate(new_list)])))
            sum_matrix.append(row)
            h= [0. for _ in range(p)]
    else:
        for x in range(n+2):
            row = []
            for y in range(n+2):
                row.append(float(sum([i[x]*i[y]*h[num] for num,i in enumerate(new_list)])))
            sum_matrix.append(row)
        
    Q_matrix = []
    for j in range (n+1):
        row = []
        #line c
        for i in range (n+1):
            row.append(k2*sum_matrix[j][i])  #c
            row.append(0.)  #a
        row.append(0)  #e
        Q_matrix.append(row)
        
        row=[]
        # line a
        for i in range ((n+1)):
            row.append(0.)  #c
            row.append(k1*sum_matrix[j][i])  #a
        row.append(0.)  #e
        Q_matrix.append(row)
    
    row = [0 for _ in range((n+1)*2)]
    row.append(k3)
    Q_matrix.append(row)
    
    Q=np.array(Q_matrix) *2
    
    p_vector = []
    for j in range(n+1):
        p_vector.append(0.)  #c
        p_vector.append(-2*sum_matrix[n+1][j]*k1)  #a
    p_vector.append(0.)  #e
    p = np.array(p_vector)

    # Gh
    G_buffer = []
    h_buffer = []

    for num,el in enumerate(new_list):
        if el in R:
            #lower then upper
            row = []
            for j in range(n+1):
                row.append(-(1-h[num])*el[j])  #c
                row.append(el[j])  #a
            row.append(0.)  #e
            G_buffer.append(row)
            h_buffer.append(el[n+1])

            #higher then lower
            row=[]
            for j in range(n+1):
                row.append(-(1-h[num])*el[j])  #c
                row.append(-el[j])  #a
            row.append(0.)  #e
            G_buffer.append(row)
            h_buffer.append(-el[n+1])
            
        if el in S:
            #lower then upper + e
            row = []
            for j in range(n+1):
                row.append(-(1-h[num])*el[j])
                row.append(el[j])
            row.append(-1.)
            G_buffer.append(row)
            h_buffer.append(el[n+1])

            #higher then lower + e
            row=[]
            for j in range(n+1):
                row.append(-(1-h[num])*el[j])
                row.append(-el[j])
            row.append(-1.)
            G_buffer.append(row)
            h_buffer.append(-el[n+1])

    #c_j >= 0
    for j in range(n+1):
        row =[]
        for i in range(n+1):
            if i == j:
                row.append(-1.)
                row.append(0.)
            else:
                row.append(0.)
                row.append(0.)
        row.append(0.)
        G_buffer.append(row)
        h_buffer.append(0.)

    G = np.array(G_buffer)
    h = np.array(h_buffer)

    return cvxopt_solve_qp(Q, p, G, h)

def fuz_sym_lin_reg_QP_expert(list_of_coordinates, h, k1=1, k2=1):
    
    n = len(list_of_coordinates[0])-1
    new_list = [(1,*c) for c in list_of_coordinates]
    
    
    #caching sum to access later
    sum_matrix = []
    for x in range(n+2):
        row = []
        for y in range(n+2):
            row.append(sum([i[x]*i[y]*h[num] for num,i in enumerate(new_list)]))
        sum_matrix.append(row)
        
    Q_matrix = []
    for j in range (n+1):
        row = []
        #line c
        for i in range (n+1):
            row.append(k2*sum_matrix[j][i])
            row.append(0.)  #a
        Q_matrix.append(row)
        
        row=[]
        # line a
        for i in range ((n+1)):
            row.append(0.)  #c
            row.append(k1*sum_matrix[j][i])
        Q_matrix.append(row)
    
    Q=np.array(Q_matrix) *2
    
    p_vector = []
    for j in range(n+1):
        p_vector.append(0.)  #c
        p_vector.append(-2*sum_matrix[n+1][j]*k1)
    p = np.array(p_vector)

    # Gh
    G_buffer = []
    h_buffer = []

    for num,el in enumerate(new_list):
        #lower then upper
        row = []
        for j in range(n+1):
            row.append(-(1-h[num])*el[j])
            row.append(el[j])
        G_buffer.append(row)
        h_buffer.append(el[n+1])
        
        #higher then lower
        row=[]
        for j in range(n+1):
            row.append(-(1-h[num])*el[j])
            row.append(-el[j])
        G_buffer.append(row)
        h_buffer.append(-el[n+1])

    #c_j >= 0
    for j in range(n+1):
        row =[]
        for i in range(n+1):
            if i == j:
                row.append(-1.)
                row.append(0.)
            else:
                row.append(0.)
                row.append(0.)
        G_buffer.append(row)
        h_buffer.append(0.)

    G = np.array(G_buffer)
    h = np.array(h_buffer)

    return cvxopt_solve_qp(Q, p, G, h)