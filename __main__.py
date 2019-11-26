from fuzzy_regression import Regression
import matplotlib.pyplot as plt


example2_keys = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
example2_values = [7., 8., 8., 12., 9., 10., 7., 10., 11., 12., 12., 13.]
example2 = list(zip(example2_keys, example2_values))
example2_expert = [0.8, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8]


if __name__ == "__main__":
    reg = Regression.of(example2)

    # sym
    print('sym.lin.LP')
    reg.symmetric.linear_LP().plot().show()
    print('sym.lin.QP')
    reg.symmetric.linear_QP().plot().show()
    print('sym.lin.QP_exp_adv')
    reg.symmetric.linear_QP_expert_adv(example2_expert).plot().show()
    print('sym.lin.QP_exp')
    reg.symmetric.linear_QP_expert(example2_expert).plot().show()

    # asym
    print('asym.lin.QP')
    reg.asymmetric.linear_QP().plot().show()
