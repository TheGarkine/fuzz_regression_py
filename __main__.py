from fuzzy_regression import *
import matplotlib.pyplot as plt


example2_keys = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
example2_values = [7., 8., 8., 12., 9., 10., 7., 10., 11., 12., 12., 13.]
example2 = list(zip(example2_keys, example2_values))


if __name__ == "__main__":
    print(fuz_sym_lin_reg_LP(example2))
