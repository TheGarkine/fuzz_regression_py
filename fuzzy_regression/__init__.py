import os
import sys

from distutils.sysconfig import get_python_lib

if os.getenv('VIRTUAL_ENV'):
    # fix cvxopt dll not found error
    os.environ['PATH'] += os.pathsep + \
        (os.path.join(os.environ['VIRTUAL_ENV'], 'Library', 'bin'))
else:
    os.environ['PATH'] += os.pathsep + \
        (os.path.join(get_python_lib(), '..', '..', 'Library', 'bin'))

from fuzzy_regression.linear.asym import (fuz_asym_lin_reg_QP,fuz_asym_lin_reg_QP_expert_adv)
from fuzzy_regression.linear.sym import (fuz_sym_lin_reg_LP,
                                         fuz_sym_lin_reg_QP,
                                         fuz_sym_lin_reg_QP_expert,
                                         fuz_sym_lin_reg_QP_expert_adv)

from fuzzy_regression.plot.plot2d import (plot, plot_sym_lin, plot_asym_lin)
from fuzzy_regression.regression import (Regression, RegressionDataType)
