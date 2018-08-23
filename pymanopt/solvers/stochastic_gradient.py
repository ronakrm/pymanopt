from __future__ import print_function, division

import time
from copy import deepcopy

from pymanopt.solvers.solver import Solver

from functools import partial 

class StochasticGradient(Solver):
    """
    Stochastic gradient descent algorithm based on
    stoachasticgradient.m from the manopt MATLAB package.
    """

    def __init__(self, stepsize=0.01, *args, **kwargs):
        super(StochasticGradient, self).__init__(*args, **kwargs)

        self._stepsize = stepsize;

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None, feed_dict=None):
        """
        Perform optimization using stochastic gradient descent.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - feed_dict=None
                Optional parameter. Dictionary from which to compute partial
                objectives and gradients. 
        Returns:
            - x
                New point on manifold.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        if feed_dict is not None:
            objective = partial(problem.cost, feed_dict=feed_dict)
            gradient = partial(problem.grad, feed_dict=feed_dict)
        else:
            objective = problem.cost
            gradient = problem.grad

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        # if verbosity >= 2:
            # print(" iter\t\t   cost val\t    grad. norm")
            # print(" cost val\t    grad. norm")

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'stepsize': self._stepsize})

        # while True:
            # Calculate new cost, grad and gradnorm
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
            # iter = iter + 1

        # if verbosity >= 2:
            # print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

        if self._logverbosity >= 2:
            self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            # Descent direction is minus the gradient
        desc_dir = -grad
        # Perform SGD update
        x = man.retr(x, self._stepsize*desc_dir)

        stop_reason = self._check_stopping_criterion(time0, iter=iter,
            objective=cost, stepsize=self._stepsize, gradnorm=gradnorm)

        if stop_reason:
            if verbosity >= 1:
                print(stop_reason)
                print('')

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x, objective(x), stop_reason, time0,
                              stepsize=self._stepsize, gradnorm=gradnorm,
                              iter=iter)
            return x, self._optlog