from recommendation import Recommendation
import numpy as np
import cvxpy as cp


class Optimal:
    name = 'Optimal'

    def __init__(self, dataloader, args, mask, logger):
        self.A = dataloader.A
        self.s = dataloader.s
        self.L = dataloader.L
        self.target_mask = mask
        self.old_conflict = dataloader.old_conflict
        self.args = args
        self.logger = logger
        self.dataloader = dataloader

    def recommend(self):
        Aplus, new_conflict, Lprime = self.findOptimalNewGraph(self.A, self.L, self.s, self.target_mask,
                                                               d=self.args.d, beta=self.args.beta,
                                                               solver=self.args.solver,
                                                               verbose=self.args.verbose)

        return Recommendation(Aplus, new_conflict, self.dataloader)

    def findOptimalNewGraph(self, A, L, s, target_mask, beta=10.0, d=1.0, verbose=False, solver='SCS'):
        if len(s.shape) == 1:
            s = s[:, None]
        n = A.shape[0]


        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        Aplus = cp.Variable((n, n), symmetric=True)
        Lplus = cp.diag(cp.sum(Aplus, axis=0)) - Aplus
        t = cp.Variable((1, 1))
        Lprime = L+Lplus

        beta = cp.Parameter(nonneg=True, value=beta)

        constraints = [0<=Aplus,
                       Aplus<=d,
                       cp.trace(Aplus)==0,
                       cp.sum(Aplus)==2*beta,
                       cp.bmat([[np.eye(n)+Lprime, s], [s.T, t]]) >> 0]
        if not target_mask.all():
            constraints += [cp.sum(cp.multiply(Aplus, 1-target_mask)) == 0]
        prob = cp.Problem(cp.Minimize(t[0, 0]),
                          constraints)
        prob.solve(solver=solver, verbose=bool(verbose))
        return Aplus.value, t[0, 0].value, Lprime.value
