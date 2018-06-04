import numpy as np
from Auxilary import xMx, constrained,StopArm
from sklearn.metrics import mean_squared_error as mse

class Controller:

    def __init__(self, model, simulator, plotter):
        self.xDim = 8
        self.uDim = 2
        self.R = self.setR()
        self.Q = self.setQ()
        self.finalQ = self.setFinalQ()
        self.numOfSteps = 8
        self.t = 0
        self.threshold = 1e-4
        self.model = model
        self.dt = 1
        self.maxIter = 150
        self.lambMax = 100000
        self.lambFactor = 1.2
        self.simulator = simulator
        self.plotter = plotter
        self.ball = np.copy(self.simulator.getBall())
        self.U = np.random.random((self.numOfSteps,self.uDim))
        self.X = np.zeros((self.numOfSteps, self.xDim))
        self.prevCost = 10000


    # Calculate next step using the iLQR algorithm
    def calculateNextAction(self, x0):
        if self.ball[0] != self.simulator.getBall()[0] or self.ball[1] != self.simulator.getBall()[1]:
            self.reset()


        U = np.copy(np.roll(self.U, -2))
        uStop=StopArm(self.model,self.X[-1,:])
        U[-1,:] = uStop
        self.X, self.U, cost = self.ilqr(x0, U)
        nextAction = self.U[0]

        # Plotting trajectory
        self.plotter.updateTrajectoryState(self.X, self.simulator.getBall(), self.t, nextAction)
        self.plotter.updateCostHistory(cost[0])


        return nextAction


    def ilqr(self, x0, U=None):
        """ use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function
        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        U = self.U if U is None else U
        tN = self.numOfSteps #TODO DELETE U.shape[0]  # number of time steps
        dt = self.dt  # time step

        alpha = 1
        lamb = 1  # regularization parameter  todo u chage it
        sim_new_trajectory = True
        X, cost = self.calculateTrajectory(x0, U) #TODO: Delete

        for ii in range(self.maxIter):

            if sim_new_trajectory == True:
                #X, cost = self.calculateTrajectory(x0, U) #TODO: return if needed!!

                f_x = np.zeros((tN, self.xDim, self.xDim))  # df / dx
                f_u = np.zeros((tN, self.xDim, self.uDim))  # df / du
                l = np.zeros((tN, 1))  # immediate state cost
                l_x = np.zeros((tN, self.xDim))  # dl / dx
                l_xx = np.zeros((tN, self.xDim, self.xDim))  # d^2 l / dx^2
                l_u = np.zeros((tN, self.uDim))  # dl / du
                l_uu = np.zeros((tN, self.uDim, self.uDim))  # d^2 l / du^2
                l_ux = np.zeros((tN, self.uDim, self.xDim))  # d^2 l / du / dx
                for t in range(tN - 1):
                    A, B = self.deriveAB(X[t], U[t])
                    f_x[t] = A
                    f_u[t] = B
                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = self.immediateCost(X[t], U[t])
                l[-1], l_x[-1], l_xx[-1] = self.finalCost(X[-1])
                ##print "l sum: {}".format(l.sum())
                sim_new_trajectory = False
            ##print "I'm here! {}".format(t)

            # optimize things!
            # initialize Vs with final state cost and set up k, K
            V = l[-1].copy()  # value function
            V_x = l_x[-1].copy()  # dV / dx
            V_xx = l_xx[-1].copy()  # d^2 V / dx^2
            k = np.zeros((tN, self.uDim))  # feedforward modification
            K = np.zeros((tN, self.uDim, self.xDim))  # feedback gain

            # NOTE: they use V' to denote the value at the next timestep,
            # they have this redundant in their notation making it a
            # function of f(x + dx, u + du) and using the ', but it makes for
            # convenient shorthand when you drop function dependencies

            # work backwards to solve for V, Q, k, and K
            for t in range(tN - 2, -1, -1):
                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.

                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t]))
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

                # Calculate Q_uu^-1 with regularization term set by
                # Levenberg-Marquardt heuristic (at end of this loop)
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                ##print "Q_uu_evals: {}".format(Q_uu_evals) #TODO: Check lambda's scale
                Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))
                #Q_uu_inv=inv(Q_uu)
                # 5b) k = -np.dot(Q_uu^-1, Q_u)
                k[t] = -np.dot(Q_uu_inv, Q_u)
                # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

            Unew = np.zeros((tN, self.uDim))
            # calculate the optimal change to the control trajectory
            xnew = x0.copy().squeeze()  # 7a)

            for t in range(tN - 1):
                # use feedforward (k) and feedback (K) gain matrices
                # calculated from our value function approximation
                # to take a stab at the optimal control signal

                Unew[t] = U[t] + alpha*k[t] + np.dot(K[t], xnew - X[t])  # 7b)
                Unew[t] = constrained(np.copy(Unew[t]))

                xnew = self.plantDynamics(xnew, Unew[t])  # 7c)

            # evaluate the new trajectory
            ##print "Calling from Unew:"

            Xnew, newCost = self.calculateTrajectory(x0, Unew)
            #print "Unew Cost: {}".format(newCost)
            # Levenberg-Marquardt heuristic
            if newCost < cost:
                #print "newCost is lower!"
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lambFactor

                X = np.copy(Xnew)  # update trajectory
                U = np.copy(Unew)  # update control signal
                ##print "The Unew is: {}".format(U)
                oldCost = np.copy(cost)
                cost = np.copy(newCost)
                sim_new_trajectory = True  # do another rollout

                # #print("iteration = %d; Cost = %.4f;"%(ii, newCost) +
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldCost - cost) / cost) < self.threshold):
                    ##print "Under threshold!"
                    ##print("Converged at iteration = %d; Cost = %.4f;" % (ii, newCost) +
                    #      " logLambda = %.1f" % np.log(lamb))
                    break

            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lambFactor
                # #print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lambMax:
                    #print "Lambda is bigger than max"
                    ##print("lambda > max_lambda at iteration = %d;" % ii +
                     #     " Cost = %.4f; logLambda = %.1f" % (cost,np.log(lamb)))
                    break

            #print "{}. Cost: {}, alpha: {}".format(ii, newCost[0],alpha)
            alpha *=0.7#cost/4000
            ##print "{}. cost: {}".format(ii, cost)
           # self.plotter.updateTempTrajectoryState(X, self.simulator.getBall(), ii, cost)
           # self.plotter.plot()

        return X, U, cost


    #TODO
    def calculateTrajectory(self, x0, U):
        """ do a rollout of the system, starting at x0 and
                applying the control sequence U
                x0 np.array: the initial state of the system
                U np.array: the control sequence to apply
                """
        ##print "inside Calculate Trajectory:"
        ##print "x0:"
        ##print x0
        ##print "U:"
        ##print U

        tN = U.shape[0]
        dt = self.dt
        X = np.zeros((tN, self.xDim))
        X[0] = np.copy(x0).squeeze()
        cost = 0

        # Run simulation with substeps
        for t in range(tN - 1):
            X[t + 1] = self.plantDynamics(X[t], U[t])
            l, _, _, _, _, _ = self.immediateCost(X[t], U[t])
            cost = cost + dt * l
        # Adjust for final cost, subsample trajectory
        l_f, _, _ = self.finalCost(X[-1])
        ##print "Cost: {} FinalCost: {}".format(cost, l_f)
        cost = cost + l_f

        ##print "Cost: {}".format(cost)
        return X, cost

    # TODO: Document
    def immediateCost(self, x, u):
        """ the immediate state cost function """
        u = constrained(u)
        u_ = np.reshape(np.copy(u), (self.uDim, 1))
        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        xTarget = np.copy(x_)
        xTarget[4, 0] = (x_[4, 0] - self.simulator.getBall()[0])
        xTarget[5, 0] = (x_[5, 0] - self.simulator.getBall()[1])
        # compute cost
        l = 0.5*xMx(u_, self.R) + 0.5 * xMx(xTarget, self.Q)
        ##print "uRu: {}, xQx: {}".format(xMx(u_, self.R), xMx(xTarget, self.Q))
        # compute derivatives of cost
        #l_x = np.zeros(self.xDim).squeeze()
        #l_xx = np.zeros((self.xDim, self.xDim))
        l_x = np.matmul(xTarget.T, self.Q).squeeze()  # TODO: Make sure dims are good
        l_xx = self.Q
        l_u = np.matmul(u_.T, self.R).squeeze()
        l_uu = self.R
        l_ux = np.zeros((self.uDim, self.xDim))

        return l, l_x, l_xx, l_u, l_uu, l_ux

    def finalCost(self, x):
        ##print "Final X is: {},{}".format(x[4],x[5])
        """ the final state cost function """
        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        xTarget = np.copy(x_)
        xTarget[4, 0] = (x_[4, 0] - self.simulator.getBall()[0])
        xTarget[5, 0] = (x_[5, 0] - self.simulator.getBall()[1])
        l = 0.5*xMx(xTarget, self.finalQ)
        l_x = np.matmul(xTarget.T, self.finalQ).squeeze()  # TODO: Make sure dims are good
        l_xx = self.finalQ
        # Final cost only requires these three values
        return l, l_x, l_xx

    # TODO: This function needs to be verified
    def deriveAB(self, x, u, eps=1e-6):
        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        u_ = np.reshape(np.copy(u), (self.uDim, 1))
        xk1 = self.model.predict(x_, u_)
        A, B = self.model.deriveAB(x, u)

        return A, B

    #TODO
    def plantDynamics(self, x, u):
            """ simulate a single time step of the plant, from
            initial state x and applying control signal u
            x np.array: the state of the system
            u np.array: the control signal
            """
            xNext = self.model.predict(x, u)

            return xNext.squeeze()


    def reset(self):
        """ reset the state of the system """
        # Index along current control sequence
        self.t = 0
        self.U = np.zeros((self.numOfSteps, self.uDim))
        self.ball = np.copy(self.simulator.getBall())

    # Evaluates the LTI error and updates the Graph's data
    def evaluateLTIError(self, xk, uk, xk1):
        xk_ = np.reshape(np.copy(xk), (self.xDim, 1))
        uk_ = np.reshape(np.copy(uk), (self.uDim, 1))
        xk1_ = np.reshape(np.copy(xk1), (self.xDim, 1))
        ##print "Calling from evaluate: {}".format(uk)
        A, B = self.deriveAB(xk, uk)
        xk1_lti = np.matmul(A, xk_) + np.matmul(B, uk_)
        ltiErr = mse(xk1_lti, xk1_)
        self.plotter.updateLTIHistory(ltiErr)


    # set Q function
    def setQ(self):

        Q = np.zeros((self.xDim, self.xDim))
        Q[0, 0] = 0    # cos(theta) of outer arm
        Q[1, 1] = 0    # cos(theta) of inner arm
        Q[2, 2] = 0    # sin(theta) of outer arm
        Q[3, 3] = 0    # sin(theta) of inner arm
        Q[4, 4] = 1e2  # distance between ball and fingertip - X axis
        Q[5, 5] = 1e2  # distance between ball and fingertip - Y axis
        Q[6, 6] = 1    # velocity of inner arm
        Q[7, 7] = 1    # velocity of outer arm

        return Q

    # set Q function
    def setFinalQ(self):
        Q = np.zeros((self.xDim, self.xDim))
        Q[0, 0] = 0   # cos(theta) of outer arm
        Q[1, 1] = 0   # cos(theta) of inner arm
        Q[2, 2] = 0   # sin(theta) of outer arm
        Q[3, 3] = 0   # sin(theta) of inner arm
        Q[4, 4] = 2e2   # distance between ball and fingertip - X axis
        Q[5, 5] = 2e2   # distance between ball and fingertip - Y axis
        Q[6, 6] = 5e1   # velocity of inner arm
        Q[7, 7] = 5e1   # velocity of outer arm
        #Q = self.setQ()
        return Q

    def setR(self):
        return np.identity(self.uDim)*1e-2