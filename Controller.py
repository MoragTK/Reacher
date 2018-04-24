import scipy
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
from Auxilary import plot_curve, xMx, solveRiccati




class Controller:

    def __init__(self, model, simulator):
        self.xDim = 8
        self.uDim = 2
        self.R = np.identity(self.uDim)*1e-4
        self.Q = self.setQ()
        self.numOfSteps = 15
        self.t = 0
        self.threshold = 1e-5
        self.model = model
        self.dt = 1e-3 #TODO: Look at size of dt
        self.maxIter = 10000
        self.lambMax = 1000
        self.lambFactor = 10
        self.simulator = simulator
        self.ball = np.copy(self.simulator.getBall())
        self.U = np.zeros((self.numOfSteps, self.uDim))
        self.X = np.zeros((self.numOfSteps, self.xDim))

    # Calculate next step using the iLQR algorithm
    def calculateNextAction(self, x0):
        if self.ball[0] != self.simulator.getBall()[0] or self.ball[1] != self.simulator.getBall()[1]:
            self.reset()

        if self.t >= self.numOfSteps - 1:
            self.reset()
        U = np.copy(self.U[self.t:])  # TODO: Check that still controllable
        self.X, self.U[self.t:], cost = self.ilqr(x0, U)
        #print 'cost: ' +str(cost)
        plot_curve(self.X,self.simulator.getBall(),cost, self.t)
        nextAction = self.U[self.t]

        # move us a step forward in our control sequence
        self.t += 1
        return nextAction

    # TODO: Document
    def ilqr(self, x0, U=None):
        """ use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function
        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        U = self.U if U is None else U

        tN = U.shape[0]  # number of time steps
        dt = self.dt  # time step

        lamb = 1.0  # regularization parameter
        sim_new_trajectory = True

        for ii in range(self.maxIter):

            if sim_new_trajectory == True:
                # simulate forward using the current control trajectory
                X, cost = self.calculateTrajectory(x0, U)
                oldCost = np.copy(cost)  # copy for exit condition check

                # now we linearly approximate the dynamics, and quadratically
                # approximate the cost function so we can use LQR methods

                # for storing linearized dynamics
                # x(t+1) = f(x(t), u(t))
                f_x = np.zeros((tN, self.xDim, self.xDim))  # df / dx
                f_u = np.zeros((tN, self.xDim, self.uDim))  # df / du
                # for storing quadratized cost function
                l = np.zeros((tN, 1))  # immediate state cost
                l_x = np.zeros((tN, self.xDim))  # dl / dx
                l_xx = np.zeros((tN, self.xDim, self.xDim))  # d^2 l / dx^2
                l_u = np.zeros((tN, self.uDim))  # dl / du
                l_uu = np.zeros((tN, self.uDim, self.uDim))  # d^2 l / du^2
                l_ux = np.zeros((tN, self.uDim, self.xDim))  # d^2 l / du / dx
                # for everything except final state
                for t in range(tN - 1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    A, B = self.deriveAB(X[t], U[t])
                    f_x[t] = np.eye(self.xDim) + A * dt
                    f_u[t] = B * dt

                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = self.immediateCost(X[t], U[t])
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt
                l[-1], l_x[-1], l_xx[-1] = self.finalCost(X[-1])

                sim_new_trajectory = False

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
                Q_uu_inv = np.dot(Q_uu_evecs,
                                  np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))

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
                Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t])  # 7b)
                # given this u, find our next state
                xnew = self.plant_dynamics(xnew, Unew[t])  # 7c)

            # evaluate the new trajectory
            Xnew, newCost = self.calculateTrajectory(x0, Unew)

            # Levenberg-Marquardt heuristic
            if newCost < cost:
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lambFactor

                X = np.copy(Xnew)  # update trajectory
                U = np.copy(Unew)  # update control signal
                oldCost = np.copy(cost)
                cost = np.copy(newCost)

                sim_new_trajectory = True  # do another rollout

                # print("iteration = %d; Cost = %.4f;"%(ii, newCost) +
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldCost - cost) / cost) < self.threshold):
                    #print("Converged at iteration = %d; Cost = %.4f;" % (ii, newCost) +
                    #      " logLambda = %.1f" % np.log(lamb))
                    break

            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lambFactor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lambMax:
                    #print("lambda > max_lambda at iteration = %d;" % ii +
                     #     " Cost = %.4f; logLambda = %.1f" % (cost,np.log(lamb)))
                    break

        return X, U, cost


    #TODO
    def calculateTrajectory(self, x0, U):
        """ do a rollout of the system, starting at x0 and
                applying the control sequence U
                x0 np.array: the initial state of the system
                U np.array: the control sequence to apply
                """
        tN = U.shape[0]
        dt = self.dt

        X = np.zeros((tN, self.xDim))
        X[0] = np.copy(x0).squeeze()
        cost = 0

        # Run simulation with substeps
        for t in range(tN - 1):
            X[t + 1] = self.plant_dynamics(X[t], U[t])
            l, _, _, _, _, _ = self.immediateCost(X[t], U[t])
            cost = cost + dt * l # todo this is origin
            #cost = cost +  l
        # Adjust for final cost, subsample trajectory
        l_f, _, _ = self.finalCost(X[-1])
        cost = cost + l_f

        return X, cost



    # TODO: This function needs to be verified
    def deriveAB(self, x, u, eps=1e-4):

        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        u_ = np.reshape(np.copy(u), (self.uDim, 1))
        A = np.ones((self.xDim, self.xDim))
        B = np.ones((self.xDim, self.uDim))

        for i in range(0, self.xDim):
            xk = np.copy(x_)
            xk[i, 0] += eps
            state_inc = self.model.predict(xk, u_)
            xk = np.copy(x_)
            xk[i, 0] -= eps
            state_dec = self.model.predict(xk, u_)
            A[:, i] = (state_inc[:, 0] - state_dec[:, 0]) / (2 * eps)  # TODO: Is this how A should be? or transpose?

        for i in range(0, self.uDim):
            uk = np.copy(u_)
            uk[i, 0] += eps
            state_inc = self.model.predict(x_, uk)
            uk = np.copy(u_)
            uk[i, 0] -= eps
            state_dec = self.model.predict(x_, uk)
            B[:, i] = (state_inc[:, 0] - state_dec[:, 0]) / (2 * eps)

        return A, B

    #TODO
    def plant_dynamics(self, x, u):
            """ simulate a single time step of the plant, from
            initial state x and applying control signal u
            x np.array: the state of the system
            u np.array: the control signal
            """
            xNext = self.model.predict(x,u)
            # calculate the change in state
            #xdot = ((xNext - x) / self.arm.dt).squeeze()

            return xNext.squeeze()


    def immediateCost(self, x, u):
        """ the immediate state cost function """
        u_ = np.reshape(np.copy(u), (self.uDim, 1))
        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        xTarget = np.copy(x_)
        xTarget[4, 0] = abs(x_[4, 0] - self.simulator.getBall()[0])
        xTarget[5, 0] = abs(x_[5, 0] - self.simulator.getBall()[1])
        # compute cost
        l = xMx(u_, self.R) + xMx(xTarget, self.Q)
        #print "uRu = {}, xQx = {}".format(xMx(u_, self.R), xMx(xTarget, self.Q))
        # compute derivatives of cost
        l_x = (2 * np.matmul(xTarget.T, self.Q)).squeeze()  # TODO: Make sure dims are good
        l_xx = 2 * self.Q
        l_u = (2 * np.matmul(u_.T, self.R)).squeeze()
        l_uu = 2 * self.R
        l_ux = np.zeros((self.uDim, self.xDim))

        # returned in an array for easy multiplication by time step
        '''
        u_=np.reshape(np.copy(np.copy(u)), (2,1))
        l_u=xMx(u_,self.R).squeeze()
        l_uu=2*self.R

        l_x=np.zeros((8,1)).squeeze()
        l_xx=np.zeros((8,8))
        l=0
        l_ux = np.zeros((self.uDim, self.xDim))
        '''
        return l, l_x, l_xx, l_u, l_uu, l_ux

    def finalCost(self, x):
        """ the final state cost function """
        x_ = np.reshape(np.copy(x), (self.xDim, 1))
        xTarget = np.copy(x_)
        xTarget[4, 0] = abs(x_[4, 0] - self.simulator.getBall()[0])
        xTarget[5, 0] = abs(x_[5, 0] - self.simulator.getBall()[1])
        #print "Ball in final cost:  (X,Y) : ({},{})".format(self.simulator.getBall()[0],self.simulator.getBall()[1])
        #print "Reacher in final cost: (X,Y) : ({},{})".format(x_[4, 0], x_[5, 0])
        #print "Distance: ({},{})".format(xTarget[4],xTarget[5])
        l =  xMx(xTarget, self.Q)
        l_x = (2 * np.matmul(xTarget.T, self.Q)).squeeze()  # TODO: Make sure dims are good
        l_xx = 2 * self.Q

        # Final cost only requires these three values
        return l, l_x, l_xx

    def reset(self):
        """ reset the state of the system """
        print " i just entered reset!"
        # Index along current control sequence
        self.t = 0
        self.U = np.zeros((self.numOfSteps, self.uDim))
        self.ball = np.copy(self.simulator.getBall())

    # set Q function
    def setQ(self):

        Q = np.zeros((self.xDim, self.xDim))
        Q[0, 0] = 0   # cos(theta) of outer arm
        Q[1, 1] = 0   # cos(theta) of inner arm
        Q[2, 2] = 0   # sin(theta) of outer arm
        Q[3, 3] = 0   # sin(theta) of inner arm
        Q[4, 4] = 1e6   # distance between ball and fingertip - X axis
        Q[5, 5] = 1e6   # distance between ball and fingertip - Y axis
        Q[6, 6] = 1e2   # velocity of inner arm
        Q[7, 7] = 1e2   # velocity of outer arm

        return Q

