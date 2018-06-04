import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error as mse

import getpass

username = getpass.getuser()
figPath = '/home/' + username + '/PycharmProjects/Reacher/Figures/'
def StopArm(model,x,N=50):
    x_ = np.reshape(np.copy(x), (1, 8))
    xtarget = np.copy(x_)
    xtarget[0, 6] = 0
    xtarget[0, 7] = 0
    min = 100
    uStop = np.zeros((1, 2))
    for i in range(N):
        u = np.random.random((1, 2))
        x1=model.predict(x_,u)
        MSE = mse(xtarget, x1)
        if MSE < min:
            min = MSE
            uStop = u
    return uStop


def xMx(x, M):
    xt = np.transpose(x)
    xt_M = np.matmul(xt, M)
    xt_M_x = np.matmul(xt_M, x)
    return xt_M_x

def constrained(u, lowerLim=-1, upperLim=1):
    if u[0] > upperLim:
        u[0] = upperLim
    if u[0] < lowerLim:
        u[0] = lowerLim

    if u[1] > upperLim:
        u[1] = upperLim
    if u[1] < lowerLim:
        u[1] = lowerLim

    return u


def evaluateLTIError(controller, xk, uk, xk1):
    xk_ = np.reshape(np.copy(xk), (8, 1))
    uk_ = np.reshape(np.copy(uk), (2, 1))
    xk1_ = np.reshape(np.copy(xk1), (8, 1))
    A, B = controller.deriveAB(xk, uk)
    xk1_lti = np.matmul(A, xk_) + np.matmul(B, uk_)
    ltiErr = mse(xk1_lti, xk1_)
    print ltiErr


'''
def plotCurve(X, ball, step, uk):
    from numpy import *
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    plt.figure(0)
    x = X[:, 4]
    y = X[:, 5]
    plt.ion()
    plt.plot(x, y)
    plt.title("Step: {}".format(step))
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
    plt.plot(ball[0], ball[1], 'go')
    plt.text(ball[0], ball[1], 'ball')
    plt.plot(X[0, 4], X[0, 5], 'ro')
    plt.text(X[0, 4], X[0, 5], 'start')
    plt.plot(X[-1, 4], X[-1, 5], 'ro')
    plt.text(X[-1, 4], X[-1, 5], 'End')
    plt.figtext(0.15, 0.12, "Uk: {}".format(uk))
    number = "%03d" % step
    #plt.savefig(figPath + 'step_' + number)
    # plt.text(X[-1, 4], X[-1, 5]+0.03, str(cost))
    plt.show(block=False)
    # print "cost: " +str(cost)
    # print(X[0,4],X[0,5])
    plt.pause(0.5)
    plt.clf()



def TrainErrorPlot(errHistory):
    history=np.asarray(errHistory)
    xaxis=range(len(history))
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    plt.figure(1)
    ax=plt.subplot()
    plt.ion()
    #ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE error')
    plt.plot(xaxis,history)
    plt.title('Training Error')
    plt.pause(0.5)
    plt.clf()

def OnlineErrorPlot(onlineErrHistory):
    history=np.asarray(onlineErrHistory)
    xaxis=range(len(history))
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    plt.figure(2)
    ax=plt.subplot()
    plt.ion()
    #ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Error')
    plt.plot(xaxis, history)
    plt.title('Online Error')
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()




def plot_next_pos(xk1_pred, xk1_real):
    xk1_real_ = np.reshape(np.copy(xk1_real), (1, 8))
    # from numpy import *
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    plt.ion()
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
    plt.plot(xk1_pred[0, 4], xk1_pred[0, 5], 'ro')
    plt.text(xk1_pred[0, 4], xk1_pred[0, 5], 'xk_pred')
    plt.plot(xk1_real_[0, 4], xk1_real_[0, 5], 'ro')
    plt.text(xk1_real_[0, 4] + 0.02, xk1_real_[0, 5] + 0.02, 'xk1_real')
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()
    # plt.close(fig)


def solveRiccati(A, B, Pk, Q, R):

    At = np.transpose(A)
    Bt = np.transpose(B)
    AtPk = np.matmul(At, Pk)
    BtPk = np.matmul(Bt, Pk)

    AtPkA = np.matmul(AtPk, A)
    AtPkB = np.matmul(AtPk, B)
    BtPkA = np.matmul(BtPk, A)
    BtPkB = np.matmul(BtPk, B)

    inv_BtPkB_R = inv(BtPkB + R)  # (bt*Pk*B + R)^-1
    t_ = np.matmul(AtPkB, inv_BtPkB_R)  # (At*Pk*B) * (bt*Pk*B + R)^-1
    t = np.matmul(t_, BtPkA)           # (At*Pk*B) * (bt*Pk*B + R)^-1 * (Bt*Pk*A)

    result = AtPkA - t + Q
    return np.copy(result)

def getError(A, B, model, uk, xnu, xreal):
    x = xnu[0, 0:6]
    x = np.hstack((x, uk))
    netOut = model.predict(x)
    x = np.transpose(x[0, 0:6])
    ABOout = np.matmul(A, x)
    ABOout = ABOout + np.matmul(B, np.transpose(uk))
    ABOout = np.transpose(ABOout)
    print "net-real: " + str(mean_squared_error(netOut, xreal))
    print "AB-real: " + str(mean_squared_error(ABOout, xreal))
   # print "AB-net: " + str(mean_squared_error(ABOout, netOut))
    return




def LqrFhD(A,B,Q,R,N=10):
    PN=Q
    At=np.transpose(A)
    Bt=np.transpose(B)
    for i in range(0,N-1):
        a1=np.matmul(At,PN)
        a2=np.matmul(a1,A)+Q # At*Pk*A+Q
        a3=np.matmul(At,PN)
        a4=np.matmul(a3,B)  # At*Pk*B
        a5=np.matmul(Bt,PN)
        a6=np.matmul(a5,B)+R # Bt*Pk*B+R
        a7=inv(a6)            # inv(R+Bt*Pk*B)
        a8=np.matmul(Bt,PN)
        a9=np.matmul(a8,A)    # Bt*Pk*A

        b1=np.matmul(a4,a7) # (At*Pk*B)*inv(R+Bt*Pk*B)
        b2=-np.matmul(b1,a9)+a2 # At*Pk*A+Q-(At*Pk*B)*inv(R+Bt*Pk*B)*(Bt*Pk*A)

        PN=b2

    c1=np.matmul(Bt,PN)
    c2=np.matmul(c1,B)+R # Bt*Pk*B+R
    c3=inv(c2)
    c4=np.matmul(Bt,PN)
    c5=np.matmul(c4,A)  # Bt*Pk*A
    F=np.matmul(c3,c5) # inv(Bt*Pk*B+R)*(Bt*Pk*A)
    return F





'''