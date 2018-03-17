import numpy

def CalculateAB(xu, model, delta=0.001):
    A = numpy.ones((6, 6))
    B = numpy.ones((6, 2))
    net = model
    d1 = net.predict(xu)
    for i in range(0, 8):
        xu[0, i] = xu[0, i] + delta
        d2 = net.predict(xu)
        xu[0, i] = xu[0, i] - delta  # fix for next time
        if i < 6:
            A[:, i] = (d2 - d1) / delta
        else:
            B[:, i - 6] = (d2 - d1) / delta
    return [A, B]
