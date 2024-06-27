import numpy as np


class Entropy:

    @staticmethod
    def _calcPI(image):
        """calculating number of I"""
        N = len(image)
        P = {}
        for i in image:
            if str(i) in P.keys():
                P[str(i)] += 1.0 / N
            else:
                P[str(i)] = 1.0 / N
        return P

    @staticmethod
    def _calcPIJ(image):
        """calculating probility of [IJ], given I already"""
        N = len(image)
        PIJ = {}
        for i in range(N - 1):
            k = str(image[i]) + "-" + str(image[i + 1])
            if k in PIJ.keys():
                PIJ[k] += 1.0 / (N - 1)
            else:
                PIJ[k] = 1.0 / (N - 1)
        return PIJ

    @staticmethod
    def _calcPIJK(image):
        N = len(image)
        PIJK = {}
        for i in range(N - 2):
            k = str(image[i]) + "-" + str(image[i + 1]) + "-" + str(image[i + 2])
            if k in PIJK.keys():
                PIJK[k] += 1.0 / (N - 2)
            else:
                PIJK[k] = 1.0 / (N - 2)

        return PIJK

    @staticmethod
    def _calcE(P):
        e = 0.0
        for k in P.keys():
            p = P[k]
            if p > 0.0:
                e += -p * np.log2(p)
        return e

    @staticmethod
    def _F1(image):
        P = Entropy._calcPI(image)
        return Entropy._calcE(P)

    @staticmethod
    def _F2(image):
        PIJ = Entropy._calcPIJ(image)
        return Entropy._calcE(PIJ)

    @staticmethod
    def _F3(image):
        PIJK = Entropy._calcPIJK(image)
        return Entropy._calcE(PIJK)

    @staticmethod
    def E1(image):
        return Entropy._F1(image)

    @staticmethod
    def E2(image):
        return Entropy._F2(image) - Entropy._F1(image)

    @staticmethod
    def E3(image):
        return Entropy._F3(image) - Entropy._F2(image)


def calcPI(image):
    """calculating number of I"""
    N = len(image)
    P = {}
    for i in image:
        if str(i) in P.keys():
            P[str(i)] += 1.0 / N
        else:
            P[str(i)] = 1.0 / N
    return P


def calcPIJ(image):
    """calculating probility of [IJ], given I already"""
    N = len(image)
    P = calcPI(image)
    PIJ = {}
    for i in range(N - 1):
        k = str(image[i]) + "-" + str(image[i + 1])
        if k in PIJ.keys():
            PIJ[k] += 1.0 / (N - 1)
        else:
            PIJ[k] = 1.0 / (N - 1)
    return PIJ


def calcPIJK(image):
    N = len(image)
    PIJK = {}
    M = N - 1
    for i in range(N - 2):
        k = str(image[i]) + "-" + str(image[i + 1]) + "-" + str(image[i + 2])
        if k in PIJK.keys():
            PIJK[k] += 1.0 / (N - 2)
        else:
            PIJK[k] = 1.0 / (N - 2)

    return PIJK


def calcE(P):
    e = 0.0
    for k in P.keys():
        p = P[k]
        if p > 0.0:
            e += -p * np.log2(p)
    return e


def F1(image):
    P = calcPI(image)
    return calcE(P)


def F2(image):
    PIJ = calcPIJ(image)
    return calcE(PIJ)


def F3(image):
    PIJK = calcPIJK(image)
    return calcE(PIJK)


def E1(image):
    return F1(image)


def E2(image):
    return F2(image) - F1(image)


def E3(image):
    return F3(image) - F2(image)


def test_run():
    size = 100000
    N1 = min(size, 256)
    N2 = min(size - 1.0, 256 * 256)
    N3 = min(size - 2.0, 256 * 256 * 256)
    e1 = np.log2(N1)
    e2 = np.log2(N2)
    e3 = np.log2(N3)
    print(e1, e2 - e1, e3 - e2)
    print("==============")
    for i in range(100):
        A = np.random.randint(256, size=size)
        f1 = F1(A)
        f2 = F2(A)
        f3 = F3(A)
        e1 = f1
        e2 = f2 - f1
        e3 = f3 - f2
        print(e1, e2, e3)
