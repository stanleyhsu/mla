import svmMLiA
import matplotlib.pyplot as plt
from numpy import *

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols = []
    for l in lst:
        if l == -1:
            cols.append('red')
        else:
            cols.append('blue')
    return cols


# Create the colors list using the function above

# Function to map the colors as a list from the input list of x variables
def pltXY(lst):
    X = [];
    Y = []
    for l in lst:
        X.append(l[0])
        Y.append(l[1])
    return X, Y


def pltData(dataArr, labelArr) :
    cols = pltcolor(labelArr)
    X, Y = pltXY(dataArr)
    plt.scatter(X, Y, s=5, c=cols)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    for i in range(100):
        if alphas[i] > 0.0: print(i, dataArr[i], labelArr[i])
    print(b)
    pltData(dataArr, labelArr)


