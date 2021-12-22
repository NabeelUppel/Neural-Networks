import math
import numpy as np
from Data import DataSetup
import itertools

"""
DESCRIPTION OF NN:
INPUT LAYER: 2 INPUT NODES + 1 BIAS NODE
HIDDEN LAYER 1: 3 NODES + 1 BIAS NODE
OUTPUT LAYER: 1 NODE

ACTIVATION FUNCTION: SIGMOID
"""


def rand_weights_bias(neuronsFirst, neuronsSecond):
    weight_matrix = np.random.rand(neuronsFirst, neuronsSecond)
    bias_vector = np.random.rand(neuronsSecond)
    return weight_matrix, bias_vector


def initialise(layerArray):
    output = []
    length = len(layerArray)
    for i in range(length + 1):
        if i + 2 <= length:
            x = layerArray[i:i + 2]
            y = rand_weights_bias(x[0], x[1])
            output.append(y)
    return output


def sigmoid(z):
    s = 1 / (1 + math.exp(-z))
    return s


def primeSigmoid(a):
    return a * (1 - a)


def forwardPropagation(network, dataPoint):
    layers = len(network)
    vFunct = np.vectorize(sigmoid)
    layerActivations = []
    dp = dataPoint
    for i in range(layers):
        w = network[i][0].T
        b = network[i][1]
        z = np.matmul(w, dp)
        z = np.add(z, b)
        a = vFunct(z)
        layerActivations.append(a)
        # layerSums.append(z)
        if i != layers - 1:
            dp = a
        else:
            return layerActivations


def fullPropagation(network, dataArray):
    layers = len(network)
    vFunct = np.vectorize(sigmoid)
    prediction = []
    for row in dataArray:
        dp = row
        for i in range(layers):
            w = network[i][0].T
            b = network[i][1]
            z = np.matmul(w, dp)
            z = np.add(z, b)
            a = vFunct(z)
            if i == layers - 1:
                prediction.append((a[0]))
            else:
                dp = a
    return prediction


def backPropagation(network, dataArray, labels, lamb):
    vFunct = np.vectorize(primeSigmoid)
    ERROR_ARRAY = []
    delta = []
    for n in network:
        x = np.zeros(n[0].shape)
        delta.append(x)

    weights, bias = zip(*network)
    regularise = weights * lamb

    for i in range(len(dataArray)):

        dp = dataArray[i]
        layerAct = forwardPropagation(network, dp)
        layersActLen = len(layerAct)
        target = labels[i]
        predict = layerAct[-1]
        error = predict - target

        for j in reversed(range(layersActLen)):
            ERROR_ARRAY.append(error)
            curr_Act = layerAct[j]
            g_prime = vFunct(curr_Act)
            w = network[j][0]
            f1 = np.multiply(g_prime, error)
            error = np.matmul(w, f1)

        gradient = gradientCalc(dp, layerAct, ERROR_ARRAY)
        delta = matrixCalc(delta, gradient, np.add)

    ##REGULARISATION NEEDED
    ARG = matrixCalc(delta, regularise, np.add)
    return ARG


def gradientDescent(network, dataArray, labels, alpha, epoch, epsilon, lamb):
    n = network
    x, y = zip(*n)

    curr_it = 0
    previous_step_size = StepCheck(x)
    while previous_step_size > epsilon and curr_it < epoch:
        allGradients = backPropagation(n, dataArray, labels, lamb)
        newWeights = []
        newBias = []
        weights, bias = zip(*n)
        for i in range(len(allGradients)):
            # UPDATE WEIGHTS
            gradient = allGradients[i]
            gradient = gradient * alpha
            w = weights[i]
            newW = np.subtract(w, gradient)
            newWeights.append(newW)

            # UPDATE BIASES
            b = bias[i]
            biasGradient = b * alpha
            newB = np.subtract(b, biasGradient)
            newBias.append(newB)
        previous_step_size = StepCheck(newWeights)
        curr_it += 1

        n = list(zip(newWeights, newBias))
    return n


def StepCheck(Array):
    norms = []
    for w in Array:
        x = np.linalg.norm(w)
        norms.append(x)
    return sum(norms) / len(Array)


def matrixCalc(listMatrix1, listMatrix2, function):
    newList = []

    for i in range(len(listMatrix1)):
        add = function(listMatrix1[i], listMatrix2[i])
        newList.append(add)
    return newList


def gradientCalc(act0, activations, deltaError):
    activations.insert(0, act0)
    d = []
    i_max = len(activations)
    j_max = len(deltaError)

    for i, j in zip(range(i_max), reversed(range(j_max))):
        e = deltaError[j].reshape(-1, 1)
        a = activations[i].reshape(-1, 1).T
        z = np.matmul(e, a)
        d.append(z.T)
    return d


def Counts(r0, r1, w0, w1, predicted, label):
    right1 = r1
    wrong1 = w1
    right0 = r0
    wrong0 = w0
    isOne = predicted == 1
    isZero = predicted == 0

    if label == 1 and isOne:
        right1 += 1
    elif label == 1 and isZero:
        wrong1 += 1
    elif label == 0 and isZero:
        right0 += 1
    elif label == 0 and isOne:
        wrong0 += 1
    return right0, right1, wrong0, wrong1


def confusionMatrix(predictions, labels):
    right1 = 0
    right0 = 0
    wrong1 = 0
    wrong0 = 0
    for i in range(len(predictions)):
        x = round(predictions[i])
        y = labels[i]
        right0, right1, wrong0, wrong1 = Counts(right0, right1, wrong0, wrong1, x, y)
    # CONFUSION MATRIX
    r = [" 0", "C0", "C1"]
    r0 = ["P0", right0, wrong1]
    r1 = ["P1", wrong0, right1]
    matrix = np.array([r, r0, r1])
    total = len(predictions)
    accuracy = sum(list(map(int, np.diagonal(matrix)))) / total * 100
    return matrix, accuracy


def errorCalc(predictions, labels):
    errors = []
    for i in range(len(predictions)):
        x = predictions[i]
        y = labels[i]
        errors.append((x - y) ** 2)
    return sum(errors) / len(errors)


def trainModel(data, network, iterations):
    labels = data[:, 2]
    data = data[:, :2]
    newNetwork = gradientDescent(network, data, labels, 0.1, iterations, 0.05, 1)
    return newNetwork


def Model(data, network):
    labels = data[:, 2]
    data = data[:, :2]
    predictions = fullPropagation(network, data)
    errors = errorCalc(predictions, labels)
    cm, accuracy = confusionMatrix(predictions, labels)
    return cm, errors


def hyperParametersConfiguration(trainingData, validationData, iterations):
    params = hyperParameters(2, 1)
    hp_perf = {}
    for i in range(len(params)):
        curr_hp = params[i]
        initNetwork = initialise(curr_hp)
        newNetwork = trainModel(trainingData, initNetwork, iterations)
        err = Model(validationData, newNetwork)[1]
        hp_perf[err] = curr_hp
    v = min(list(hp_perf.keys()))
    return hp_perf[v]


def hyperParameters(inputNodes, outputNodes):
    params = []
    original = [2, 3, 1]
    params.append(original)
    for i in range(10):

        size = np.random.randint(low=3, high=6)
        x = np.random.randint(low=3, high=6, size=size)
        x[0] = inputNodes
        x[-1] = outputNodes
        params.append(x.tolist())

    params.sort()
    params = list(num for num, _ in itertools.groupby(params))
    params.sort(key=len)

    return params


def main():
    trainingData = DataSetup(100).Data
    validationData = DataSetup(100).Data
    testingData = DataSetup(100).Data

    it = 1000

    initNetwork = initialise([2, 3, 1])
    # print("Learning Rate: 0.1","Epsilon:0.5", "Epoch:1000")
    cm0, err0 = Model(trainingData, initNetwork)
    print("Training Data through Network Before Weight Adjustments")
    print(cm0)
    print(f"Error: {err0}")

    print("\n")

    newNetwork1 = trainModel(trainingData, initNetwork, it)

    print("Training Data through Network After Weight Adjustments")
    cm1, err1 = Model(trainingData, newNetwork1)
    print(cm1)
    print(f"Error: {err1}")

    print("\n")

    cm2, err2 = Model(validationData, newNetwork1)
    print("Validation Data through Network After Weight Adjustments")
    print(cm2)
    print(f"Error: {err2}")
    print("\n")

    cm4, err4 = Model(testingData, newNetwork1)
    print("Testing Data through Network After Weight Adjustments")
    print(cm4)
    print(f"Error: {err4}")
    print("\n")



    print("Configuring Hyper-Parameters Using Validation Data")

    structure = hyperParametersConfiguration(trainingData, validationData, it)
    print("New Network Architecture", structure)
    print("\n")
    print("Training Data through New Network")
    hypNetwork = initialise(structure)
    newNetwork = trainModel(trainingData, hypNetwork, it)
    cm3, err3 = Model(testingData, newNetwork)
    print(cm3)
    print(f"Error: {err3}")





if __name__ == "__main__":
    main()
