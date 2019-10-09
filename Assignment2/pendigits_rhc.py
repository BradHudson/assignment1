import csv
import sys
sys.path.append("./ABAGAIL.jar")

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.tester.AccuracyTestMetric as AccuracyTestMetric
import shared.tester.ConfusionMatrixTestMetric as ConfusionMatrixTestMetric
import shared.tester.NeuralNetworkTester as NeuralNetworkTester
import shared.tester.TestMetric as TestMetric
import shared.tester.Tester;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory as FeedForwardNeuralNetworkFactory
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule
import opt.RandomizedHillClimbing as RandomizedHillClimbing
from func.nn.activation import RELU
import shared.Instance;
import time

iteration_set = [10,100,200, 400, 600, 800, 1000,2500,5000]
labels = [0,1,2,3,4,5,6,7,8,9]
train_instances = []
with open('wine_train.csv', "r") as data:
    file = csv.reader(data)

    for row in file:
        i = Instance([float(value) for value in row[:-1]])
        i.setLabel(Instance(float(row[-1])))
        train_instances.append(i)

validate_instances = []
with open('wine_validate.csv', "r") as data:
    file = csv.reader(data)

    for row in file:
        i = Instance([float(value) for value in row[:-1]])
        i.setLabel(Instance(float(row[-1])))
        train_instances.append(i)

test_instances = []
with open('wine_test.csv', "r") as data:
    file = csv.reader(data)

    for row in file:
        i = Instance([float(value) for value in row[:-1]])
        i.setLabel(Instance(float(row[-1])))
        train_instances.append(i)

train_times = []

for i in range(len(iteration_set)):
    factory = FeedForwardNeuralNetworkFactory()
    network = factory.createClassificationNetwork([16, 16, 1], RELU())
    measure = SumOfSquaresError()
    set = DataSet(train_instances)
    nno = NeuralNetworkOptimizationProblem(set, network, measure)

    o = RandomizedHillClimbing(nno)
    iteration = iteration_set[i]
    fit = FixedIterationTrainer(o, iteration)
    start = time.clock()
    fit.train()
    full_time = time.clock() - start

    train_times.append(full_time)

    opt = o.getOptimal()
    network.setWeights(opt.getData())

    acc = AccuracyTestMetric()
    t = NeuralNetworkTester(network, acc)
    t.test(validate_instances)

    acc.printResults()