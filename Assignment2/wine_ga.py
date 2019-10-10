import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
from func.nn.activation import RELU
import time
import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

def initialize_instances(file):
    instances = []

    with open(file, "r") as r:
        reader = csv.reader(r)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) <= 6 else 1))
            instances.append(instance)

    return instances

def train(oa, network, oaName, instances, measure,range):
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(range):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print "%0.03f" % error

def main():
    train_instances = initialize_instances('wine_train.csv')
    validate_instances = initialize_instances('wine_validate.csv')
    test_instances = initialize_instances('wine_test.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)
    iteration_list = [10, 100, 500, 1000, 2500, 5000]
    pop_list = [10, 20, 50, 100, 200, 500]
    mate_list = [5, 10, 20, 50, 100, 250]
    mutate_list = [2, 5, 5, 10, 10, 20]


    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([11, 22, 1], RELU())
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    with open("Results/NN/GA_Train.csv", 'w') as f:
        f.write('iterations,pop,mate,mutate,correct,accuracy,train_time,test_time\n')

    with open("Results/NN/GA_Validate.csv", 'w') as f:
        f.write('iterations,pop,mate,mutate,correct,accuracy,train_time,test_time\n')

    with open("Results/NN/GA_Test.csv", 'w') as f:
        f.write('iterations,pop,mate,mutate,correct,accuracy,train_time,test_time\n')

    for p in range(len(pop_list)):
        for i in range(len(iteration_list)):
            pop = pop_list[p]
            mate = mate_list[p]
            mutate = mutate_list[p]

            iteration = iteration_list[i]
            start = time.time()
            correct = 0
            incorrect = 0
            sim = StandardGeneticAlgorithm(pop, mate, mutate, nnop[0])

            train(sim, networks[0], oa_names[0], train_instances, measure,iteration)
            end = time.time()
            training_time = end - start

            optimal_instance = sim.getOptimal()
            networks[0].setWeights(optimal_instance.getData())

            start = time.time()
            for instance in train_instances:
                networks[0].setInputValues(instance.getData())
                networks[0].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[0].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1

            end = time.time()
            testing_time = end - start

            results += "\nResults for Training %s: \nCorrectly classified %d instances." % ('GA', correct)
            results += "\nIncorrectly classified Training %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

            data = '{},{},{},{},{},{},{},{}\n'.format(iteration, pop,mate,mutate, correct, float(correct)/(correct+incorrect)*100.0, training_time,testing_time)
            print(data)
            with open("Results/NN/GA_Train.csv", 'a') as f:
                f.write(data)

            correct = 0
            incorrect = 0

            for instance in validate_instances:
                networks[0].setInputValues(instance.getData())
                networks[0].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[0].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1

            results += "\nResults for Cross Validation %s: \nCorrectly classified %d instances." % ('GA', correct)
            results += "\nIncorrectly classified Cross Validation %d instances.\nPercent correctly classified: %0.03f%%" % (
            incorrect, float(correct) / (correct + incorrect) * 100.0)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

            data = '{},{},{},{},{},{},{},{}\n'.format(iteration, pop, mate, mutate, correct,
                                                      float(correct) / (correct + incorrect) * 100.0, training_time,
                                                      testing_time)
            print(data)
            with open("Results/NN/GA_Validate.csv", 'a') as f:
                f.write(data)

            correct = 0
            incorrect = 0

            for instance in test_instances:
                networks[0].setInputValues(instance.getData())
                networks[0].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[0].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1

            results += "\nResults for Testing %s: \nCorrectly classified %d instances." % ("GA", correct)
            results += "\nIncorrectly classified Testing %d instances.\nPercent correctly classified: %0.03f%%" % (
            incorrect, float(correct) / (correct + incorrect) * 100.0)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

            data = '{},{},{},{},{},{},{},{}\n'.format(iteration, pop, mate, mutate, correct,
                                                      float(correct) / (correct + incorrect) * 100.0, training_time,
                                                      testing_time)
            print(data)
            with open("Results/NN/GA_Test.csv", 'a') as f:
                f.write(data)



    print results

if __name__ == "__main__":
    main()

