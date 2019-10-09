import sys
sys.path.append("./ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import opt.RandomizedHillClimbing as RHC
import opt.SimulatedAnnealing as SA
import opt.ga.SingleCrossOver as SCO
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GHC
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GGAP
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GPOP
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.FlipFlopEvaluationFunction as FlipFlopEvaluationFunction
from array import array
from itertools import product
import time
import random

#Small problem
random.seed(200)

N = 10

iteration_list = [10,100,500,1000,2500,5000,8000,10000]
runs = 5

#RHC
output_directory = "Results/Small/FlipFlop_RHC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')


for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    rhc_total = 0
    rhc_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        rhc_problem = RHC(hill_problem)
        fit = FixedIterationTrainer(rhc_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        rhc_total += fitness.value(rhc_problem.getOptimal())
        rhc_time += full_time

    rhc_total_avg = rhc_total/runs
    rhc_time_avg = rhc_time/runs

    data = '{},{},{}\n'.format(iteration, rhc_total_avg, rhc_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)


#SA
output_directory = "Results/Small/FlipFlop_SA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    sa_total = 0
    sa_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        sa_problem = SA(100, .5, hill_problem)
        fit = FixedIterationTrainer(sa_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        sa_total += fitness.value(sa_problem.getOptimal())
        sa_time += full_time

    sa_total_avg = sa_total/runs
    sa_time_avg = sa_time/runs

    data = '{},{},{}\n'.format(iteration, sa_total_avg, sa_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#GA
output_directory = "Results/Small/FlipFlop_GA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GGAP(fitness, discrete_dist, discrete_mutation, crossover)

        start = time.clock()
        ga_problem = StandardGeneticAlgorithm(200, 100, 10, genetic_problem)
        fit = FixedIterationTrainer(ga_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        ga_total += fitness.value(ga_problem.getOptimal())
        ga_time += full_time

    ga_total_avg = ga_total/runs
    ga_time_avg = ga_time/runs

    data = '{},{},{}\n'.format(iteration, ga_total_avg, ga_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#MIMIC
output_directory = "Results/Small/FlipFlop_MIMIC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GPOP(fitness, discrete_dist, discrete_dependency)

        start = time.clock()
        mimic_problem = MIMIC(200, 20, genetic_problem)
        fit = FixedIterationTrainer(mimic_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        mimic_total += fitness.value(mimic_problem.getOptimal())
        mimic_time += full_time

    mimic_total_avg = mimic_total/runs
    mimic_time_avg = mimic_time/runs

    data = '{},{},{}\n'.format(iteration, mimic_total_avg, mimic_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#MIMIC analysis
generate_list = [10,20,40,80,120,200]
keep_list = [5,10,20,40,60,100]
output_directory = "Results/Small/FlipFlop_MIMIC_GenKeep.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(generate_list)):
    generate = generate_list[i]
    keep = keep_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GPOP(fitness, discrete_dist, discrete_dependency)

        start = time.clock()
        mimic_problem = MIMIC(generate, keep, genetic_problem)
        fit = FixedIterationTrainer(mimic_problem, 1000)
        fit.train()
        end = time.clock()
        full_time = end - start
        mimic_total += fitness.value(mimic_problem.getOptimal())
        mimic_time += full_time

    mimic_total_avg = mimic_total/runs
    mimic_time_avg = mimic_time/runs

    data = '{},{},{}\n'.format(str(generate) + '-' + str(keep), mimic_total_avg, mimic_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#Large problem
random.seed(200)

N = 60

iteration_list = [10,100,500,1000,2500,5000,8000,10000]
runs = 5

#RHC
output_directory = "Results/Large/FlipFlop_RHC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')


for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    rhc_total = 0
    rhc_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        rhc_problem = RHC(hill_problem)
        fit = FixedIterationTrainer(rhc_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        rhc_total += fitness.value(rhc_problem.getOptimal())
        rhc_time += full_time

    rhc_total_avg = rhc_total/runs
    rhc_time_avg = rhc_time/runs

    data = '{},{},{}\n'.format(iteration, rhc_total_avg, rhc_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)


#SA
output_directory = "Results/Large/FlipFlop_SA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    sa_total = 0
    sa_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        sa_problem = SA(100, .5, hill_problem)
        fit = FixedIterationTrainer(sa_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        sa_total += fitness.value(sa_problem.getOptimal())
        sa_time += full_time

    sa_total_avg = sa_total/runs
    sa_time_avg = sa_time/runs

    data = '{},{},{}\n'.format(iteration, sa_total_avg, sa_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#GA
output_directory = "Results/Large/FlipFlop_GA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GGAP(fitness, discrete_dist, discrete_mutation, crossover)

        start = time.clock()
        ga_problem = StandardGeneticAlgorithm(200, 100, 10, genetic_problem)
        fit = FixedIterationTrainer(ga_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        ga_total += fitness.value(ga_problem.getOptimal())
        ga_time += full_time

    ga_total_avg = ga_total/runs
    ga_time_avg = ga_time/runs

    data = '{},{},{}\n'.format(iteration, ga_total_avg, ga_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#MIMIC
output_directory = "Results/Large/FlipFlop_MIMIC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GPOP(fitness, discrete_dist, discrete_dependency)

        start = time.clock()
        mimic_problem = MIMIC(200, 20, genetic_problem)
        fit = FixedIterationTrainer(mimic_problem, iteration)
        fit.train()
        end = time.clock()
        full_time = end - start
        mimic_total += fitness.value(mimic_problem.getOptimal())
        mimic_time += full_time

    mimic_total_avg = mimic_total/runs
    mimic_time_avg = mimic_time/runs

    data = '{},{},{}\n'.format(iteration, mimic_total_avg, mimic_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#MIMIC analysis
generate_list = [10,20,40,80,120,200]
keep_list = [5,10,20,40,60,100]
output_directory = "Results/Large/FlipFlop_MIMIC_GenKeep.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(generate_list)):
    generate = generate_list[i]
    keep = keep_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = FlipFlopEvaluationFunction()
        discrete_dist = DiscreteUniformDistribution(ranges)
        discrete_neighbor = DiscreteChangeOneNeighbor(ranges)
        discrete_mutation = DiscreteChangeOneMutation(ranges)
        crossover = SCO()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GPOP(fitness, discrete_dist, discrete_dependency)

        start = time.clock()
        mimic_problem = MIMIC(generate, keep, genetic_problem)
        fit = FixedIterationTrainer(mimic_problem, 1000)
        fit.train()
        end = time.clock()
        full_time = end - start
        mimic_total += fitness.value(mimic_problem.getOptimal())
        mimic_time += full_time

    mimic_total_avg = mimic_total/runs
    mimic_time_avg = mimic_time/runs

    data = '{},{},{}\n'.format(str(generate) + '-' + str(keep), mimic_total_avg, mimic_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)