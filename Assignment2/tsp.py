import sys
sys.path.append("./ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import opt.RandomizedHillClimbing as RHC
import opt.SimulatedAnnealing as SA
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GHC
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.ga.GenericGeneticAlgorithmProblem as GGAP
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GPOP
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
from array import array
from itertools import product
import time
import java.util.Random as Random
import random

# Small problem
random.seed(200)

N = 10

iteration_list = [10,100,500,1000,2500,5000,8000,10000]
runs = 5

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = Random().nextDouble()
    points[i][1] = Random().nextDouble()

#RHC
output_directory = "Results/Small/TSP_RHC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')


for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    rhc_total = 0
    rhc_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
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
output_directory = "Results/Small/TSP_SA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    sa_total = 0
    sa_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        sa_problem = SA(1E11, .95, hill_problem)
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
output_directory = "Results/Small/TSP_GA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        crossover = TravelingSalesmanCrossOver(fitness)
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
output_directory = "Results/Small/TSP_MIMIC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(runs):
        ranges = array('i', [N] * N)
        fitness = TravelingSalesmanSortEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
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

#GA Analysis
output_directory = "Results/Small/TSP_GA_pop_mate_mutate.csv"
pop_list = [10,20,50,100,200,500]
mate_list = [5,10,20,50,100,250]
mutate_list = [2,5,5,10,10,20]
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(pop_list)):
    pop = pop_list[i]
    mate = mate_list[i]
    mutate = mutate_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        crossover = TravelingSalesmanCrossOver(fitness)
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GGAP(fitness, discrete_dist, discrete_mutation, crossover)

        start = time.clock()
        ga_problem = StandardGeneticAlgorithm(pop, mate, mutate, genetic_problem)
        fit = FixedIterationTrainer(ga_problem, 5000)
        fit.train()
        end = time.clock()
        full_time = end - start
        ga_total += fitness.value(ga_problem.getOptimal())
        ga_time += full_time

    ga_total_avg = ga_total/runs
    ga_time_avg = ga_time/runs

    data = '{},{},{}\n'.format(str(pop) + '-' + str(mate) + '-' + str(mutate), ga_total_avg, ga_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

# #Large problem
random.seed(200)

N = 60

iteration_list = [10,100,500,1000,2500,5000,8000,10000]
runs = 5

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = Random().nextDouble()
    points[i][1] = Random().nextDouble()

#RHC
output_directory = "Results/Large/TSP_RHC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')


for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    rhc_total = 0
    rhc_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
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
output_directory = "Results/Large/TSP_SA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    sa_total = 0
    sa_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        hill_problem = GHC(fitness, discrete_dist, discrete_neighbor)

        start = time.clock()
        sa_problem = SA(1E11, .95, hill_problem)
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
output_directory = "Results/Large/TSP_GA.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        crossover = TravelingSalesmanCrossOver(fitness)
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
output_directory = "Results/Large/TSP_MIMIC.csv"
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(iteration_list)):
    iteration = iteration_list[i]
    mimic_total = 0
    mimic_time = 0

    for x in range(1):
        ranges = array('i', [N] * N)
        fitness = TravelingSalesmanSortEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
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

    mimic_total_avg = mimic_total/1
    mimic_time_avg = mimic_time/1

    data = '{},{},{}\n'.format(iteration, mimic_total_avg, mimic_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)

#GA Analysis
output_directory = "Results/Large/TSP_GA_pop_mate_mutate.csv"
pop_list = [10,20,50,100,200,500]
mate_list = [5,10,20,50,100,250]
mutate_list = [2,5,5,10,10,20]
with open(output_directory, 'w') as f:
    f.write('iterations,fitness,time\n')

for i in range(len(pop_list)):
    pop = pop_list[i]
    mate = mate_list[i]
    mutate = mutate_list[i]
    ga_total = 0
    ga_time = 0

    for x in range(runs):
        ranges = array('i', [2] * N)
        fitness = TravelingSalesmanRouteEvaluationFunction(points)
        discrete_dist = DiscretePermutationDistribution(N)
        discrete_neighbor = SwapNeighbor()
        discrete_mutation = SwapMutation()
        crossover = TravelingSalesmanCrossOver(fitness)
        discrete_dependency = DiscreteDependencyTree(.1, ranges)
        genetic_problem = GGAP(fitness, discrete_dist, discrete_mutation, crossover)

        start = time.clock()
        ga_problem = StandardGeneticAlgorithm(pop, mate, mutate, genetic_problem)
        fit = FixedIterationTrainer(ga_problem, 5000)
        fit.train()
        end = time.clock()
        full_time = end - start
        ga_total += fitness.value(ga_problem.getOptimal())
        ga_time += full_time

    ga_total_avg = ga_total/runs
    ga_time_avg = ga_time/runs

    data = '{},{},{}\n'.format(str(pop) + '-' + str(mate) + '-' + str(mutate), ga_total_avg, ga_time_avg)
    print(data)
    with open(output_directory, 'a') as f:
        f.write(data)