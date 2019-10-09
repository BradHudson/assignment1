import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():

    # Small Problems
    i = 0

    iterations_array = []
    rhc_fitness_array = []
    rhc_time_array = []

    #Plot CP
    with open('Results/Small/Continuous_Peaks_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                iterations_array.append(int(row[0]))
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Small/Continuous_Peaks_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Small/Continuous_Peaks_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Small/Continuous_Peaks_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_sa_fitness_array = []
    analyze_sa_time_array = []
    i = 0

    with open('Results/Small/Continuous_Peaks_SA_Cooling.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_sa_fitness_array.append(float(row[1]))
                analyze_sa_time_array.append(float(row[2]))

    plt.plot([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95], analyze_sa_fitness_array, label='SA')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Fitness vs SA Cooling', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Cooling')
    plt.savefig('images/Small/Continuous_Peaks_SA_Cooling_Fitness.png')
    plt.close()

    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], analyze_sa_time_array, label='SA')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Time vs SA Cooling', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Cooling')
    plt.savefig('images/Small/Continuous_Peaks_SA_Cooling_Time.png')
    plt.close()

    #fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    #plt.xlim(10000)
    plt.savefig('images/Small/Continuous_Peaks_Fitness.png')
    plt.close()



    #time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    #plt.xlim(10000)
    plt.savefig('images/Small/Continuous_Peaks_Time.png')
    plt.close()


    i = 0
    rhc_fitness_array = []
    rhc_time_array = []

    #Plot FlipFlop
    with open('Results/Small/FlipFlop_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Small/FlipFlop_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Small/FlipFlop_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Small/FlipFlop_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_mimic_fitness_array = []
    analyze_mimic_time_array = []
    i = 0
    generate_list = [10, 20, 40, 80, 120, 200]
    keep_list = [5, 10, 20, 40, 60, 100]

    with open('Results/Small/FlipFlop_MIMIC_GenKeep.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_mimic_fitness_array.append(float(row[1]))
                analyze_mimic_time_array.append(float(row[2]))

    plt.plot(['10-5','20-10','40-20','80-40','120-60','200-100'], analyze_mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Fitness vs Mimic Generate/Keep', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Generate-Keep')
    plt.savefig('images/Small/FlipFlop_MIMIC_GenKeep_Fitness.png')
    plt.close()

    plt.plot(['10-5','20-10','40-20','80-40','120-60','200-100'], analyze_mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Time vs Mimic Generate/Keep', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Generate-Keep')
    plt.savefig('images/Small/FlipFlop_MIMIC_GenKeep_Time.png')
    plt.close()

    #fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    #plt.xlim(10000)
    plt.savefig('images/Small/FlipFlop_Fitness.png')
    plt.close()



    #time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    #plt.xlim(10000)
    plt.savefig('images/Small/FlipFlop_Time.png')
    plt.close()

    i = 0
    rhc_fitness_array = []
    rhc_time_array = []

    #Plot TSP
    with open('Results/Small/TSP_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Small/TSP_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Small/TSP_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Small/TSP_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_ga_fitness_array = []
    analyze_ga_time_array = []
    i = 0
    pop_list = [10, 20, 50, 100, 200, 500]
    mate_list = [5, 10, 20, 50, 100, 250]
    mutate_list = [2, 5, 5, 10, 10, 20]

    with open('Results/Small/TSP_GA_pop_mate_mutate.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_ga_fitness_array.append(float(row[1]))
                analyze_ga_time_array.append(float(row[2]))

    plt.plot(['10-5-2', '20-10-5', '50-20-5', '100-50-10', '200-100-10', '500-250-20'], analyze_ga_fitness_array, label='GA')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Fitness vs GA Pop/Mate/Mutate', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Pop/Mate/Mutate')
    plt.savefig('images/Small/TSP_GA_pop_mate_mutate_Fitness.png')
    plt.close()

    plt.plot(['10-5-2', '20-10-5', '50-20-5', '100-50-10', '200-100-10', '500-250-20'], analyze_ga_time_array, label='GA')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Time vs GA Pop/Mate/Mutate', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Pop/Mate/Mutate')
    plt.savefig('images/Small/TSP_GA_pop_mate_mutate_Time.png')
    plt.close()

    #fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    #plt.xlim(10000)
    plt.savefig('images/Small/TSP_Fitness.png')
    plt.close()



    #time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.savefig('images/Small/TSP_Time.png')
    plt.close()


    #Large Problems

    i = 0

    iterations_array = []
    rhc_fitness_array = []
    rhc_time_array = []

    # Plot CP
    with open('Results/Large/Continuous_Peaks_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                iterations_array.append(int(row[0]))
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Large/Continuous_Peaks_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Large/Continuous_Peaks_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Large/Continuous_Peaks_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_sa_fitness_array = []
    analyze_sa_time_array = []
    i = 0

    with open('Results/Large/Continuous_Peaks_SA_Cooling.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_sa_fitness_array.append(float(row[1]))
                analyze_sa_time_array.append(float(row[2]))

    plt.plot([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95], analyze_sa_fitness_array, label='SA')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Fitness vs SA Cooling', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Cooling')
    plt.savefig('images/Large/Continuous_Peaks_SA_Cooling_Fitness.png')
    plt.close()

    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], analyze_sa_time_array, label='SA')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Time vs SA Cooling', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Cooling')
    plt.savefig('images/Large/Continuous_Peaks_SA_Cooling_Time.png')
    plt.close()

    # fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/Continuous_Peaks_Fitness.png')
    plt.close()

    # time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('Continous Peaks Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/Continuous_Peaks_Time.png')
    plt.close()

    i = 0
    rhc_fitness_array = []
    rhc_time_array = []

    # Plot FlipFlop
    with open('Results/Large/FlipFlop_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Large/FlipFlop_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Large/FlipFlop_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Large/FlipFlop_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_mimic_fitness_array = []
    analyze_mimic_time_array = []
    i = 0
    generate_list = [10, 20, 40, 80, 120, 200]
    keep_list = [5, 10, 20, 40, 60, 100]

    with open('Results/Large/FlipFlop_MIMIC_GenKeep.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_mimic_fitness_array.append(float(row[1]))
                analyze_mimic_time_array.append(float(row[2]))

    plt.plot(['10-5','20-10','40-20','80-40','120-60','200-100'], analyze_mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Fitness vs Mimic Generate/Keep', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Generate-Keep')
    plt.savefig('images/Large/FlipFlop_MIMIC_GenKeep_Fitness.png')
    plt.close()

    plt.plot(['10-5','20-10','40-20','80-40','120-60','200-100'], analyze_mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Time vs Mimic Generate/Keep', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Generate-Keep')
    plt.savefig('images/Large/FlipFlop_MIMIC_GenKeep_Time.png')
    plt.close()

    # fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/FlipFlop_Fitness.png')
    plt.close()

    # time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('FlipFlop Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/FlipFlop_Time.png')
    plt.close()

    i = 0
    rhc_fitness_array = []
    rhc_time_array = []

    # Plot TSP
    with open('Results/Large/TSP_RHC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                rhc_fitness_array.append(float(row[1]))
                rhc_time_array.append(float(row[2]))

    sa_fitness_array = []
    sa_time_array = []
    i = 0

    with open('Results/Large/TSP_SA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                sa_fitness_array.append(float(row[1]))
                sa_time_array.append(float(row[2]))

    ga_fitness_array = []
    ga_time_array = []
    i = 0

    with open('Results/Large/TSP_GA.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                ga_fitness_array.append(float(row[1]))
                ga_time_array.append(float(row[2]))

    mimic_fitness_array = []
    mimic_time_array = []
    i = 0

    with open('Results/Large/TSP_MIMIC.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                mimic_fitness_array.append(float(row[1]))
                mimic_time_array.append(float(row[2]))

    analyze_ga_fitness_array = []
    analyze_ga_time_array = []
    i = 0
    pop_list = [10, 20, 50, 100, 200, 500]
    mate_list = [5, 10, 20, 50, 100, 250]
    mutate_list = [2, 5, 5, 10, 10, 20]

    with open('Results/Large/TSP_GA_pop_mate_mutate.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if i == 0:
                i = 1
                continue
            else:
                analyze_ga_fitness_array.append(float(row[1]))
                analyze_ga_time_array.append(float(row[2]))

    plt.plot(['10-5-2', '20-10-5', '50-20-5', '100-50-10', '200-100-10', '500-250-20'], analyze_ga_fitness_array,
             label='GA')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Fitness vs GA Pop/Mate/Mutate', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Pop/Mate/Mutate')
    plt.savefig('images/Large/TSP_GA_pop_mate_mutate_Fitness.png')
    plt.close()

    plt.plot(['10-5-2', '20-10-5', '50-20-5', '100-50-10', '200-100-10', '500-250-20'], analyze_ga_time_array,
             label='GA')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Time vs GA Pop/Mate/Mutate', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Pop/Mate/Mutate')
    plt.savefig('images/Large/TSP_GA_pop_mate_mutate_Time.png')
    plt.close()

    # fitness

    plt.plot(iterations_array, rhc_fitness_array, label='RHC')
    plt.plot(iterations_array, sa_fitness_array, label='SA')
    plt.plot(iterations_array, ga_fitness_array, label='GA')
    plt.plot(iterations_array, mimic_fitness_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Fitness vs Iterations', fontdict={'size': 16})
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/TSP_Fitness.png')
    plt.close()

    # time

    plt.plot(iterations_array, rhc_time_array, label='RHC')
    plt.plot(iterations_array, sa_time_array, label='SA')
    plt.plot(iterations_array, ga_time_array, label='GA')
    plt.plot(iterations_array, mimic_time_array, label='MIMIC')

    plt.legend(loc=4, fontsize=8)
    plt.title('TSP Time vs Iterations', fontdict={'size': 16})
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    # plt.xlim(10000)
    plt.savefig('images/Large/TSP_Time.png')
    plt.close()

    # i = 0
    #
    # #Plot NN-Backprop
    #
    # i = 0
    #
    # #Plot NN-RHC
    #
    # i = 0
    #
    # #Plot NN-SA
    #
    # i = 0
    #
    # #Plot NN-GA

    print('done')

if __name__== "__main__":
  main()