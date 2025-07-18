import sys
import os
import time
import ais_outcomes as ais
import cohort
import constants
import create_random_sets as random_sets
import optimal_strategy
import tqdm
import pandas as pd
import numpy as np

import input_patients

'''

For default probabilistic model, turn on random times and
random LVO, but turn off comparison between times and LVO.

Note that random times does NOT add variation to travel times;
those are always discreet estimates.

'''
SETTINGS = {
    # Options for simulation type are:
    # 1) 'Base Case' 2) 'Random Sets' 3) 'Input File'
    'Simulation Type': 'Input File',
    'Probabilistic Model': {
        'on': True,
        'Random Times': True,
        'Random LVO': True,
        'Compare Times vs. LVO': False,
        #Simulation time of 597.8173010349274 seconds for 1000 sets
        #Simulation time of 58.652158975601196 seconds for 100 sets
        # Simulation time of 0.9114828109741211 seconds for 1 set
        'evals per set': 100,
        'Results': [],
        'current_set_counter': 0
    },
    'Base Case Options': {
        'sex': constants.Sex.FEMALE,
        'age': 70,
        'RACE': 2,
        'time_since_symptoms': 120,
        'time_to_primary': 22,
        'time_to_comprehensive': 37,
        'transfer_time': 21
    },
    'Random Set Options': {
        'Number of Random Sets': 15,
        # Set options to none implies a random parameter,
        # if they're given a value here they are assumed to be a constant
        'sex': None,
        'age': None,
        'RACE': None,
        'time_since_symptoms': None,
        'time_to_primary': None,
        'time_to_comprehensive': None,
        'transfer_time': None
    },
    # Misc. options, especially for sensitivity analyses etc.
    'ICER Threshold': 100000,
    # Input either 'lifetime' or a string for the number of years
    # post-stroke to base the decision on.
    # Note, MUST BE STRING, i.e. '1' not 1
    'Horizon': 'lifetime'
}

if not os.path.isdir('output'):
    os.makedirs('output')


def random_out_name():
    options = SETTINGS['Random Set Options']
    args = []
    sex = options['sex']
    if sex is not None:
        args.append('M' if sex is constants.Sex.MALE else 'F')
    else:
        args.append('B')
    age = options['age']
    args.append(str(age) if age is not None else 'age')
    race = options['RACE']
    args.append(f'{race:.2}' if race is not None else 'RACE')
    for time_type in [
            'time_since_symptoms', 'time_to_primary', 'time_to_comprehensive',
            'transfer_time'
    ]:
        time = options[time_type]
        args.append(str(time) if time is not None else time_type)

    base = os.path.join('output', 'random_sets')
    if not os.path.isdir(base):
        os.makedirs(base)

    return os.path.join(base, '-'.join(args) + '.csv')


OUTPUT_FILE = None
INPUT_VARIABLES = [
    'sex', 'age', 'RACE', 'time_since_symptoms', 'time_to_primary',
    'time_to_comprehensive', 'transfer_time'
]
OUTPUT_VARIABLES = [
    'Optimal Location', 'Primary Cost', 'Primary QALYs', 'Comprehensive Cost',
    'Comprehensive QALYs', 'Drip and Ship Cost', 'Drip and Ship QALYs',
    'Location with Maximum Benefit', 'Horizon'
]
# For now ...
PROBABILITY_MODEL_OUTPUT = [
    'Percent Primary', 'Percent Comprehensive', 'Percent Drip and Ship',
    'Horizon'
]

PROBABILITY_MODEL_OUTPUT_COMPARISON = [
    'Random Times Percent Primary', 'Random Times Percent Comprehensive',
    'Random Times Percent Drip and Ship', 'Random LVO Percent Primary',
    'Random LVO Percent Comprehensive', 'Random LVO Percent Drip and Ship',
    'Random Both Percent Primary', 'Random Both Percent Comprehensive',
    'Random Both Percent Drip and Ship', 'Horizon'
]

STRATEGIES = ['Primary', 'Comprehensive', 'Drip and Ship']


def read_input_file():
    f = open('input/scenarios_default.csv', 'r')
    # Parse header
    f.readline()
    strategies = []
    for line in f:
        strategy = {}
        values = line.split(sep=',')
        if values[0] == 'male':
            strategy['sex'] = constants.Sex.MALE
        else:
            strategy['sex'] = constants.Sex.FEMALE
        strategy['age'] = int(values[1])
        if values[2] == 'NA':
            strategy['RACE'] = constants.nihss_to_race(float(values[3]))
        else:
            strategy['RACE'] = float(values[2])
        strategy['time_since_symptoms'] = float(values[4])
        strategy['time_to_primary'] = float(values[5])
        strategy['time_to_comprehensive'] = float(values[6])
        strategy['transfer_time'] = float(values[7])
        strategies.append(strategy)
        print(strategies)

    return strategies


input_patients.set_input_patients("female", 70, 9, 120, 'input/RACE9.csv')

# set up data frames for input patients and writing output, these maintain location information
df_input = pd.read_csv('input/RACE9.csv')
# df_input = df_input.sample(n=10, random_state=47)
df_output = df_input.copy(deep=True)
df_output['Percent Primary'] = np.nan
df_output['Percent Comprehensive'] = np.nan
df_output['Percent Drip and Ship'] = np.nan

def read_input_file_new():

    df_renamed = df_input.rename(columns={'CSC_travel_time_minutes': 'time_to_comprehensive', 'PSC_travel_time_minutes':'time_to_primary'})
    strategies = df_renamed[['sex', 'age', 'RACE', 'time_since_symptoms','time_to_primary','time_to_comprehensive','transfer_time']].to_dict(orient='records')
    for strat in strategies:
        if strat['sex'] == 'female':
            strat['sex'] = constants.Sex.FEMALE
        else:
            strat['sex'] = constants.Sex.MALE

    # print(strategies)
    # print(len(strategies))
    return strategies

def output_to_csv(output_file):
 # Save to CSV
    df_output.to_csv(output_file, index=False)
    print(f"wrote df_output to {output_file}")  

def print_model_output(results, arguments):
    for item in INPUT_VARIABLES:
        OUTPUT_FILE.write(str(arguments[item]) + ',')
    OUTPUT_FILE.write(results['Optimal Location'] + ',')
    for strategy in STRATEGIES:
        OUTPUT_FILE.write(str(results['Costs'][strategy]) + ',')
        OUTPUT_FILE.write(str(results['QALYs'][strategy]) + ',')
    OUTPUT_FILE.write(results['Location with Maximum Benefit'] + ',')
    OUTPUT_FILE.write(SETTINGS['Horizon'] + '\n')


def print_probabilistic_model_output(arguments):

    comparison = SETTINGS['Probabilistic Model']['Compare Times vs. LVO']

    # Standard printing of input variables

    for item in INPUT_VARIABLES:
        OUTPUT_FILE.write(str(arguments[item]) + ',')

    # However, what's different is that now we care about something
    # slightly different ...

    percents = {}
    p_good = {}
    p_tpa = {}
    p_evt = {}
    

    for strategy in STRATEGIES:
        if comparison:
            percents[strategy] = [0, 0, 0]
        else:
            percents[strategy] = 0
            p_good[strategy] = 0
            p_tpa[strategy] = 0
            p_evt[strategy] = 0


    multiplier = 1 / SETTINGS['Probabilistic Model']['evals per set'] * 100

    # pt_counter = 0
    for result in SETTINGS['Probabilistic Model']['Results']:
        if comparison:
            percents[result[0]['Optimal Location']][0] += multiplier
            percents[result[1]['Optimal Location']][1] += multiplier
            percents[result[2]['Optimal Location']][2] += multiplier
        else:
            for strategy in STRATEGIES:
                p_good[strategy] += result['p_good'][strategy]
                p_tpa[strategy] += result['p_tpa'][strategy]
                p_evt[strategy] += result['p_evt'][strategy]
            # p_good[strategy] += result['p_good'][strategy]
            # print(result['p_good'][strategy])
            # print(p_good[strategy])
            percents[result['Optimal Location']] += multiplier
        # print(pt_counter)
        # print(percents)
        # # df_output[df_output.loc[df_output.index[pt_counter], 'col2']]
        # pt_counter += 1
    
    # for strategy in p_good:
    #     p_good[strategy] =  float(round(p_good[strategy]/SETTINGS['Probabilistic Model']['evals per set'],2))
    # for strategy in p_tpa:
    #     p_tpa[strategy] =  float(round(p_tpa[strategy]/SETTINGS['Probabilistic Model']['evals per set'],2))

    # for strategy in p_evt:
    #     p_evt[strategy] =  float(round(p_evt[strategy]/SETTINGS['Probabilistic Model']['evals per set'],2))


    # print("p_good_outcome: ", p_good)
    # print("p_tpa: ", p_tpa)
    # print("p_EVT: ", p_evt)

    if comparison:
        for strategy in STRATEGIES:
            OUTPUT_FILE.write(str(percents[strategy][0]) + ',')
        for strategy in STRATEGIES:
            OUTPUT_FILE.write(str(percents[strategy][1]) + ',')
        for strategy in STRATEGIES:
            OUTPUT_FILE.write(str(percents[strategy][2]) + ',')
    else:
        for strategy in STRATEGIES:
            OUTPUT_FILE.write(str(percents[strategy]) + ',')
    OUTPUT_FILE.write(SETTINGS['Horizon'] + '\n')

    # a stupid way to match output with the correct row of the input df by finding the first NAN value, but it works

    first_nan_index = df_output['Percent Primary'].isna().idxmax()  # Gets the index of the first NaN

    df_output.loc[first_nan_index, 'Percent Primary'] = percents['Primary']
    df_output.loc[first_nan_index, 'Percent Comprehensive'] = percents['Comprehensive']
    df_output.loc[first_nan_index, 'Percent Drip and Ship'] = percents['Drip and Ship']

    # print(percents)

    # Prepare the probabilstic model results for the next set
    SETTINGS['Probabilistic Model']['Results'].clear()
    SETTINGS['Probabilistic Model']['current_set_counter'] = 0


def run_model(arguments):
    '''
    Probabilities of good outcomes are set to a dictionary containing:
    "Primary", "Comprehensive" and "Drip and Ship".
    Calculated for a patient with AIS, cohort stratification into mimics
    and hemorrhagic strokes take place later.
    '''
    global SETTINGS

    if SETTINGS['Probabilistic Model']['Random Times'] is True:
        constants.Times.get_random_set()
    else:
        constants.Times.set_to_default()

    ais_model = setup_model(arguments)

    results = {
        'Optimal Location': None,
        'Location with Maximum Benefit': 'Based on cutoff',
        'Costs': {
            'Primary': 0,
            'Comprehensive': 0,
            'Drip and Ship': 'N/A'
        },
        'QALYs': {
            'Primary': 0,
            'Comprehensive': 0,
            'Drip and Ship': 'N/A'
        },
        'p_good':{
            'Primary': 0,
            'Comprehensive': 0,
            'Drip and Ship': 'N/A',
        },
         'p_tpa':{
            'Primary': 0,
            'Comprehensive': 0,
            'Drip and Ship': 'N/A',
        },
         'p_evt':{
            'Primary': 0,
            'Comprehensive': 0,
            'Drip and Ship': 'N/A',
        }
    }

    # Early exit if the location is based only on RACE cutoff (because
    # no treatment options)
    if ais_model.model_is_necessary is not True:
        results['Optimal Location'] = ais_model.cutoff_location
        return results

    strategies = None
    max_qaly = {'strategy': 'N/A', 'QALYs': 0}
    if ais_model.run_primary_then_ship() is False:
        strategies = STRATEGIES[:-1]
    else:
        strategies = STRATEGIES
    for strategy in strategies:
        ischemic_outcomes = ais_model.get_ais_outcomes(strategy)
        # print(ischemic_outcomes)
        results['p_good'][strategy] = ischemic_outcomes['p_good']
        results['p_tpa'][strategy] = ischemic_outcomes['p_tpa']
        results['p_evt'][strategy] = ischemic_outcomes['p_evt']

        markoved_population = cohort.Population(ischemic_outcomes, strategy)
        results['Costs'][strategy] = markoved_population.costs
        results['QALYs'][strategy] = markoved_population.qalys
        if results['QALYs'][strategy] > max_qaly['QALYs']:
            max_qaly['QALYs'] = results['QALYs'][strategy]
            max_qaly['strategy'] = strategy

    results['Location with Maximum Benefit'] = max_qaly['strategy']
    optimal_strategy.get_optimal(results, strategies,
                                 SETTINGS['ICER Threshold'])
    return results


def setup_model(arguments):
    cohort.Population.start_age = arguments['age']
    cohort.Population.sex = arguments['sex']
    cohort.Population.NIHSS = constants.race_to_nihss(arguments['RACE'])
    if SETTINGS['Horizon'] == 'lifetime':
        cohort.Population.horizon = None
    else:
        cohort.Population.horizon = int(SETTINGS['Horizon'])
    return ais.IschemicModel(arguments,
                             SETTINGS['Probabilistic Model']['Random LVO'])


def setup_output_file(output_file):
    for item in INPUT_VARIABLES:
        output_file.write(item + ',')
    if SETTINGS['Probabilistic Model']['on'] is True:
        if SETTINGS['Probabilistic Model']['Compare Times vs. LVO']:
            output_variables = PROBABILITY_MODEL_OUTPUT_COMPARISON
        else:
            output_variables = PROBABILITY_MODEL_OUTPUT
    else:
        output_variables = OUTPUT_VARIABLES
    for item in output_variables[:-1]:
        output_file.write(item + ',')
    output_file.write(output_variables[-1] + '\n')

#running probabilitstic model model
def run_probabilistic_set(argument_set):
    prob_settings = SETTINGS['Probabilistic Model']
    if prob_settings['Compare Times vs. LVO'] is True:
        # Start with random times only
        prob_settings['Random Times'] = True
        prob_settings['Random LVO'] = False
        r1 = run_model(argument_set)
        # Set time back to default and rerun with random LVO distribution
        prob_settings['Random Times'] = False
        prob_settings['Random LVO'] = True
        r2 = run_model(argument_set)
        prob_settings['Random Times'] = True
        prob_settings['Random LVO'] = True
        r3 = run_model(argument_set)
        results = (r1, r2, r3)
    else:
        # The random lvo'ness is taken care of by passing as an argument
        # in the run_model function (for get ischemic outcomes)
        if prob_settings['Random Times'] is True:
            constants.Times.get_random_set()
        results = run_model(argument_set)
    return results

#loop through to runmodel 1000 times (evals per set)
def run_argument_set(argument_set):
    # Alias because annoying to keep typing
    prob_settings = SETTINGS['Probabilistic Model']
    if prob_settings['on'] is True:
        while (prob_settings['current_set_counter'] <
               prob_settings['evals per set']):
            results = run_probabilistic_set(argument_set)
            # print(results)
            prob_settings['current_set_counter'] += 1
            prob_settings['Results'].append(results)
        print_probabilistic_model_output(argument_set)
    else:
        results = run_model(argument_set)
        print_model_output(results, argument_set)
    # print(prob_settings['Results'])


def run():

    global OUTPUT_FILE

    # Setup the inputs and the argument files.

    arguments = None

    if SETTINGS['Simulation Type'] == 'Base Case':
        OUTPUT_FILE = open('output/base_case.csv', 'w')
        arguments = []
        arguments.append(SETTINGS['Base Case Options'])
    elif SETTINGS['Simulation Type'] == 'Random Sets':
        OUTPUT_FILE = open(random_out_name(), 'w')
        arguments = tqdm.tqdm(
            random_sets.create_random_sets(SETTINGS['Random Set Options'])
        )
    elif SETTINGS['Simulation Type'] == 'Input File':
        OUTPUT_FILE = open('output/input_file_scenarios.csv', 'w')
        arguments = tqdm.tqdm(read_input_file_new())

    setup_output_file(OUTPUT_FILE)
    for argument_set in arguments:
        run_argument_set(argument_set)
    
    output_to_csv('output/RACE9.csv')


if __name__ == '__main__':
    START = time.time()
    constants.Costs.inflate(2016)
    run()
    END = time.time()
    print('Simulation time of', END - START, 'seconds.')
