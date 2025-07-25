import math
import constants
from life_tables import LifeTables
import numpy as np

class Population(object):
    '''
    Hold a cohort and run the markov model simulation.
    '''

    start_age = None
    sex = None
    NIHSS = None
    horizon = None

    def __init__(self, ais_outcomes, simtype):
        '''
        We get a dictionary containing (for ischemic stroke patients):
        'p_good', 'p_tpa', 'p_evt', 'p_transfer'
        '''

        # Current edit: TODO -> sketch out patients that we need to divide
        # out between the various states to capture costs
        # -> tpa, evt, etc.
        # does it make sense to do this here or does it make sense to do it
        # in a separate section of the model??
        self.ais_outcomes = ais_outcomes
        self.costs_per_year = []
        self.simtype = simtype
        self.states_in_markov = self.break_into_states()
        self.states = self.run_markov(self.states_in_markov,
                                      Population.start_age, Population.sex,
                                      num_years=None if Population.horizon == 'lifetime' else int(Population.horizon))
        self.qalys_per_year = get_qalys(self.states)
        self.costs_per_year = get_costs_per_year(self.states)
        self.qalys = simpsons_1_3rd_correction(self.qalys_per_year,
                                               Population.horizon)
        self.costs = simpsons_1_3rd_correction(self.costs_per_year,
                                               Population.horizon)

    def break_into_states(self):

        call_population = 1
        pop_mimic = call_population * constants.p_call_is_mimic()
        pop_hemorrhagic = call_population * constants.p_call_is_hemorrhagic()
        pop_ischemic = call_population - pop_mimic - pop_hemorrhagic

        states = np.zeros(constants.States.NUMBER_OF_STATES)
        # We assume that mimics are at gen pop (headache, migraine, etc.)
        states[constants.States.GEN_POP] += pop_mimic

        # Get the mRS breakdown of patients with acute ischemic strokes and
        # remember to adjust for population of ischemic patients when
        # adding into state matrix

        mrs_of_ais = np.array(constants.break_up_ais_patients(
            self.ais_outcomes['p_good'], Population.NIHSS))
        states_ischemic = pop_ischemic * mrs_of_ais

        # Now we need the mRS breakdown for patients with hemorrhagic strokes
        # Currently making the conservative estimate that there is no
        # difference in outcomes for ICH versus AIS patients, even though
        # there is evidence to suggest to suggest ICH patients do almost
        # about twice as well.
        # This estimate also adjusts hemorrhagic stroke outcomes based on
        # time to center.

        states_hemorrhagic = pop_hemorrhagic * mrs_of_ais

        # Add on first year costs
        baseline_year_one_costs = constants.first_year_costs(
            states_hemorrhagic, states_ischemic)
        baseline_year_one_costs += (
            constants.cost_ivt() * self.ais_outcomes['p_tpa'] * pop_ischemic)
        baseline_year_one_costs += (
            constants.cost_evt() * self.ais_outcomes['p_evt'] * pop_ischemic)
        baseline_year_one_costs += (
            constants.cost_transfer() * self.ais_outcomes['p_transfer'] *
            pop_ischemic)
        self.costs_per_year.append(baseline_year_one_costs)

        states = np.array([
            states[i] + states_ischemic[i] + states_hemorrhagic[i]
            for i in range(constants.States.NUMBER_OF_STATES)
        ])

        return states

    def run_markov(self, states, start_age, sex, num_years=None):
        '''
        import a list of states at each year from start to age 100
        '''
        age_arr = np.arange(start_age, start_age + num_years if num_years else 100)
        mrs_range = np.arange(constants.States.DEATH)
        all_states = np.zeros((len(age_arr) + 1, len(mrs_range) + 1))
        all_states[0] = states
        # current_age = start_age
        hazard_morts = constants.hazard_mort(mrs_range)
        p_dead_arr = LifeTables.adjusted_mortality(
                    sex, age_arr, hazard_morts)
        for i, p_dead in enumerate(p_dead_arr):
            # all_states.append(current_state)
            # We run a range up to death because we don't want to include death
            # since it only markovs to itself
            
            death_probs = (all_states[i,:-1] * p_dead)
            all_states[i + 1,:-1] = all_states[i,:-1] - death_probs
            all_states[i + 1, -1] = all_states[i, -1] + death_probs.sum()
            # for mrs in range(constants.States.DEATH):
            #     p_dead = LifeTables.adjusted_mortality(
            #         sex, current_age, constants.hazard_mort(mrs))
            #     current_state[constants.States.DEATH] += current_state[
            #         mrs] * p_dead
            #     current_state[mrs] -= current_state[mrs] * p_dead
            # current_age += 1
        # Add on final age
        # all_states.append(current_state)
        return all_states


def get_costs_per_year(states):
    continuous_discount = 0.03
    discreet_discount = np.pow(np.exp(continuous_discount), np.arange(states.shape[0]))
    costs_per_year = constants.annual_cost(states, axis=1) / discreet_discount
    return costs_per_year[1:]
    # for cycle, state in enumerate(states):
    #     # We added on cycle 0, first year, costs during the markov model since
    #     # it's dependent on hemorrhagic vs. ischemic
    #     if cycle == 0:
    #         continue
    #     else:
    #         costs = constants.annual_cost(state)
    #         costs /= ((1 + discreet_discount)**(cycle))
    #         costs_per_year.append(costs)


def get_qalys(states):
    '''
    states: numpy matrix where each row corresponds to a year and each column
    is a probability of being in the given mrs state.
    
    Returns an array of discounted quality-adjusted life-years at each year
    '''
    continuous_discount = 0.03
    discreet_discount = np.pow(np.exp(continuous_discount), np.arange(states.shape[0]))
    mrs_utilities = np.array([constants.utilities_mrs(mrs) 
                              for mrs in range(constants.States.DEATH)])
    qalys = (states[:,:-1] * mrs_utilities).sum(axis=1) / discreet_discount
    # for cycle, state in enumerate(states):
    #     qaly = 0
    #     for mrs in range(constants.States.DEATH):
    #         qaly += state[mrs] * constants.utilities_mrs(mrs)
    #     # Discount
    #     qaly /= ((1 + discreet_discount)**(cycle))
    #     qalys.append(qaly)

    return qalys


def simpsons_1_3rd_correction(yearly_value, years_horizon=None):
    '''
    Returns the sum of the one-dimensional array inputted for either
    discounted costs or QALYs. Default is to run a lifetime horizon, but
    can run for the correction for any number of years as long as it is
    specified.
    '''
    multiplier = 1 / 3
    sum_ = yearly_value[0] * multiplier
    start_index = 1
    end_index = len(yearly_value) - 1
    if years_horizon is not None and years_horizon <= end_index:
        end_index = years_horizon
    # Since for in range in [a, b)
    for i in range(start_index, end_index + 1):
        if i == end_index:
            multiplier = 1 / 3
        else:
            if i % 2 == 0:
                multiplier = 2 / 3
            else:
                multiplier = 4 / 3
        sum_ += (yearly_value[i] * multiplier)
    return sum_
