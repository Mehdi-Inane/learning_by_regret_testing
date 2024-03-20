import numpy as np
import random

# Payoffs are in the form: {(Player1_action, Player2_action): (Player1_payoff, Player2_payoff)}
payoffs = {('C', 'C'): (3, 3), ('C', 'D'): (0, 5),
           ('D', 'C'): (5, 0), ('D', 'D'): (1, 1)}


#Strategy is of form {'action':probability}
def choose_action(distraction_rate,strategy):
    if random.random() < distraction_rate: # If player is distracted
        return random.choice(list(strategy.keys())),True
    else: # Follow the policy of player
        actions, probabilities = zip(*strategy.items())
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        return chosen_action,False

def update_stuff(rand,action,payoff,random_actions,policy_actions):
    if rand : 
        if action not in random_actions.keys():
            random_actions[action] = [payoff]
        else:
            random_actions[action].append(payoff)
    else:
        if action not in policy_actions.keys():
            policy_actions[action] = [payoff]
        else:
            policy_actions[action].append(payoff)

def change_strategy(random_actions,policy_actions,tolerance):
    average_payoff_policy = 0
    for action in policy_actions :
        average_payoff_policy += np.sum(policy_actions[action])
    action_counts = 0
    for action in policy_actions:
        action_counts += len(policy_actions[action])
    if action_counts:
        average_payoff_policy /= action_counts
    regret = -average_payoff_policy
    if regret < -5:
        print(regret)
        print()
    for action in random_actions:
        regret = np.mean(random_actions[action]) - average_payoff_policy
        if regret > tolerance:
            return regret,True
    return regret,False

def simulate_day(payoffs,s,distraction_rate,policy_p1,policy_p2,tolerance):
    random_actions_p1 = {}
    random_actions_p2 = {}
    policy_actions_p1 = {}
    policy_actions_p2 = {}
    for i in range(s):
        # Player 1 selects an action
        action1,rand1 = choose_action(distraction_rate,policy_p1)
        # Player 2 selects an action
        action2,rand2 = choose_action(distraction_rate,policy_p2)
        payoff1,payoff2 = payoffs[(action1,action2)]
        # Update players memory 
        update_stuff(rand1,action1,payoff1,random_actions_p1,policy_actions_p1)
        update_stuff(rand2,action2,payoff2,random_actions_p2,policy_actions_p2)
        regret1,change1 = change_strategy(random_actions_p1,policy_actions_p1,tolerance)
        _,change2 = change_strategy(random_actions_p2,policy_actions_p2,tolerance)
    return change1,change2,regret1



def generate_random_permutations(n,h,P_i): # If more than two actions
    s = set()
    while len(s) < P_i:
        random_list = random.sample(list(range(0,h+1)),n)
        if sum(random_list) == h:
            s.add(tuple(random_list))
    return s

def generate_all_possible_frac_simplex(h,actions):
    s = []
    for i in range(h+1):
        dico = {}
        dico[actions[0]] = i/h
        dico[actions[1]] = 1 - i/h
        s.append(dico)
    return s


def generate_probabilities(n,h,P_i,actions):
    distributions = generate_random_permutations(n,h,P_i)
    dico_list = []
    for distribution in distributions:
        dico = {action : 0 for action in actions}
        for i,action in enumerate(actions):
            dico[action] = distribution[i] / h
        dico_list.append(dico)
    return dico_list

from tqdm import tqdm
def regret_testing(h,P_i,payoffs,T,s,distraction_rate,tolerance):
    # Init policies
    pay_keys = list(payoffs.keys())
    actions_p1 = list(set([action[0] for action in pay_keys]))
    actions_p2 = list(set([actions[1] for actions in pay_keys]))
    # Generate hats
    if P_i == 2:
        distributions_p1 = generate_all_possible_frac_simplex(h,actions_p1)
        distributions_p2 = generate_all_possible_frac_simplex(h,actions_p2)
    else:
        distributions_p1 = generate_probabilities(len(actions_p1),h,P_i,actions_p1)
        distributions_p2 = generate_probabilities(len(actions_p2),h,P_i,actions_p2)
    # Initial hat
    current_dist_p1 = random.choice(distributions_p1)
    current_dist_p2 = random.choice(distributions_p2)
    regrets = []
    for t in range(T):
        change_p1,change_p2,regret = simulate_day(payoffs,s,distraction_rate,current_dist_p1,current_dist_p2,tolerance)
        if change_p1:
            current_dist_p1 = random.choice(distributions_p1)
        if change_p2:
            current_dist_p2 = random.choice(distributions_p2)
        regrets.append(regret)
    return current_dist_p1,current_dist_p2,regrets

import matplotlib.pyplot as plt
epsilon = 1
print('epsilon = ',epsilon)
tolerance = (epsilon ** 2) / 48
distraction_rate = tolerance / 16
print(f'tolerance = {tolerance}')
print(f'distraction rate = {distraction_rate}')
h = int((8 * np.sqrt(2)) / tolerance)
print(f'log value {(np.log((1e5 * 2) / (epsilon**2 * tolerance **7) ))}')
s = int(((1000 * 4) /(distraction_rate * tolerance **2)) * np.log((1e5 * 2) / (epsilon**2 * tolerance **7) ))
print(f'h = {h}')
print(f's = {s}')
print(f'log of s {np.log(float(s))}')
P_i = 2
T = 1000
current_dist1,current_dist2,regrets,probas = regret_testing(h,P_i,payoffs,T,s,distraction_rate,tolerance)
plt.plot(list(range(T)),probas)
plt.xlabel('Iterations')
plt.ylabel('Probability of action D')
plt.savefig('probs.png', dpi=300)
plt.close()