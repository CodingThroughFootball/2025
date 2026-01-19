import numpy as np
from scipy.special import factorial

LAMBDA = 1.6
DEFAULT_GOALS_COUNT = 7

def poisson(x, lam):
    '''Calculates probability of scoring x goals, given mean count of goals as lam. '''
    return (np.exp(-lam) * lam**x) / factorial(x)

def normalize_pmf(x):
    '''Normalizes the probability mass function. '''
    return x/np.sum(x)


def simulate(home_team, away_team, home_pts, away_pts, strength_factor=300, matches=1):
    '''Prints the results of the matches simulated using modified Poisson distribution
       using FIFA Ranking points of the two teams. '''
    goals = np.arange(0, DEFAULT_GOALS_COUNT +1)
    home_lam = max(0.1, LAMBDA + (home_pts-away_pts)/strength_factor)
    away_lam = max(0.1, LAMBDA + (away_pts-home_pts)/strength_factor)
    home_prob = normalize_pmf(poisson(goals, lam=home_lam))
    away_prob = normalize_pmf(poisson(goals, lam=away_lam))
    
    home_goals = np.random.choice(goals, matches, p=home_prob)
    away_goals = np.random.choice(goals, matches, p=away_prob)
    
    for _ , (home_goal,away_goal) in enumerate(zip(home_goals,away_goals)):
        print(f"{home_team} {home_goal}-{away_goal} {away_team}")
  
if __name__ == "__main__":
    simulate(home_team= "Argentina",
             away_team= "Ghana",
             home_pts= 1873,
             away_pts= 1351,
             strength_factor=400,
             matches=5)



