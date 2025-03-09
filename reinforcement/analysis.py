#python gridworld.py -m # analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

#good
def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    #low discount -> small horizon, aka short term focus
    #high discount  -> focus on future rewards, encourage long term planning

    #low living reward -> shorter path
    #high living reward -> longer path

    #low noise -> encourages pontentially risky strategy
    #high noise -> encourages safe strategy
    answerDiscount = 0.3 #prioritize close reward
    answerNoise = 0.01 #encourage staying near cliff
    answerLivingReward = -2 #short path
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """

    answerDiscount = 0.7 #prioritize future reward bcs we need to take long way
    answerNoise = 0.5 #encourage safe strategy
    answerLivingReward = -2 #pick close exit
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """

    answerDiscount = 0.9 #prioritize distant exit with big rew
    answerNoise = 0.1 #we go with risky strategy
    answerLivingReward = -0.5 #so we prioritize shorter
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

#good
def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = 0.9 #prioritize distant exit with big rew
    answerNoise = 0.4 #safe strategy
    answerLivingReward = 0.2 #its ok to take longer path
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

#good
def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """

    answerDiscount = 0 #ignore rewards
    answerNoise = 0 #no risk whatever we do
    answerLivingReward = 1 # can be anything (ok with -1, -0.5, 1, 2..)
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
