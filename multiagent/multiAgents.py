# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # food related
        foodList = newFood.asList()
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDistance = min(foodDistances)
            score += 2 / (closestFoodDistance + 1) #getting closer to food

        # ghost related
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        for ghost, dist, scaredTime in zip(newGhostStates, ghostDistances, newScaredTimes):
            if scaredTime > 0:  # ghosts are scared
                score += 2 / (dist + 1)  # good if close to ghosts
            else:
                if dist < 2:  # ghosts arent scared
                    score -= 10 / (dist + 1)# bad if close to ghosts
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    #decides what the best move is for pacman
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, state):
            # terminal state or depth limit reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                # pacman the only max player
                return maxValue(agentIndex, depth, state) #pacman = max
            else:
                return minValue(agentIndex, depth, state) #ghosts = min


        def maxValue(agentIndex, depth, state):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            # max value of successor states
            maxEval = float('-inf') #lowest val so that we try to increase
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action) #succesor after our action
                maxEval = max(maxEval, minimax(1, depth, successor))  # get rec call for ghost move
            return maxEval

        def minValue(agentIndex, depth, state):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            minEval = float('inf') #highest so we try to decrease
            nextAgent = agentIndex + 1  # next ghost or Pacman

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)

                if nextAgent == state.getNumAgents():
                    # if next is pacman -> increase depth, restart agent index (pacman = 0)
                    minEval = min(minEval, minimax(0, depth + 1, successor))
                else:
                    #if next is another ghost
                    minEval = min(minEval, minimax(nextAgent, depth, successor))
            return minEval

        # best action for Pacman
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(1, 0, successor)  # G1
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # same as min max + alpha and beta vals
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            # terminal state or depth limit reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                #pacman the only max player
                return maxValue(agentIndex, depth, state, alpha, beta)  #pacman = max
            else:
                return minValue(agentIndex, depth, state, alpha, beta) #ghosts = min

        def maxValue(agentIndex, depth, state, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            maxEval = float('-inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action) #succesor after our action
                maxEval = max(maxEval, alphaBeta(1, depth, successor, alpha, beta))  # g1
                if maxEval > beta: # !!! PRUNE
                    return maxEval
                alpha = max(alpha, maxEval)  # update alpha as max player
            return maxEval

        def minValue(agentIndex, depth, state, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            minEval = float('inf') #highest so we try to decrease
            nextAgent = agentIndex + 1  # next ghost or Pacman

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if nextAgent == state.getNumAgents():
                    # if next is pacman -> increase depth, restart agent index (pacman = 0)
                    minEval = min(minEval, alphaBeta(0, depth + 1, successor, alpha, beta))
                else:
                    # if next is another ghost
                    minEval = min(minEval, alphaBeta(nextAgent, depth, successor, alpha, beta))

                if minEval < alpha: # !!! PRUNE
                    return minEval
                beta = min(beta, minEval) # update beta as min player
            return minEval

        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successor, alpha, beta)  # g1
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
