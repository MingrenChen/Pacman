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


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # *** YOUR CODE HERE ***
        score = 0

        # The distances of ghosts.
        ghost_positions = [ghost.getPosition() for ghost in newGhostStates]
        distances_ghosts = [manhattanDistance(i, newPos) for i in ghost_positions]

        # The distances for food
        food_positions = newFood.asList()
        distances_foods = [manhattanDistance(i, newPos) for i in food_positions]

        if successorGameState.isLose():
            return -(10**5)

        if successorGameState.isWin():
            return 10 ** 5
        elif len(food_positions) == 1 and min(distances_ghosts) > 1:
            return 10 ** 5 - 1

        # try not to catch by ghost.
        if distances_ghosts == 1:
            return -500
        if len(food_positions) < len(currentGameState.getFood().asList()):
            score += 30

        # it's better if after move, pacman is far away from closest ghosts
        origin_ghost_positions = [i.getPosition() for i in currentGameState.getGhostStates()]
        origin_ghost_distance = min([manhattanDistance(i, currentGameState.getPacmanPosition())
                                     for i in origin_ghost_positions])
        if origin_ghost_distance < min(distances_ghosts):
            score += 10

        # it's better if after move, pacman is closer from closest food.
        origin_food_distance = min([manhattanDistance(i, currentGameState.getPacmanPosition())
                                    for i in currentGameState.getFood().asList()])
        if origin_food_distance > min(distances_foods):
            score += 20

        if action == Directions.STOP:
            score -= 100

        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minValue(self, gameState, numGhosts, depth):
        """
        Returns the min action of this ghost.
        numGhosts represents the number of unmoved ghosts in this turn.
        """
        if depth > self.depth or gameState.isWin()or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Case need to go deeper.
        actions = gameState.getLegalActions(numGhosts)
        successors = [gameState.generateSuccessor(numGhosts, action) for action in actions]
        # Case there is no ghosts left.
        res = []
        for successor in successors:
            if numGhosts == (gameState.getNumAgents() - 1):
                res.append((self.maxValue(successor, 1, depth + 1)))
            else:
                res.append((self.minValue(successor, numGhosts +  1, depth)))
        minimum = min(res)
        minIndex = res.index(minimum)
        return [minimum, actions[minIndex]]


    def maxValue(self, gameState, numGhosts, depth):
        """
        Returns the max action of this pacman.
        numGhosts represents the number of unmoved ghosts in this turn.
        """
        if  depth > self.depth or gameState.isWin()or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Case need to go deeper.
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        # Case there is no ghosts left.
        res = []
        for successor in successors:
            res.append((self.minValue(successor, 1, depth))[1])
        maximum = max(res)
        maxIndex = res.index(maximum)

        return [maximum, actions[maxIndex]]


    def getAction(self, gameState):
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
        """
        # *** YOUR CODE HERE ***
        # List of successors of current state.
        if self.index == 0:
            max = self.maxValue(gameState, 1, 1)
            return max[1]
        else:
            min =  self.minValue(gameState, 1, 1)
            return min[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def abminValue(self, gameState, numGhosts, depth, alpha, beta):
        """
        Returns the min action of this ghost.
        numGhosts represents the number of unmoved ghosts in this turn.
        """
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Case need to go deeper.
        actions = gameState.getLegalActions(numGhosts)
        successors = [gameState.generateSuccessor(numGhosts, action) for action in actions]
        # Case there is no ghosts left.
        res = []
        for successor in successors:
            if numGhosts == (gameState.getNumAgents() - 1):
                res.append((self.abmaxValue(successor, 1, depth + 1, alpha, beta)))
            else:
                res.append((self.abminValue(successor, numGhosts + 1, depth, alpha, beta)))
        minimum = min(res)
        minIndex = res.index(minimum)
        return [minimum, actions[minIndex]]

    def abmaxValue(self, gameState, numGhosts, depth, alpha, beta):
        """
        Returns the max action of this pacman.
        numGhosts represents the number of unmoved ghosts in this turn.
        """
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Case need to go deeper.
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        # Case there is no ghosts left.
        res = []
        maximum = 0
        for successor in successors:
            cur = self.abminValue(successor, 1, depth, alpha, beta)
            res.append(cur)
            if cur > beta:
                return cur
            alpha = max(alpha, cur)
            maximum = max(maximum, cur)
        maxIndex = res.index(maximum)

        return [maximum, actions[maxIndex]]


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # *** YOUR CODE HERE ***

        if self.index == 0:
            max = self.abmaxValue(gameState, 1, 1)
            return max[1]
        else:
            min =  self.abminValue(gameState, 1, 1)
            return min[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction