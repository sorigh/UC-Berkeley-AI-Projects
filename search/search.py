# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


class Node:
    def __init__(self, state, action, parent):
        self.state = state
        self.action = action
        self.parent = parent

    def __repr__(self):
        return f"Node(state={self.state}, action={self.action})"


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack
    frontier = Stack()
    frontier.push(Node(problem.getStartState(), None, None))

    expanded = set()

    # cat timp exista noduri in frontiera de expandat
    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_state =  current_node.state

        if problem.isGoalState(current_state):
            return reconstruct_path(current_node)

        if current_state in expanded:
            continue

        # it was already considered
        expanded.add(current_state)

        for next_state, action, _ in problem.getSuccessors(current_state):
            #if the next possible state wasnt already considered/ explored we add it now
            if next_state not in expanded:
                # create the next node for search problem
                next_node = Node(next_state, action, current_node)
                # se adauga in frontiera toate nodurile posibile
                frontier.push(next_node)

    return []  # Return an empty path if no solution is found

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    """
    #before node
    frontier = util.Queue()
    frontier.push((problem.getStartState(), []))
    expanded = []
    while not frontier.isEmpty():
        current, path = frontier.pop()
        if problem.isGoalState(current):
            return path
        else:
            expanded.append(current)
            for nextstate, action, _ in problem.getSuccessors(current):
                if nextstate not in expanded:
                    frontier.push((nextstate, path + [action]))
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    start_node = Node(problem.getStartState(), None, None)
    frontier.push(start_node)

    #check = [problem.getStartState()]
    expanded = set()

    while not frontier.isEmpty():
        current_node = frontier.pop()
        current_state = current_node.state
        if problem.isGoalState(current_state):
            return reconstruct_path(current_node)

        #or check in expanded directly without using check
        if current_state in expanded:
            continue
        expanded.add(current_state)

        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state not in expanded: #and next_state not in check:
                frontier.push(Node(next_state,action, current_node))
                #check.append(next_state) # we could use an additional check to check the expanded states
    return []



def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize the priority queue with the start node
    start_node = Node(state=problem.getStartState(), action=None, parent=None)
    frontier = PriorityQueue()
    frontier.push(start_node, 0)  # Node and priority (cost)

    expanded = set()  # Track expanded states to avoid revisiting them
    cost_so_far = {problem.getStartState(): 0}  # Cost to reach each state

    while not frontier.isEmpty():
        current_node = frontier.pop()  # Get the node with the lowest cost
        current_state = current_node.state

        # If the current state is the goal, reconstruct the path
        if problem.isGoalState(current_state):
             return reconstruct_path(current_node)

        # Skip if the state has already been expanded
        if current_state in expanded:
            continue

        # Mark the state as expanded
        expanded.add(current_state)

        # Explore successors
        for next_state, action, step_cost in problem.getSuccessors(current_state):
            new_cost = cost_so_far[current_state] + step_cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                next_node = Node(state=next_state, action=action, parent=current_node)
                frontier.push(next_node, new_cost)

    return []  # Return an empty path if no solution is found

def reconstruct_path(node: Node) -> List[Directions]:
    """Reconstruct the path from the goal node to the start node."""
    path = []
    while node.parent is not None:  # Traverse up to the root
        path.append(node.action)
        node = node.parent
    return list(reversed(path))  # Reverse the path to get the correct order

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    frontier = util.PriorityQueue()
    expanded = {}  # lowest g for each state
    start_node = Node(problem.getStartState(), None, None)
    # frontier str: (node, g), h
    frontier.push((start_node, 0), heuristic(problem.getStartState(), problem))

    while not frontier.isEmpty():
        current_node, g = frontier.pop() #current node & g

        if problem.isGoalState(current_node.state):
            return reconstruct_path(current_node)

        # if it hasn't been expanded, or we found a cheaper path
        if current_node.state not in expanded or g < expanded[current_node.state]:
            expanded[current_node.state] = g  #lowest cost to reach current state

            for next_state, action, step_cost in problem.getSuccessors(current_node.state):
                new_g = g + step_cost
                new_h = heuristic(next_state, problem)
                f = new_h + new_g  # f = g + h

                next_node = Node(state=next_state, action=action, parent=current_node)
                frontier.push((next_node, new_g), f)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
