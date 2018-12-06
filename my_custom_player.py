
import math
import random
import sys

from isolation import DebugState, play
from isolation.isolation import _HEIGHT, _WIDTH
from sample_players import DataPlayer

class MCTSNode():
    """
      state: current state,
      parent: pointer to Parent Node
      action: what action did we take to get here?
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

        self.visits = 0 # n
        self.score = 0 # t
        self.children = []

    # is this node a leaf node? (leaf nodes have no children)
    def leaf(self):
        return True if len(self.children) == 0 else False

    def sampled(self):
        return True if self.visits > 0 else False

    # v̅ symbol
    def avg_score(self):
        if self.visits == 0:
            return float("inf")

        return self.score / self.visits

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    total_visits = 0

    def random_get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
      
        #dbstate = DebugState.from_state(state)
        #print(dbstate)
        self.queue.put(random.choice(state.actions()))

    # SELECTION: bias choice of child nodes
    def traversal(self, children=[]):
        if (len(children) == 0):
            return None

        non_visited_nodes = [c for c in children if c.visits == 0]
        if len(non_visited_nodes) > 0:
            return non_visited_nodes[0]

        # UCB1 (Si) = v̅ + 2 √(ln N/n) --> where
        #   v̅ is average score
        #   N is total visits
        #   n is node visits
        def UCB1(node):
            return float('inf') if node.visits == 0 \
              else node.avg_score() \
                + (2 * math.sqrt(math.log(self.total_visits)/node.visits))

        return max(children, key=lambda x: UCB1(x))

    # EXPANSION: create child nodes
    def expansion(self, node):
        for action in node.state.actions():
            node.children.append(
              MCTSNode(
                node.state.result(action), # returns state
                node, # parent node
                action))

        if len(node.children) == 0:
            return None

        return node.children[0]

    # SIMULATION: randomly play out a game from a given node
    #  returns moves if we won and None otherwise
    def rollout(self, state, moves=0):
        moves += 1
        if state.terminal_test():
            return moves if state.utility(self.player_id)> 0 else None

        return self.rollout(
          state.result(
            random.choice(state.actions())
          ), moves
        )

    # BACKPROPAGATION: pass the result up to parent nodes
    def backpropagation(self, node, score=0):
        node.score += score
        node.visits += 1

        if node.parent is None: # at base
            self.total_visits = node.visits
            return

        self.backpropagation(node.parent, score)

    def monte_carlo(self, node):
        if node.state.terminal_test():
            return

        if node.leaf(): # no children
            if not node.sampled(): # visits are 0
                node = self.expansion(node)
                if (node is None): # sterile (can't produce children)
                    return

            moves = self.rollout(node.state)
            score = (100 - moves) if moves is not None else 0
            
            self.backpropagation(node, score)
            return
        
        selected_child = self.traversal(node.children)
        self.monte_carlo(selected_child)

    def get_action(self, state):
        if state.terminal_test():
            return

        base_node = MCTSNode(state)
        first_child = self.expansion(base_node)
        
        while True:
            if state.terminal_test():
                return

            self.monte_carlo(
              first_child if self.total_visits == 0 else base_node
            )

            choice_node = self.traversal(base_node.children)
            self.queue.put(choice_node.action)
