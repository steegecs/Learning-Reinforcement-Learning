import numpy as np
from copy import deepcopy


class Node: 
    def __init__ (self, parent_id, action, state, player): 
        self.id = parent_id + (action,)
        self.parent = parent_id
        self.state = state
        self.actions = []
        self.player = player
        self.reward = 0
        self.visits = 0
        self.q = 0

class Tree: 
    def __init__ (self): 
        self.nodes = {}
    
    def add_node(self, parent_id, action, state, player=None):
        self.nodes[parent_id + (action,)] = Node(parent_id, action, state, player)

class MCTS:
    def __init__(self, env, n_iterations=10000, depth=15, exploration_constant=10.0):

        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_simulation_count = 0
        self.env = env

        self.tree = Tree()
        self.tree.add_node((), 0, env.state, env.player)

    def selection(self): 

        leaf_node_found = False
        leaf_node_id = (0,)

        while not leaf_node_found:

            node = self.tree.nodes[leaf_node_id]
            
            if len(node.actions) == 0:
                leaf_node_id = node.id
                leaf_node_found = True
            else: 
                MAX_UCT = -100
                for action_idx, action in enumerate(node.actions):
                    child = self.tree.nodes[node.id + (action_idx,)]

                    # prevent divide by zero where child.visits == 0
                    # Setting n to a low value also guarantees that unexplored nodes will be explored
                    n = child.visits
                    if n == 0:
                        n = 1e-4

                    exploitation_value = child.reward / n
                    exploration_value  = np.sqrt(np.log(self.total_simulation_count) / n )
                    uct_value = exploitation_value + self.exploration_constant * exploration_value

                    if uct_value > MAX_UCT:
                        MAX_UCT = uct_value
                        leaf_node_id = child.id

        depth = len(leaf_node_id) # as node_id records selected action set

        return leaf_node_id, depth
                    

    def expansion(self, leaf_node_id):
        '''
        create all possible outcomes from leaf node
        in: tree, leaf_node
        out: expanded tree (self.tree),
             randomly selected child node id (child_node_id)
        '''
        leaf_node = self.tree.nodes[leaf_node_id]
        winner = self.env.get_winner(leaf_node.state)

        child_node_id = leaf_node.id
        if winner is None:
            '''
            when leaf state is not terminal state
            '''
            leaf_node.actions = list(map(tuple, self.env.get_legal_actions(leaf_node.state)))
            for action_idx, action in enumerate(leaf_node.actions):
                state = deepcopy(leaf_node.state)

                if leaf_node.player == 1:
                    next_turn = -1
                    state[action] = 1
                else:
                    next_turn = 1
                    state[action] = -1

                #Node id is a tuple of action set
                self.tree.add_node(leaf_node_id, action_idx, state, next_turn)

            #Select a random action to simulate from the expanded node of tree
            random_action_idx = np.random.randint(low=0, high=len(leaf_node.actions), size=1)[0]
            child_node_id = leaf_node.id + (random_action_idx,)
        return child_node_id

    
    def simulation(self, child_node_id):
        '''
        simulate game from child node's state until it reaches the resulting state of the game.
        in:
        - child node id (randomly selected child node id from `expansion`)
        out:
        - winner ('o', 'x', 'draw')
        '''
        self.total_simulation_count += 1

        #Deep copy so as to not update the actual node
        state = deepcopy(self.tree.nodes[child_node_id].state)
        previous_player = deepcopy(self.tree.nodes[child_node_id].player)
        anybody_win = False

        while not anybody_win:
            winner = self.env.get_winner(state)
            if winner is not None:
                anybody_win = True
            else:
                possible_actions = self.env.get_legal_actions(state)
   
                # randomly choose action for simulation (= random rollout policy)
                rand_idx = np.random.randint(low=0, high=len(possible_actions), size=1)[0]
                action, _ = possible_actions[rand_idx]

                if previous_player == 1:
                    current_player = -1
                    state[action] = 1
                else:
                    current_player = 1
                    state[action] = -1

                previous_player = current_player
        return winner

    def backprop(self, child_node_id, winner):

        if winner == 0:
            reward = 0
        elif winner == 1:
            reward = 1
        else:
            reward = -1

        node_id = child_node_id
        while (True):
            self.tree.nodes[node_id].visits += 1
            self.tree.nodes[node_id].reward += reward
            self.tree.nodes[node_id].q = self.tree.nodes[node_id].reward / self.tree.nodes[node_id].visits
            parent_id = self.tree.nodes[node_id].parent
            if parent_id == (0,):
                self.tree.nodes[parent_id].visits += 1
                self.tree.nodes[parent_id].reward += reward
                self.tree.nodes[parent_id].q = self.tree.nodes[parent_id].reward / self.tree.nodes[parent_id].visits
                break
            else:
                node_id = parent_id

    def get_action(self):
        for i in range(self.n_iterations):
            leaf_node_id, depth_searched = self.selection()
            child_node_id = self.expansion(leaf_node_id)
            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)

            if depth_searched > self.depth:
                break

        current_node_id = (0,)
        all_qs = []
        all_actions = []
        for action_idx, action in enumerate(self.tree.nodes[current_node_id].actions): 
            all_qs.append(self.tree.nodes[current_node_id + (action_idx,)].q)
            all_actions.append(action)

        if self.env.player == 1:
            best_index = all_qs.index(max(all_qs))
            best_q = all_qs[best_index]
            best_action = all_actions[best_index]
            print("Best action:", best_action, ", with q value:", best_q, ", depth:", depth_searched)
            return best_action, best_q, depth_searched

        else:
            best_index = all_qs.index(min(all_qs))
            best_q = all_qs[best_index]
            best_action = all_actions[best_index]
            print("Best action:", best_action, ", with q value:", best_q, ", depth:", depth_searched)
            return best_action, best_q, depth_searched
