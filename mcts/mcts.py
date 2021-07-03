import math
import random
import copy
import gc
import multiprocessing as mp


class MCTS:
    def __init__(self, action_space, c=math.sqrt(2), iteration_limit=300, horizon=80, branching_factor=4,
                 discount_factor=0.99, computation_budget=1000):
        """

        :param action_space:
        :param c:
        :param iteration_limit:
        :param horizon:
        :param branching_factor:
        :param discount_factor:
        :param computation_budget:
        """
        self.c = c
        self.iteration_limit = iteration_limit
        self.horizon = horizon
        self.branching_factor = branching_factor
        self.discount_factor = discount_factor
        self.computation_budget = computation_budget
        self.root = None
        self.action_space = action_space
        self.nodes = []
        self.node_count = 0
        self.executable = True

    def reset(self):
        """

        :return:
        """
        self.root = None
        self.nodes.clear()
        self.executable = True
        self.node_count = 0
        gc.collect()

    def policy(self, env):
        """

        :param env:
        :return:
        """
        self.reset()
        self.root = Node(state=env)
        for i in range(self.iteration_limit):
            if self.executable:
                self.construct_tree()
            else:
                break
        assert self.root.children, "root node must have children, otherwise MCTS was initialised with terminal state"
        best_node = self.return_policy()
        return best_node.action

    def construct_tree(self):
        """

        :return:
        """
        node = self.select(self.root)
        if self.executable:
            self.expand(node)

    def select(self, node):
        """

        :param node:
        :return:
        """
        if node.children:
            expandable_nodes = [child for child in node.children if not child.terminal and not child.fully_expanded]
            if expandable_nodes:
                best = max(expandable_nodes, key=self.ucb1)
                return self.select(best)
            else:
                node.terminal = True
                if node == self.root:
                    self.executable = False
                    return None
                return self.select(node.parent)
        else:
            return node

    def ucb1(self, node):
        """

        :param node:
        :return:
        """
        assert node.parent is not None, "Node {} must have a parent unless it is root node".format(node.node_id)
        exploitation = node.Q
        exploration = self.c * math.sqrt((math.log(node.parent.visits) / node.visits))
        return exploitation + exploration

    def expand(self, node):
        """

        :param node:
        :return:
        """
        assert node.terminal is False, "terminal nodes can't be expanded"
        assert node.fully_expanded is False, "fully expanded nodes can't be expanded"
        assert node.unexplored_actions, "no unexplored actions left to explore, node should be marked as fully expanded"
        nodes_for_simulation = []
        for _ in range(self.branching_factor):
            action = random.choice(node.unexplored_actions)
            model_copy = copy.deepcopy(node.state)
            _, reward, done, _ = model_copy.step(action)
            new_node = Node(model_copy, root=False, parent=node, action=action, depth=node.depth+1,
                            node_id=self.node_count+1, terminal=done)
            self.node_count += 1
            self.nodes.append(new_node)
            node.children.append(new_node)
            node.unexplored_actions.remove(action)
            if done or new_node.depth >= self.horizon:
                new_node.terminal = True
                self.backpropagate(reward, new_node, 0)
            else:
                nodes_for_simulation.append(new_node)
            if not node.unexplored_actions or len(node.children) >= self.branching_factor:
                node.fully_expanded = True
                break
        pool = mp.Pool(processes=mp.cpu_count())
        results = [pool.apply_async(self.simulate, args=(nfs,)) for nfs in nodes_for_simulation]
        discounted_returns = [r.get() for r in results]
        assert len(discounted_returns) == len(nodes_for_simulation), \
            "lengths of d_returns and nodes_for_sim should be equal "
        for d_return, n in zip(discounted_returns, nodes_for_simulation):
            self.backpropagate(d_return, n, 0)

    def simulate(self, node):
        """

        :param node:
        :return:
        """

        model_copy = copy.deepcopy(node.state)
        discounted_return = 0
        t = 0
        while not model_copy.done and t <= self.computation_budget:
            action = random.choice(self.action_space)
            _, reward, done, _ = model_copy.step(action)
            discounted_return += reward * (self.discount_factor ** t)
            t += 1
        return discounted_return

    def backpropagate(self, discounted_return, node, t):
        """

        :param discounted_return:
        :param node:
        :param t:
        :return:
        """
        node.visits += 1
        discounted_return *= self.discount_factor ** t
        node.Q_list.append(discounted_return)
        node.Q = sum(node.Q_list) / len(node.Q_list)
        if node.parent:
            self.backpropagate(discounted_return, node.parent, t+1)

    def return_policy(self):
        return max(self.root.children, key=lambda node: node.Q)


class Node:
    def __init__(self, state, root=True, parent=None, action=-1, depth=0, node_id=0,
                 terminal=False, fully_expanded=False):
        """

        :param state:
        :param root:
        :param parent:
        :param action:
        :param depth:
        :param node_id:
        :param terminal:
        :param fully_expanded:
        """
        self.root = root
        self.parent = parent
        self.state = state
        self.action = action
        self.terminal = terminal
        self.fully_expanded = fully_expanded
        self.children = []
        self.Q = 0.0
        self.Q_list = []
        self.visits = 1
        self.depth = depth
        self.unexplored_actions = copy.deepcopy(state.action_space)
        self.node_id = node_id

    def __str__(self):
        """

        :return:
        """
        return "Q / Reward: {}, visits: {}, terminal: {}, fully_expanded: {}, depth: {}, node_id: {}, root: {}" \
            .format(self.Q, self.visits, self.terminal, self.fully_expanded,
                    self.depth, self.node_id, self.root)
