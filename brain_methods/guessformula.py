#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Peter Petev"
__copyright__ = "Copyright 1/1/2019"
__credits__ = ["Peter Petev"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Peter Petev"
__email__ = "pspetev@gmail.com"
__status__ = "Development"

import random
import math
import copy
import numpy


class Node:
    """
    A class to define a single Node
    """
    def __init__(self, id, node_function):
        """
        Create a new Node

        :param id: pointer for the Node
        :param node_function: the function that the Node returns as an output as a function of inputs
        """
        # Variables to store information during a prediction
        self.touched = False
        self.incoming_signals = []
        self.value = 0

        # Identifiers and lists to show relative place
        self.id = id
        self.predecessors = []
        self.successors = []
        self.function = node_function

    def output(self):
        """
        A function of the incoming signals from previous Nodes

        :return: the Node's output function value
        """
        if self.function == "bias":
            return 1
        if self.function == "input":
            return self.value
        if self.function == "sigmoid":
            return 1 / (1 + math.exp(-sum(self.incoming_signals)))
        if self.function == "relu":
            return math.log(1 + math.exp(sum(self.incoming_signals)))
        if self.function == "abs":
            return abs(sum(self.incoming_signals))
        if self.function == "negate":
            return -sum(self.incoming_signals)
        if self.function == "square":
            return pow(sum(self.incoming_signals), 2)
        if self.function == "sine":
            return math.sin(sum(self.incoming_signals))
        if self.function == "add":
            return sum(self.incoming_signals)
        if self.function == "multiply":
            prod = 1
            for value in self.incoming_signals:
                prod = prod * value
            return prod
        if self.function == "polar":
            total = 0
            for value in self.incoming_signals:
                total += pow(value, 2)
            return math.sqrt(total)
        if len(self.incoming_signals) > 0:
            if self.function == "max":
                return max(self.incoming_signals)
            if self.function == "min":
                return min(self.incoming_signals)
        return sum(self.incoming_signals)

    def __str__(self):
        """
        Prints a Node

        :return: string value of a Node, including id and function
        """
        # return "Node: " + str(self.id) + ", Value: " + str(self.value) + ", function: " + str(self.function) + ", inc sigs:" + str(self.incoming_signals) + ", pred: " + str(self.predecessors) + ", succ: " + str(self.successors)
        return "Node: " + str(self.id) + ", function: " + str(self.function)


class Edge:
    """
    A class to define an edge between two Nodes
    """
    def __init__(self, nodes, from_id, to_id, weight):
        """
        Create an edge between two nodes, with a weight

        :param nodes: full Node list, so that Nodes can be updated with their relative place
        :param from_id: id of the Node giving the output signal
        :param to_id: id of the Node receiving the output signal
        :param weight: output signal multiplier
        """
        self.to_id = to_id
        self.from_id = from_id
        self.weight = weight

        # Update the nodes with their new links
        nodes[from_id].successors.append(to_id)
        nodes[to_id].predecessors.append(from_id)
        self.used = False

    def __str__(self):
        """
        Prints an Edge

        :return: string value of an Edge, including Nodes linked and weight
        """
        return str(self.from_id) + " -> " + str(self.to_id) + ": " + str(round(self.weight, 3))


class Structure:
    """
    A class to represent a Structure of Nodes and Edges
    """
    def __init__(self, number_inputs, number_outputs):
        """
        Create a Structure of Nodes and Edges

        :param number_inputs: number of inputs in the X matrix
        :param number_outputs: number of outputs in the Y matrix
        """
        self.fitness = 0
        self.tick = 0
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs

        # Important to store nodes and edges, but also inputs and outputs so they can be handled carefully
        self.nodes = {}
        self.ins = []
        self.outs = []
        self.edges = []

        # Create Input Nodes
        for i in range(number_inputs):
            identifier = self.new_id()
            self.ins.append(identifier)
            self.nodes[identifier] = Node(identifier, "input")

        # Create Output Nodes
        for i in range(number_outputs):
            identifier = self.new_id()
            self.outs.append(identifier)
            self.nodes[identifier] = Node(identifier, "output")

        # Link Inputs to Outputs through their own Edges
        for i in range(len(self.ins)):
            for j in range(len(self.outs)):
                self.edges.append(Edge(self.nodes, self.ins[i], self.outs[j], random_weight()))

    def print_nodes(self):
        """
        Print all Nodes in the Structure

        :return: string forms of each Node in the Structure
        """
        return [print(str(self.nodes[node])) for node in self.nodes.keys()]

    def print_edges(self):
        """
        Print all Edges in the Structure

        :return: sorted edges in the Structure, so following along is easier
        """
        self.edges = sorted(self.edges, key=lambda x: x.from_id, reverse=False)
        return [print(str(edge)) for edge in self.edges if edge.used]

    def find_fitness(self):
        """
        Find the Structure's fitness

        :return: a Structure's fitness, which is a function of error and size
        """
        error = 0

        # Find the sum of squared error the data points
        for data_point in range(len(Globals.X)):
            # Make a list of the order to send signals through
            active = []

            # Set input Nodes to current input value, and put in order
            for variable in range(len(self.ins)):
                self.nodes[self.ins[variable]].value = Globals.X[data_point][variable]
                active.append(variable)

            # Clear signals from last run
            for i in self.nodes.keys():
                self.nodes[i].touched = False
                self.nodes[i].incoming_signals = []

            # See if an edge is used in the final result
            for edge in self.edges:
                edge.used = False

            # Go through active list
            while len(active) > 0:
                current = active[0]
                for node in range(len(self.nodes[current].successors)):
                    if (not self.nodes[self.nodes[current].successors[node]].touched) or (
                            self.nodes[current].successors[node] in self.outs):
                        active.append(self.nodes[current].successors[node])
                        for edge in self.edges:
                            # Send each node's output signal to next node
                            if edge.from_id == current and edge.to_id == self.nodes[current].successors[node]:
                                self.nodes[self.nodes[current].successors[node]].incoming_signals.append(
                                    self.nodes[current].output() * edge.weight)
                                edge.used = True
                self.nodes[current].touched = True
                while current in active:
                    active.remove(current)
            # Find the error, square it, and add it to the total
            goal_output = Globals.Y[data_point]
            for out in range(len(self.outs)):
                error += pow(self.nodes[self.outs[out]].output() - goal_output[out], 2)
        # Create a fitness function to punish error and complexity
        self.fitness = pow(1 / (1 + error), len(self.nodes))
        return self.fitness

    def predict(self, x):
        """
        Predict a given input matrix

        :param x: input matrix
        :return: a prediction based on the alpha (best) Node
        """

        # Repeat fitness finding function, but store outputs instead of calculating error
        outputs = []
        for data_point in range(len(x)):
            outputs.append([])
            active = []
            for variable in range(len(self.ins)):
                self.nodes[self.ins[variable]].value = x[data_point][variable]
                active.append(variable)
            for i in self.nodes.keys():
                self.nodes[i].touched = False
                self.nodes[i].incoming_signals = []
            while len(active) > 0:
                current = active[0]
                for node in range(len(self.nodes[current].successors)):
                    if (not self.nodes[self.nodes[current].successors[node]].touched) or (
                            self.nodes[current].successors[node] in self.outs):
                        active.append(self.nodes[current].successors[node])
                        for edge in self.edges:
                            if edge.from_id == current and edge.to_id == self.nodes[current].successors[node]:
                                self.nodes[self.nodes[current].successors[node]].incoming_signals.append(
                                    self.nodes[current].output() * edge.weight)
                self.nodes[current].touched = True
                while current in active:
                    active.remove(current)
            for out in range(len(self.outs)):
                outputs[data_point].append(self.nodes[self.outs[out]].output())
        return outputs

    def mutate(self, probability):
        """
         Mutate random Nodes and Edges

        :param probability: probability of a Node undergoing each mutation
        :return: this Structure after mutation
        """

        # Add a node and two edges to two Nodes already connected by an Edge
        if random.random() < probability:
            replicated_edge = random.choice(self.edges)
            point_a = replicated_edge.from_id
            point_b = replicated_edge.to_id
            identifier = self.new_id()
            self.nodes[identifier] = Node(identifier, random.choice(Globals.node_types))
            self.edges.append(Edge(self.nodes, point_a, identifier, random_weight()))
            self.edges.append(Edge(self.nodes, identifier, point_b, random_weight()))

        # Find possible starting Nodes (not outputs) and ending Nodes (not inputs)
        if random.random() < probability:
            possible_starts = []
            [possible_starts.append(i) for i in self.nodes.keys()]
            [possible_starts.remove(i) for i in self.outs]
            possible_starts = list(set(possible_starts))
            possible_ends = []
            [possible_ends.append(i) for i in self.nodes.keys()]
            [possible_ends.append(i) for i in self.ins]
            possible_ends = list(set(possible_ends))
            chosen_a = random.choice(possible_starts)
            if chosen_a in possible_ends:
                possible_ends.remove(chosen_a)
            chosen_b = random.choice(possible_ends)
            self.edges.append(Edge(self.nodes, chosen_a, chosen_b, random_weight()))

        # Remove a random edge
        if random.random() < probability:
            self.edges.remove(random.choice(self.edges))

        return self

    def new_id(self):
        """
        Ensure that Node ids are unique

        :return: a unique id for each Node
        """
        self.tick += 1
        return self.tick - 1


def random_weight():
    """
    Creates a random weight

    :return: a random weight for an Edge
    """
    return 2 * random.random() - 1


class Globals:
    """
    A class for the global variables
    TODO: Implement this better?
    """
    node_types = ["bias", "negate", "sigmoid", "relu", "abs", "square", "sine", "add", "multiply", "polar", "max", "min"]
    X = []
    Y = []


class GuessFormula:
    """
    A class to guess the formula linking input and output matrices
    """
    def __init__(self, x, y):
        """
        Create a population of the simplest possible Structures, mutate them all

        :param x: input matrix
        :param y: output matrix
        """
        Globals.X = x
        Globals.Y = y
        self.population = []
        self.size = 10000

        # Create most basic Structures, mutate them all slightly
        for birth in range(self.size):
            s = Structure(len(x[0]), len(y[0]))
            s.mutate(1.0)
            self.population.append(s)

    def fit(self, accuracy):
        """
        Runs loop to mutate best guesses and discard worst guesses

        :param accuracy: required fitness score for best Node
        """
        best_fits = [0.0]
        while max(best_fits) < accuracy and len(self.population) > 0:
            self.population = sorted(self.population, key=lambda x: x.find_fitness(), reverse=True)
            fitness_list = [s.fitness for s in self.population]
            if fitness_list[0] > max(best_fits):
                best_fits.append(fitness_list[0])
                print("Fitness:", round(best_fits[-1], 5))
            avg_fit = sum(fitness_list) / len(fitness_list)

            survivors = []
            for p in self.population:
                if p.fitness >= avg_fit:
                    a = p
                    b = copy.deepcopy(a)
                    b.mutate(0.1)
                    survivors.append(a)
                    survivors.append(b)
            self.population = survivors[:self.size]

    def best_guess(self):
        """
        Return the alpha Node (best guess, highest fitness)

        :return: Node with the highest fitness score
        """
        alpha = self.population[0]
        print("Best Model Guess:")
        alpha.print_nodes()
        alpha.print_edges()

    def predict(self, input_matrix):
        """
        Predict output matrix of an input matrix

        :param input_matrix: matrix of inputs
        :return: best guess of output matrix
        """
        alpha = self.population[0]
        return alpha.predict(input_matrix)

