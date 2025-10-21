#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:27:34 2025

@author: stevemolloy
"""
"""
Defines the grammar for the toy language.
This class is self-contained, so the entire language can be modified
by changing only this file.
"""
import random

class ToyLanguage:
    """A container for the terminals, non-terminals, and rules of a context-free grammar."""
    def __init__(self):
        self.terminals = {
            'the', 'a', 'big', 'small', 'happy', 'sad', 'cat', 'dog', 'boy', 'girl', 'mat',
            'saw', 'chased', 'liked', 'on', 'with', 'under'
        }

        self.non_terminals = {'S', 'NP', 'VP', 'PP', 'Det', 'N', 'V', 'P', 'Adj'}

        self.rules = {
            'S': ['NP VP'],
            'VP': ['V NP'],
            'NP': ['Det N', 'Det Adj N'],
            'Det': ['the', 'a'],
            'Adj': ['big', 'small', 'happy', 'sad'],
            'N': ['cat', 'dog', 'boy', 'girl', 'mat'],
            'V': ['saw', 'chased', 'liked'],
            'P': ['on', 'with', 'under']
        }
        
        # Helper mapping from non-terminal to its terminal words for easy access
        self.terminal_map = self._create_terminal_map()

    def _create_terminal_map(self):
        """Creates a dictionary mapping non-terminals to their corresponding terminal words."""
        terminal_map = {}
        for non_terminal, productions in self.rules.items():
            # Check if the first production rule's first element is a terminal word
            if productions and productions[0] in self.terminals:
                 terminal_map[non_terminal] = productions
        return terminal_map

    def generate_sentence(self, symbol='S'):
        """
        Generates a random, valid sentence from the grammar by recursively
        expanding the production rules.
        """
        # If the symbol is a non-terminal that maps directly to terminals (e.g., 'N', 'V'),
        # pick a random terminal word.
        if symbol in self.terminal_map:
            return random.choice(self.terminal_map[symbol])

        # If the symbol is not in the rules, it's a terminal itself.
        if symbol not in self.rules:
            if symbol in self.terminals:
                return symbol
            raise ValueError(f"Symbol '{symbol}' not found in rules or terminals.")

        # Choose a random production rule for the current symbol.
        production = random.choice(self.rules[symbol])
        
        # Recursively call generate_sentence for each part of the chosen production.
        parts = [self.generate_sentence(s) for s in production.split()]
        return " ".join(parts)

    def __repr__(self):
        return f"ToyLanguage(Terminals: {len(self.terminals)}, Non-Terminals: {len(self.non_terminals)})"


