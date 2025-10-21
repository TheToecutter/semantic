#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:29:21 2025

@author: stevemolloy
"""
"""
Defines the FiberBundle class, which sets up the vector spaces
(role spaces) based on a given language and filler space.
"""
import numpy as np
from language_definition import ToyLanguage

class FiberBundle:
    """
    Represents the fiber bundle structure with its associated vector spaces.
    
    This class creates the role spaces and holds the shared filler space.
    It also contains methods to represent sentences as vectors and to project
    them from the total space to the base space.
    """
    def __init__(self, language: ToyLanguage, filler_space: dict):
        if not isinstance(language, ToyLanguage):
            raise TypeError("language must be an instance of ToyLanguage")
        if not filler_space:
            raise ValueError("filler_space cannot be empty.")

        self.language = language
        self.filler_space = filler_space
        
        # --- Define Role Names ---
        self.base_role_names = ['agent', 'action', 'patient']
        self.total_role_names = ['det1', 'adj1', 'n1', 'v', 'det2', 'adj2', 'n2']
        
        # --- Create Role Spaces (Basis Vectors) ---
        self.base_roles = self._create_role_vectors(self.base_role_names)
        self.total_roles = self._create_role_vectors(self.total_role_names)
        
        # --- Build Projection Map ---
        self._build_projection_map()
        
        print("Fiber bundle structure initialized.")

    def _create_role_vectors(self, role_names: list) -> dict:
        """Creates a set of one-hot encoded basis vectors for a list of role names."""
        num_roles = len(role_names)
        identity_matrix = np.identity(num_roles)
        return {name: vector for name, vector in zip(role_names, identity_matrix)}
        
    def _build_projection_map(self):
        """Creates the mapping from total roles to base roles for the projection."""
        self.projection_map = {
            'n1': 'agent',
            'v': 'action',
            'n2': 'patient',
            # Fiber roles (details) map to None and are dropped by the projection.
            'det1': None, 'adj1': None,
            'det2': None, 'adj2': None
        }

    def represent_sentence_in_total_space(self, sentence: str):
        """
        Parses a sentence and represents it as a vector in the Total Space (E).
        The vector is a sum of tensor products of role and filler vectors.
        """
        words = sentence.split()
        verb = None
        verb_idx = -1
        for i, word in enumerate(words):
            if word in self.language.terminal_map['V']:
                verb = word
                verb_idx = i
                break
        if verb is None:
            raise ValueError("Sentence does not contain a valid verb.")
            
        np1_words = words[:verb_idx]
        np2_words = words[verb_idx+1:]
        
        def parse_np(np_words):
            det, adj, noun = None, None, None
            for word in np_words:
                if word in self.language.terminal_map['Det']: det = word
                elif word in self.language.terminal_map['Adj']: adj = word
                elif word in self.language.terminal_map['N']: noun = word
            return det, adj, noun

        det1, adj1, n1 = parse_np(np1_words)
        det2, adj2, n2 = parse_np(np2_words)
        
        if not (n1 and verb and n2):
            raise ValueError("Sentence must contain an agent (N), action (V), and patient (N).")

        role_filler_map = {
            'det1': det1, 'adj1': adj1, 'n1': n1,
            'v': verb,
            'det2': det2, 'adj2': adj2, 'n2': n2
        }
        
        total_role_dim = len(self.total_roles)
        filler_dim = len(next(iter(self.filler_space.values())))
        tensor_sum = np.zeros(total_role_dim * filler_dim)
        
        for role_name, word in role_filler_map.items():
            if word and word in self.filler_space:
                role_vec = self.total_roles[role_name]
                filler_vec = self.filler_space[word]
                tensor_product = np.kron(role_vec, filler_vec)
                tensor_sum += tensor_product
                
        return tensor_sum

    def project_to_base_space(self, total_space_vector):
        """
        Projects a vector from the Total Space (E) to the Base Space (B).
        This is done by deconstructing and reconstructing the tensor sum,
        keeping only the core semantic roles.
        """
        total_role_dim = len(self.total_roles)
        base_role_dim = len(self.base_roles)
        filler_dim = len(next(iter(self.filler_space.values())))
        
        base_space_vector = np.zeros(base_role_dim * filler_dim)

        for i, role_name in enumerate(self.total_role_names):
            start_idx = i * filler_dim
            end_idx = start_idx + filler_dim
            filler_vec_slice = total_space_vector[start_idx:end_idx]
            
            if np.any(filler_vec_slice):
                base_role_name = self.projection_map.get(role_name)
                if base_role_name:
                    base_role_vec = self.base_roles[base_role_name]
                    tensor_product = np.kron(base_role_vec, filler_vec_slice)
                    base_space_vector += tensor_product
                    
        return base_space_vector

    def __repr__(self):
        filler_dim = len(next(iter(self.filler_space.values())))
        return (
            f"FiberBundle(\n"
            f"  Base Space Roles: {len(self.base_roles)} (dim={len(self.base_roles)}),\n"
            f"  Total Space Roles: {len(self.total_roles)} (dim={len(self.total_roles)}),\n"
            f"  Filler Space Words: {len(self.filler_space)} (dim={filler_dim})\n"
            f")"
        )
    