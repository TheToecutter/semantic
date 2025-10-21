#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:29:57 2025

@author: stevemolloy
"""
"""
This file contains the logic for generating trajectories through the semantic space.
This is where the dynamical system is implemented.
"""
import numpy as np
import random
from fiber_bundle import FiberBundle

def generate_sentence_trajectory_hardcoded(gist: tuple, bundle: FiberBundle):
    """
    The original hardcoded trajectory generation, kept for comparison.
    It generates a sentence word-by-word with a fixed structure and random choices.
    """
    agent, action, patient = gist
    
    np1_structure = [('Det', 'det1'), ('Adj', 'adj1'), ('N', 'n1')]
    vp_structure = [('V', 'v')]
    np2_structure = [('Det', 'det2'), ('Adj', 'adj2'), ('N', 'n2')]
    full_structure = np1_structure + vp_structure + np2_structure

    sentence_words = []
    trajectory_vectors = []
    trajectory_words = []
    
    print("\n--- Generating Trajectory in Total Space (Hardcoded) ---")
    
    filler_dim = len(next(iter(bundle.filler_space.values())))
    total_space_dim = len(bundle.total_roles) * filler_dim
    current_vector = np.zeros(total_space_dim)
    
    trajectory_vectors.append(current_vector.copy())
    trajectory_words.append("START")
    print(f"Step 0 (START): vector shape {current_vector.shape}")

    for word_type, role_name in full_structure:
        word_to_add = None
        if role_name == 'n1': word_to_add = agent
        elif role_name == 'v': word_to_add = action
        elif role_name == 'n2': word_to_add = patient
        else:
            if word_type == 'Adj' and random.random() < 0.5:
                continue
            
            possible_words = bundle.language.terminal_map[word_type]
            available_words = [w for w in possible_words if w in bundle.filler_space]
            if not available_words: continue
            word_to_add = random.choice(available_words)

        sentence_words.append(word_to_add)
        role_vec = bundle.total_roles[role_name]
        filler_vec = bundle.filler_space[word_to_add]
        word_role_tensor = np.kron(role_vec, filler_vec)
        current_vector = current_vector + word_role_tensor
        
        trajectory_vectors.append(current_vector.copy())
        trajectory_words.append(word_to_add)
        
        current_sentence_str = " ".join(sentence_words)
        print(f"Step {len(trajectory_words)-1} (+ '{word_to_add}'): '{current_sentence_str}'")

    final_sentence = " ".join(sentence_words)
    return final_sentence, np.array(trajectory_vectors), trajectory_words, list(range(len(trajectory_vectors)))

def generate_sentence_trajectory(gist: tuple, bundle: FiberBundle, total_steps=1500, learning_rate=0.1, rotation_strength=0.8, settle_threshold_factor=0.6):
    """
    Generates a sentence using a principled dynamical system with rotational forces.
    A dissipative force pulls the state towards the final sentence, while a
    rotational force drives the trajectory through the sequence of syntactic roles.
    Settling is detected sequentially by waiting for each role to cross an energy threshold.

    Args:
        gist: A tuple of (agent, action, patient).
        bundle: The initialized FiberBundle object.
        total_steps: Total simulation steps for the trajectory.
        learning_rate: Controls the overall speed of convergence.
        rotation_strength: Balances dissipative vs. rotational force.
        settle_threshold_factor: The energy percentage a role must reach to be "settled".

    Returns:
        A tuple containing:
            final_sentence (str): The sentence constructed from the settled words.
            trajectory_vectors (np.array): A (num_steps x 70) array of the full trajectory.
            words_added (list): A list of the words that were settled on, in order.
            settling_indices (list): A list of indices into `trajectory_vectors`.
                This is CRITICAL for plotting. It specifies exactly which points
                in the trajectory correspond to the words in `words_added`.
                For example, `trajectory_vectors[settling_indices[i]]` is the
                state vector at the moment `words_added[i]` was settled.
    """
    agent, action, patient = gist
    
    # --- 1. Determine the final target sentence (the global attractor) ---
    np1_structure = [('Det', 'det1'), ('Adj', 'adj1'), ('N', 'n1')]
    vp_structure = [('V', 'v')]
    np2_structure = [('Det', 'det2'), ('Adj', 'adj2'), ('N', 'n2')]
    full_structure = np1_structure + vp_structure + np2_structure
    
    target_word_role_pairs = []
    sentence_words_for_final_string = []
    for word_type, role_name in full_structure:
        word_to_add = None
        if role_name == 'n1': word_to_add = agent
        elif role_name == 'v': word_to_add = action
        elif role_name == 'n2': word_to_add = patient
        else:
            if word_type == 'Adj' and random.random() < 0.5: continue
            possible_words = bundle.language.terminal_map[word_type]
            available_words = [w for w in possible_words if w in bundle.filler_space]
            if not available_words: continue
            word_to_add = random.choice(available_words)
        
        target_word_role_pairs.append({'word': word_to_add, 'role': role_name})
        sentence_words_for_final_string.append(word_to_add)
    
    target_sentence = " ".join(sentence_words_for_final_string)
    print(f"\nDynamical System Target: '{target_sentence}'")
    final_vector = bundle.represent_sentence_in_total_space(target_sentence)

    # --- 2. Set up the rotational force field ---
    num_roles = len(bundle.total_roles)
    filler_dim = len(next(iter(bundle.filler_space.values())))
    
    A_roles = np.diag(np.ones(num_roles - 1), 1) - np.diag(np.ones(num_roles - 1), -1)
    A_roles[num_roles - 1, 0] = 1
    A_roles[0, num_roles - 1] = -1
    rotation_matrix = np.kron(A_roles, np.identity(filler_dim))

    # --- 3. Simulate the trajectory ---
    print("\n--- Simulating Trajectory in Total Space (Rotational Dynamics) ---")
    trajectory_vectors = [np.zeros_like(final_vector)]
    current_vector = np.zeros_like(final_vector)
    
    words_added = []
    settling_indices = [0]
    # Create a list of roles to be settled, in the correct grammatical order.
    roles_to_settle = [item for item in target_word_role_pairs]
    role_name_to_idx = {name: i for i, name in enumerate(bundle.total_roles.keys())}

    final_step_count = total_steps
    for step in range(total_steps):
        dissipative_force = final_vector - current_vector
        rotational_force = rotation_matrix @ dissipative_force
        total_force = (1 - rotation_strength) * dissipative_force + rotation_strength * rotational_force
        current_vector += learning_rate * total_force
        trajectory_vectors.append(current_vector.copy())

        # --- DETAILED LOGGING FOR CURRENT STEP ---
        base_vector = bundle.project_to_base_space(current_vector)
        print(f"--- Step {step} ---")
        print(f"  State (Norms) -> Total Space: {np.linalg.norm(current_vector):.3f}, Base Space: {np.linalg.norm(base_vector):.3f}")
        print(f"  Forces (Norms) -> Dissipative: {np.linalg.norm(dissipative_force):.3f}, Rotational: {np.linalg.norm(rotational_force):.3f}, Total: {np.linalg.norm(total_force):.3f}")


        # Check if the next role in the sequence has settled
        if not roles_to_settle:
            final_step_count = step
            break

        next_role_info = roles_to_settle[0]
        role_name = next_role_info['role']
        role_idx = role_name_to_idx[role_name]
        
        current_role_slice = current_vector[role_idx*filler_dim : (role_idx+1)*filler_dim]
        target_role_slice = final_vector[role_idx*filler_dim : (role_idx+1)*filler_dim]
        
        current_energy = np.linalg.norm(current_role_slice)
        target_energy = np.linalg.norm(target_role_slice)

        if target_energy > 1e-6 and current_energy >= target_energy * settle_threshold_factor:
            settled_word = next_role_info['word']
            print(f"!!! Settled on role '{role_name}' (word: '{settled_word}') at step {step} !!!")
            words_added.append(settled_word)
            # The trajectory list includes the start point at index 0, so the current step's
            # vector is at index 'step + 1'.
            settling_indices.append(step + 1)
            roles_to_settle.pop(0) # Move on to the next role in the sequence

    # --- 4. Construct final sentence from the actual trajectory ---
    final_sentence_from_trajectory = " ".join(words_added)
    
    print(f"\nSimulation finished after {final_step_count} steps.")
    
    return final_sentence_from_trajectory, np.array(trajectory_vectors), words_added, settling_indices


