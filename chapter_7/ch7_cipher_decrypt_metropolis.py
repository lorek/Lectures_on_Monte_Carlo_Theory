##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##
## Code for the book:                                               ##
## "Lectures on Monte Carlo Theory"                                 ##
## Authors: Paweł Lorek and Tomasz Rolski                           ##
## Affiliation: University of Wrocław                               ##
##                                                                  ##
## Publisher: Springer, 2025                                        ##
## Springer link: link.springer.com/book/9783032011893              ##
## Series: Probability Theory and  Stochastic Modelling             ##
## ISBN:  978-3-032-01189-3                                         ##
##                                                                  ##
## File: ch7_cipher_decrypt_metropolis.py                           ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Decrypt a substitution cipher using a Metropolis-style algorithm.
The program builds a language model from an input text file (e.g., "ch7_cipher_Tolstoy_War_and_Peace.txt")
to create a transition matrix and character mapping, then reads the encrypted message,
and uses a Metropolis algorithm to search for a decryption key that maximizes the likelihood
of the decrypted text. Progress is displayed at iterations 1–9 and every 1000 iterations.
The final decrypted message is saved to the specified output file.

Common functions (read_file, preprocess_text, create_mappings) are imported
from ch7_cipher_encrypt.py to avoid duplication.
"""

import argparse
import numpy as np
import string
import matplotlib.pyplot as plt
from collections import Counter
import os

# Import common functions from ch7_cipher_encrypt.py
from ch7_cipher_encrypt import read_file, preprocess_text, create_mappings, display_snippet

EPSILON = 1e-10

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Decrypt a substitution cipher via a Metropolis-style algorithm"
    )
    parser.add_argument('--input', default="ch7_cipher_Tolstoy_War_and_Peace.txt",
                        help="File to learn the transition matrix M (default: %(default)s)")
    parser.add_argument('--message_enc', default="ch7_cipher_message_encrypted.txt",
                        help="File with the encrypted message (default: %(default)s)")
    parser.add_argument('--output', default="ch7_cipher_message_decrypted_metropolis.txt",
                        help="Output file to save the final decrypted message (default: %(default)s)")
    parser.add_argument('--R', default="3000",
                        help="Total number of iterations for decryption (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the RNG (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving figures (default: %(default)s)")
    return parser.parse_args()

def build_transition_matrix(content):
    """
    Build a transition matrix from the input content.
    Only allowed characters (lower-case letters and space) are considered.
    Returns a dictionary mapping each two-character string to its normalized frequency.
    """
    print("Building transition matrix from input text...")
    allowed = " " + string.ascii_lowercase
    # Filter content to include only allowed characters
    letters = [ch for ch in content if ch in allowed]
    # Use the full allowed set for normalization
    letters_unique = set(allowed)
    cntr = Counter(zip(letters, letters[1:]))
    M2 = {''.join(pair): count for pair, count in cntr.items()}
    # Normalize: For each starting character a in allowed, sum over allowed following letters b.
    for a in letters_unique:
        total_a = sum(M2.get(a + b, 0) for b in allowed)
        if total_a > 0:
            for b in allowed:
                key = a + b
                if key in M2:
                    M2[key] = M2[key] / total_a
    print("Transition matrix computed.")
    return M2, letters_unique

def decrypt_with_candidate(encrypted_message, candidate, char_mapping, char_mapping_inv):
    """
    Decrypt the message using the candidate permutation.
    The decryption applies the inverse of the candidate key.
    """
    inv_candidate = np.arange(len(candidate))[np.argsort(candidate)]
    result = ""
    for ch in encrypted_message:
        if ch in char_mapping:
            idx = char_mapping[ch]
            result += char_mapping_inv[inv_candidate[idx]]
        else:
            result += ch
    return result

def compute_log_likelihood(decrypted, M2):
    """
    Compute the log likelihood of the decrypted text based on transition probabilities.
    """
    logL = 0.0
    for a, b in zip(decrypted, decrypted[1:]):
        key = a + b
        p = M2.get(key, EPSILON)
        logL += np.log(p)
    return logL

def propose(perm, rng):
    """
    Propose a new permutation by swapping two randomly chosen indices.
    """
    new_perm = perm.copy()
    i, j = rng.choice(len(perm), size=2, replace=False)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm


def metropolis_decrypt(encrypted_message, M2, char_mapping, char_mapping_inv, R, rng):
    """
    Run a Metropolis algorithm to search for a permutation that maximizes
    the likelihood of the decrypted text.
    Displays the first 9 steps and then every 1000 iterations.
    Returns a tuple (candidate, current_decryption, log_probs) where log_probs is a list
    of the log likelihood at each iteration.
    """
    n = len(char_mapping)
    candidate = np.arange(n)  # Start with the identity permutation.
    current_decryption = decrypt_with_candidate(encrypted_message, candidate, char_mapping, char_mapping_inv)
    current_score = compute_log_likelihood(current_decryption, M2)

    log_probs = []  # List to store log likelihood at each iteration.

    # Display progress for steps 1 to 9
    for i in range(1, min(10, R + 1)):
        if i == 1:
            print(f"Step {i:4d}: {current_decryption[:70]}  LogProb: {current_score:.2f}")
        else:
            candidate = propose(candidate, rng)
            current_decryption = decrypt_with_candidate(encrypted_message, candidate, char_mapping, char_mapping_inv)
            current_score = compute_log_likelihood(current_decryption, M2)
            print(f"Step {i:4d}: {current_decryption[:70]}  LogProb: {current_score:.2f}")
        log_probs.append(current_score)

    # Continue iterations and display every 1000 iterations
    for i in range(10, R + 1):
        proposed = propose(candidate, rng)
        proposed_decryption = decrypt_with_candidate(encrypted_message, proposed, char_mapping, char_mapping_inv)
        proposed_score = compute_log_likelihood(proposed_decryption, M2)
        if proposed_score >= current_score or rng.random() < np.exp(proposed_score - current_score):
            candidate = proposed
            current_score = proposed_score
            current_decryption = proposed_decryption
        log_probs.append(current_score)
        if i % 1000 == 0 or i == R:
            print(f"Step {i:4d}: {current_decryption[:70]}  LogProb: {current_score:.2f}")
    return candidate, current_decryption, log_probs

def main():
    args = ParseArguments()

    # Set up results directory.
    os.makedirs(args.results_path, exist_ok=True)

    print("Arguments parsed. Starting decryption process...\n")



    # Initialize RNG using PCG64 with seed
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))
    print(f"RNG initialized with PCG64 and seed: {seed}\n")
    tmp = rng.random(2)

    # Read and preprocess the language text for the transition matrix
    content = preprocess_text(read_file(args.input), lower_on="yes")
    # Read and preprocess the encrypted message (forcing lower-case)
    encrypted_message = preprocess_text(read_file(args.message_enc), lower_on="yes")

    display_snippet("Encrypted message (processed)", encrypted_message)


    # Build transition matrix and create character mappings (for lower-case and space)
    M2, _ = build_transition_matrix(content)
    char_mapping, char_mapping_inv = create_mappings()
    print(f"Character mapping created with {len(char_mapping)} unique characters.\n")

    # Run the Metropolis decryption algorithm
    R = int(args.R)
    print("Starting Metropolis decryption iterations...\n")
    candidate, final_decryption, log_probs = metropolis_decrypt(encrypted_message, M2, char_mapping, char_mapping_inv, R, rng)

    # Save the final decrypted message
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("Final decrypted message:\n" + final_decryption + "\n")

    display_snippet("Final decrypted message", final_decryption)

    print(f"\nFinal decrypted message written to {args.output}")
    print("Decryption process completed successfully.")

    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(log_probs) + 1), log_probs)
    plt.xlabel("Step")
    plt.ylabel("Log Likelihood")

    plt.grid(True)


    fig_name = os.path.join(args.results_path, f"ch7_cipher_decription_logprobs.pdf")
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print(f"Logprobs figure saved in:  {fig_name}")
    plt.title("Log Likelihood Progress")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
