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
## File: ch7_cipher_generate_sigma.py                               ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Generate a random sigma (encryption key) for the substitution cipher.
The sigma is generated from a fixed character mapping (space and lower-case ASCII letters)
and saved to the specified sigma file as a single line of comma-separated numbers.
"""

import json
import argparse
import numpy as np
import string
from ch7_cipher_encrypt import create_mappings

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Generate a random sigma for the substitution cipher"
    )
    parser.add_argument('--seed', default="31415",
                        help="Seed for the RNG (default: %(default)s)")
    parser.add_argument('--sigma_file', default="ch7_cipher_sigma.yaml",
                        help="File (yaml) to save the sigma (default: %(default)s)")
    return parser.parse_args()

def save_sigma_yaml(sigma, sigma_file, allowed_chars):
    """
    Save sigma as a YAML string where each mapping is printed with an inline comment.
    Each line will look like:
      "0": 19  # 'a' -> 't'
    """
    lines = []
    for i in range(len(sigma)):
        orig = allowed_chars[i]
        new = allowed_chars[int(sigma[i])]
        line = f'"{i}": {int(sigma[i])} # \'{orig}\' -> \'{new}\''
        lines.append(line)
    yaml_output = "\n".join(lines)
    with open(sigma_file, "w", encoding="utf-8") as f:
        f.write(yaml_output)
    print(yaml_output)
    print(f"Encryption key saved to {sigma_file} in YAML format with inline comments.")



def main():
    args = ParseArguments()
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(seed)

    # Create a fixed mapping (space and lower-case ASCII letters)
    allowed_chars = string.ascii_lowercase + " "
    char_mapping, _ = create_mappings()
    n = len(char_mapping)

    # Generate sigma as a random permutation of [0, n-1]
    sigma = rng.permutation(n)

    print("Permutation sigma (encryption key):")

    # saves and displays sigma
    save_sigma_yaml(sigma, args.sigma_file, allowed_chars)

if __name__ == "__main__":
    main()
