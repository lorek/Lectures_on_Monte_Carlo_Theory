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
## File: ch7_cipher_encrypt.py                                      ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Encrypt a plain text message using a substitution cipher.
This program reads a plain message and expects to read a sigma (encryption key)
from the sigma file. If the sigma file is missing or invalid, it displays an error
message ("Sigma file missing or invalid. You may create one with generate_sigma.py")
and exits.
The encryption key is expected to be saved as a single line of comma-separated numbers.
The character mapping is based on a predetermined set of allowed characters (lower-case a-z and space).
"""

import argparse
import numpy as np
import string
import re
import sys
import json

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Encrypt a message using a substitution cipher"
    )
    parser.add_argument('--message', default="ch7_cipher_message_Anna_Karenina_paragraph.txt",
                        help="File with the plain message (default: %(default)s)")
    parser.add_argument('--message_enc', default="ch7_cipher_message_encrypted.txt",
                        help="File to save the encrypted message (default: %(default)s)")
    parser.add_argument('--sigma_file', default="ch7_cipher_sigma.yaml",
                        help="File (yaml) to read the encryption key (default: %(default)s)")
    return parser.parse_args()

def read_file(filename):
    print(f"Reading file: {filename}")
    with open(filename, encoding="utf-8") as f:
        return f.read()

def preprocess_text(text, lower_on="yes"):
    # Force lower-case conversion
    if lower_on == "yes":
        text = text.lower()
    # Remove extra whitespace and keep only lower-case letters and space
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z ]', '', text).strip()
    return text

def create_mappings():
    """
    Create a fixed mapping based on a predetermined set of allowed characters.
    The allowed characters include a space and all lower-case ASCII letters.
    """
    allowed_chars =  string.ascii_lowercase + " " # a-z then underscore
    characters = sorted(set(allowed_chars))
    char_mapping = {ch: i for i, ch in enumerate(characters)}
    char_mapping_inv = {i: ch for i, ch in enumerate(characters)}
    return char_mapping, char_mapping_inv


def read_sigma_yaml(sigma_yaml_file, n):
    """
    Read the sigma (encryption key) from a YAML file.
    The file is expected to contain lines in the format:
      "0": 19  # 'a' -> 't'
    Returns a NumPy array of length n if successful, or None otherwise.
    """
    sigma_dict = {}
    try:
        with open(sigma_yaml_file, encoding="utf-8") as f:
            for line in f:
                # Remove inline comments and extra whitespace
                line = line.split('#')[0].strip()
                if not line:
                    continue
                # Expecting format: "0": 19
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                key_str = parts[0].strip().strip('"')
                value_str = parts[1].strip()
                key = int(key_str)
                value = int(value_str)
                sigma_dict[key] = value
        sigma = np.array([sigma_dict[i] for i in range(n)])
        if sigma.size == n:
            print(f"Sigma successfully read from {sigma_yaml_file}")
            return sigma
        else:
            print(f"Sigma size mismatch (found {sigma.size}, expected {n}).")
    except Exception as e:
        print(f"Error reading sigma from {sigma_yaml_file}: {e}")
    return None


def encrypt_message(plain_message, sigma, char_mapping, char_mapping_inv):
    """
    Encrypt the plain message by applying the encryption key sigma.
    """
    encrypted_message = ""
    for ch in plain_message:
        if ch in char_mapping:
            idx = char_mapping[ch]
            encrypted_message += char_mapping_inv[sigma[idx]]
        else:
            encrypted_message += ch
    return encrypted_message

# Define a helper function to display snippets (first 500 characters, with [...] if longer)
def display_snippet(label, text):
    snippet = text[:500]
    if len(text) > 500:
        snippet += "[...]"
    print(f"\n=== {label} START ===")
    print(snippet)
    print(f"=== {label} END ===\n")
    print()

def main():
    args = ParseArguments()
    print("Arguments parsed. Starting encryption process...\n")

    original_message = read_file(args.message)
    display_snippet("Original message", original_message)


    # Read and preprocess the plain message (forcing lower-case)
    plain_message = preprocess_text(read_file(args.message), lower_on="yes")

    message_processed_filename = args.message.replace(".txt", "_processed.txt")
    with open(message_processed_filename, "w", encoding="utf-8") as f:
        f.write(plain_message)
    display_snippet("Processed message", plain_message)
    print(f"Processed plain message saved to {message_processed_filename}")

    # Display the snippets for the original and processed messages



    # Create a fixed character mapping (using only lower-case letters and space)
    char_mapping, char_mapping_inv = create_mappings()
    print(f"Fixed character mapping created with {len(char_mapping)} characters.\n")

    # Read sigma from file; if missing or invalid, exit with an error message.
    n = len(char_mapping)
    sigma = read_sigma_yaml(args.sigma_file, n)
    if sigma is None:
        print("Sigma file missing or invalid. You may create one with generate_sigma.py")
        sys.exit(1)

    # Encrypt the message
    encrypted_message = encrypt_message(plain_message, sigma, char_mapping, char_mapping_inv)

    # Save the encrypted message
    with open(args.message_enc, "w", encoding="utf-8") as f:
        f.write(encrypted_message)
    # Display a snippet of the encrypted message
    display_snippet("Encrypted message", encrypted_message)
    print(f"Encrypted message saved to {args.message_enc}\n")
    print("Encryption process completed successfully.")

if __name__ == "__main__":
    main()
