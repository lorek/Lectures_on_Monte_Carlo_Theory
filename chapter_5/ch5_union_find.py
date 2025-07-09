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
## File: ch5_union_find.py                                          ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Union-Find Algorithm

In the example, we consider a graph with 10 vertices and the following edges:
    (1,2), (2,7), (3,4), (3,8), (4,5), (4,9), (5,10), (9,10).
The expected connected components are:
    A1 = {1, 2, 7}, A2 = {6}, A3 = {3, 4, 5, 8, 9, 10}.
"""


import argparse




def ParseArguments():
    parser = argparse.ArgumentParser(description="Union-Find Algorithm Example")
    parser.add_argument('--verbose', default="True", help='Number of strata for CDF (default: %(default)s)')

    args = parser.parse_args()
    return args

class UnionFind:
    def __init__(self, n):
        # Initialize parent array: each vertex is its own parent.
        # Here we use 1-indexed vertices.
        self.parent = list(range(n + 1))

    def find(self, i):
        # Recursive Find with path compression.
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Union: set the parent of the representative of j to be the representative of i.
        root_i = self.find(i)
        root_j = self.find(j)
        # Print current state before union.
        print(f"Union({i}, {j}): Representative of {i} = {root_i}, Representative of {j} = {root_j}")
        self.parent[root_j] = root_i
        print(f"After Union({i}, {j}): Parent array = {self.parent}")


def connected_components(uf, n):
    # Build a dictionary: key = root, value = list of vertices in that component.
    comps = {}
    for i in range(1, n + 1):
        root = uf.find(i)
        if root in comps:
            comps[root].append(i)
        else:
            comps[root] = [i]
    return comps


def main():
    args = ParseArguments()
    verbose = args.verbose.lower() in ['true', '1', 'yes']

    n_vertices = 10
    # Define the list of edges.
    edges = [(1, 2), (2, 7), (3, 4), (3, 8), (4, 5), (4, 9), (5, 10), (9, 10)]

    # Initialize UnionFind structure.
    uf = UnionFind(n_vertices)
    if verbose:
        print("Initial parent array (each vertex is its own representative):")
        print(uf.parent)
        print("\n--- Performing Unions ---")

    # Perform unions according to the edges.
    for (u, v) in edges:
        uf.union(u, v)

    print("\nFinal parent array:")
    print(uf.parent)

    # Determine connected components.
    comps = connected_components(uf, n_vertices)
    print("\nConnected Components:")
    for rep, comp in comps.items():
        print(f"Component with representative {rep}: {comp}")


if __name__ == "__main__":
    main()
