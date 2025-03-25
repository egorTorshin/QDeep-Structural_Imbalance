import click
import matplotlib
# Trap errors with importing pyplot (for testing frameworks) and
# specify "agg" backend
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

import dimod
from drawing import draw_social_network
from mmp_network import global_signed_social_network

from qdeepsdk import QDeepHybridSolver
import numpy as np

@click.command()
@click.option('--qpu', 'sampler_type', flag_value='qpu',
              help='Use the QPU')
@click.option('--cpu', 'sampler_type', flag_value='cpu',
              help='Use simulated annealing')
@click.option('--hybrid', 'sampler_type', flag_value='hybrid',
              help="Use Leap's hybrid sampler")
@click.option('--region', default='global',
              type=click.Choice(['global', 'iraq', 'syria'],
                                case_sensitive=False))
@click.option('--show', is_flag=True,
              help="show the plot rather than saving it")
@click.option('--inspect', is_flag=True,
              help=("inspect the problem, "
                    "does nothing when not using the QPU with --qpu"))
def main(sampler_type, region, show, inspect):

    if sampler_type is None:
        print("No solver selected, defaulting to hybrid")
        sampler_type = 'hybrid'

    # get the appropriate signed social network
    G = global_signed_social_network(region=region)

    # choose solver and any tuning parameters needed
    if sampler_type == 'qpu' or sampler_type == 'cpu':
        params = dict(num_reads=100)
        # Initialize QDeepHybridSolver (use this instead of SimulatedAnnealingSampler)
        sampler = QDeepHybridSolver()
        sampler.token = "your-auth-token-here"  # Replace with your actual token

    # Build the BQM for the problem
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Add the edges and the weights into the BQM (this will depend on the structure of the global_signed_social_network function)
    for (u, v, data) in G.edges(data=True):
        weight = data.get('weight', 1)  # Default weight is 1 if not specified
        bqm.add_interaction(u, v, weight)

    # Convert BQM to QUBO format
    qubo, offset = bqm.to_qubo()

    # Convert the QUBO to a NumPy matrix for solving
    n = len(G.nodes)
    matrix = np.zeros((n, n))
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    for (i, j), coeff in qubo.items():
        matrix[mapping[i], mapping[j]] = coeff

    # Solve the QUBO using QDeepHybridSolver
    result = sampler.solve(matrix)
    best_sample = result['sample']

    # Convert the best_sample back to the node labels and determine the edges and colors
    colors = {}
    for idx, value in best_sample.items():
        node = list(mapping.keys())[list(mapping.values()).index(idx)]
        colors[node] = 'red' if value == 1 else 'blue'  # Example color scheme

    edges = [(u, v) for u, v, data in G.edges(data=True) if colors[u] != colors[v]]

    print("Found", len(edges), 'violations out of', len(G.edges), 'edges')

    draw_social_network(G, colors)

    if show:
        plt.show()
    else:
        filename = 'structural_imbalance_{}.png'.format(region)
        plt.savefig(filename, facecolor='white')
        plt.clf()

if __name__ == '__main__':
    main()
