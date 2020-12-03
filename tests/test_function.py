import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graph_mbo import graph_mbo


def test_karate():
    G = nx.karate_club_graph()
    adj = nx.convert_matrix.to_numpy_matrix(G)
    u = graph_mbo(
        adj, pseudospectral=True, symmetric=True, signless=False, normalized=True
    )

    colors = ["#FF0000", "#0000FF"]
    plt.figure()
    # loc = nx.kamada_kawai_layout(G)
    loc = nx.spring_layout(G)
    nx.draw(G, node_size=1000, edge_color="black", pos=loc)
    for i in range(2):
        nodes = np.argwhere(u[:, i])
        nx.draw_networkx_nodes(
            G, node_size=1000, nodelist=nodes.flatten(), node_color=colors[i], pos=loc
        )
    plt.show()


if __name__ == "__main__":
    test_karate()
