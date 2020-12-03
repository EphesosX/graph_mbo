import networkx as nx
import numpy as np
import scipy as sp
from networkx.algorithms.community import modularity
from scipy.sparse.linalg import eigsh


def get_initial_state(
    num_nodes,
    num_communities,
    target_size,
    type="random",
    fidelity_type=None,
    fidelity_V=None,
):
    u = np.zeros((num_nodes, num_communities))
    if type == "random":
        for i in range(num_communities - 1):
            count = 0
            while count < target_size[i]:
                rand_index = np.random.randint(0, num_nodes - 1)
                if u[rand_index, i] == 0:
                    u[rand_index, i] = 1
                    count += 1
        u[np.sum(u, axis=1) < 1, -1] = 1
    elif type == "fidelity":
        return get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
    elif type == "fidelity_avg":
        u[:] = 1.0 / num_communities
        fidelity = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
        u[np.sum(fidelity, axis=1) > 0] = fidelity[np.sum(fidelity, axis=1) > 0]
    elif type == "spectral":
        u[:] = fidelity_V
        apply_threshold(u, None, "max")
    return u


def get_fidelity_term(u, type="karate", V=None):
    fidelity_term = np.zeros(u.shape)
    if type == "karate":
        fidelity_term[0, 0] = 1 - u[0, 0]
        fidelity_term[0, 1] = -u[0, 1]
        fidelity_term[-1, -1] = 1 - u[-1, -1]
        fidelity_term[-1, 0] = -u[-1, 0]
    elif type == "spectral":
        # Use the top component of each eigenvector to seed the Clusters
        if V is None:
            raise Exception()
        idxs = np.argmax(V, axis=0)
        # fidelity_term[idxs, :] = -1.0 / (u.shape[1]-1)
        fidelity_term[idxs, range(u.shape[1])] = 1  # - u[idxs, range(u.shape[1])]
    return fidelity_term


def apply_threshold(u, target_size, thresh_type):
    if thresh_type == "max":
        """Threshold to the max value across communities. Ignores target_size"""
        max_idx = np.argmax(u, axis=1)
        u[:, :] = np.zeros_like(u)
        u[(range(u.shape[0]), max_idx)] = 1
    elif thresh_type == "auction":
        """Auction between classes until target sizes are reached"""
        prices = np.zeros((1, u.shape[1]))  # Current price of community
        assignments = np.zeros_like(u)  # 1 where assigned, 0 elsewhere
        bids = np.zeros((u.shape[0],))  # Bid of each node
        epsilon = 0.01
        while np.sum(assignments) != np.sum(
            target_size
        ):  # Check if all targets are satisfied
            unassigned = np.argwhere(np.sum(assignments, axis=1) < 1)[:, 0]
            for x in unassigned:
                profit = u[x, :] - prices
                # ics = np.argmax(u[x, :]-prices)
                ics = np.flatnonzero(profit == profit.max())
                i = np.random.choice(ics)
                profit = np.delete(profit, i)
                i_next = np.random.choice(np.flatnonzero(profit == profit.max()))
                if i_next >= i:
                    i_next += 1
                price_diff = u[x, i] - prices[0, i]
                price_diff_next = u[x, i_next] - prices[0, i_next]
                bids[x] = prices[0, i] + epsilon + price_diff - price_diff_next
                if np.sum(assignments[:, i]) == target_size[i]:
                    assigned = np.argwhere(assignments[:, i] > 0)[:, 0]
                    y = np.argmin(bids[assigned])
                    y = assigned[y]
                    assignments[y, i] = 0
                    assignments[x, i] = 1
                    prices[0, i] = np.min(bids[assignments[:, i] > 0])
                else:
                    assignments[x, i] = 1
                    if np.sum(assignments[:, i]) == target_size[i]:
                        prices[0, i] = np.min(bids[assignments[:, i] > 0])
        # If there are any remaining, do max assignment
        max_idx = np.argmax(u, axis=1)
        unassigned = np.argwhere(np.sum(assignments, axis=1) == 0).flatten()
        assignments[(unassigned, max_idx[unassigned])] = 1
        u[:, :] = assignments


def get_modularity(adj, u):
    """Calculate the modularity score for the given community structure"""
    nxgraph = nx.convert_matrix.from_numpy_matrix(adj, create_using=nx.DiGraph())
    communities = [np.argwhere(u[:, i]).flatten() for i in range(u.shape[1])]
    return modularity(nxgraph, communities)


def spectral_clustering(adj, num_communities):
    graph_laplacian, degree = sp.sparse.csgraph.laplacian(adj, return_diag=True)
    D, V = eigsh(
        graph_laplacian,
        k=num_communities + 1,
        v0=np.ones((graph_laplacian.shape[0], 1)),
        which="SA",
    )
    V = V[:, 1:]
    apply_threshold(V, None, "max")
    return V
