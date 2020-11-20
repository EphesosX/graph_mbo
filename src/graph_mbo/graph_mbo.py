import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve

from .utils import get_fidelity_term, get_initial_state


def graph_mbo(
    adj_matrix,
    normalized=True,
    symmetric=True,
    signless=True,
    pseudospectral=True,
    k=100,
    num_communities=2,
    target_size=None,
    thresh_type="max",
    dt=1.8,
    min_dt=1e-4,
    max_iter=10000,
    n_inner=1000,
    fidelity_coeff=10,
    fidelity_type="karate",
):
    """
    Run the MBO scheme on a graph.
    Parameters
    ----------
    adj_matrix : np.array
        The adjacency matrix of the graph.
    normalized : bool
        Use the normalized graph Laplacian.
    signless : bool
        Use the signless graph Laplacian to find eigenvalues if normalized
    pseudospectral : bool
        Use the pseudospectral solver. If false, use CG or LU.
    k : int
        Number of eigenvalues to use for pseudospectral
    num_communities : int
        Number of communities
    target_size : list
        List of desired community sizes when using auction MBO
    thresh_type : str
        Type of thresholding to use. "max" takes the max across communities,
        "auction" does auction MBO
    dt : float
        Time step between thresholds for the MBO scheme
    min_dt : float
        Minimum time step for MBO convergence
    max_iter : int
        Maximum number of iterations
    n_inner : int
        Number of iterations for the MBO diffusion loop
    fidelity_coeff : int
        Coefficient for the fidelity term
    """

    degree = np.array(np.sum(adj_matrix, axis=-1)).flatten()
    num_nodes = len(degree)

    k = min(num_nodes, k)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    graph_laplacian, degree = sp.sparse.csgraph.laplacian(adj_matrix, return_diag=True)
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    if symmetric:
        degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
        graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)
    elif normalized:
        degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
        graph_laplacian = degree_inv @ graph_laplacian

    if pseudospectral:
        if signless:
            if normalized:
                pass
            else:
                graph_laplacian = 2 * degree_diag - graph_laplacian
        if normalized:
            D, V = sp.sparse.linalg.eigs(
                graph_laplacian,
                k=k,
                v0=np.ones((graph_laplacian.shape[0], 1)),
                which="LR" if signless else "SR",
            )
        else:
            D, V = sp.sparse.linalg.eigsh(
                graph_laplacian,
                k=k,
                v0=np.ones((graph_laplacian.shape[0], 1)),
                which="LA" if signless else "SA",
            )
        if signless:
            D = 2 * np.ones((k,)) - D  # Change D to be eigenvalues of graph Laplacian
        if normalized:
            # rescale eigenvectors to normalized space and orthogonalize
            for i in range(len(D)):
                V[:, i] /= np.sqrt(V[:, i].transpose() @ degree_diag @ V[:, i])

    last_dt = 0

    """ Initialize state """
    u = get_initial_state(num_nodes, num_communities, target_size, type="random")

    last_last_index = u == 1
    last_index = u == 1

    def apply_threshold(u, target_size, thresh_type):
        if thresh_type == "max":
            """Threshold to the max value across communities. Ignores target_size"""
            max_idx = np.argmax(u, axis=1)
            u[:, :] = np.zeros_like(u)
            u[(range(num_nodes), max_idx)] = 1
        elif thresh_type == "auction":
            pass

    if fidelity_type == "spectral":
        fidelity_D, fidelity_V = sp.sparse.linalg.eigsh(
            graph_laplacian,
            k=num_communities,
            v0=np.ones((graph_laplacian.shape[0], 1)),
            which="SA",
        )
        # apply_threshold(fidelity_V, target_size, "max")
        # return fidelity_V
    else:
        fidelity_V = None

    for n in range(max_iter):
        dti = dt / n_inner
        if pseudospectral:
            if normalized:
                a = V.transpose() @ degree_diag @ u  # Project into Hilbert space
            else:
                a = V.transpose() @ u
            d = np.zeros((k, num_communities))
            denom = sp.sparse.spdiags([1 / (1 + dti * D)], [0], k, k)
        else:
            if last_dt != dt:
                lu, piv = lu_factor(sp.sparse.eye(num_nodes) + dti * graph_laplacian)

        for j in range(n_inner):
            """ Solve system (apply CG or pseudospectral) """
            if pseudospectral:
                a = denom @ (a + fidelity_coeff * dti * d)
                u = V @ a  # Project back into normal space
                fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                # Project fidelity term into Hilbert space
                if normalized:
                    d = V.transpose() @ (degree_diag @ fidelity_term)
                else:
                    d = V.transpose() @ fidelity_term
            else:
                fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                u += fidelity_coeff * dti * fidelity_term
                for i in range(num_communities):
                    u[:, i] = lu_solve((lu, piv), u[:, i])

        """ Apply thresholding """
        apply_threshold(u, target_size, thresh_type)

        """ Stopping criterion """
        # Check that the index is changing and stop if time step becomes too small
        index = u == 1
        last_dt = dt

        norm_deviation = sp.linalg.norm(last_index ^ index) / sp.linalg.norm(index)
        if norm_deviation < 1e-4 or i % 100 == 0:
            if dt < min_dt:
                break
            else:
                dt *= 0.4
        elif np.sum(last_last_index ^ index) == 0:
            # Going back and forth
            dt *= 0.4
        last_last_index = last_index
        last_index = index

    if dt >= min_dt:
        print("MBO failed to converge")
    return u
