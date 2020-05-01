import numpy as np
import scipy as sp


def graph_mbo(adj_matrix, normalized=True, signless=True, pseudospectral=True):
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
    """
    k = 100  # Number of eigenvalues to use for pseudospectral

    dt = 0.4
    min_dt = 1e-4
    niter = 10000
    n_inner = 10

    num_communities = 5
    target_size = [1, 1, 1, 1, 1]  # TODO: fix this
    thresh_type = "flat"

    degree = np.array(np.sum(adj_matrix, axis=-1)).flatten()
    num_nodes = len(degree)

    graph_laplacian, degree = sp.sparse.csgraph.laplacian(adj_matrix, return_diag=True)
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    if signless:
        graph_laplacian = 2 * degree_diag - graph_laplacian
    if normalized:
        degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
        graph_laplacian = degree_inv @ graph_laplacian

    if pseudospectral:
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
    fidelity_coeff = 0

    """ Initialize state """
    u = np.zeros((num_nodes, num_communities))
    last_last_index = u == 1
    last_index = u == 1

    def get_fidelity_term(u):
        fidelity_term = np.zeros(u.shape)
        return fidelity_term

    def apply_threshold(u, target_size, thresh_type):
        # TODO: reimplement
        pass

    for n in range(niter):
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
                lu, piv = sp.linalg.lu_factor(
                    sp.sparse.eye(num_nodes) + dti * graph_laplacian
                )

        for j in range(n_inner):
            """ Solve system (apply CG or pseudospectral) """
            if pseudospectral:
                a = denom @ (a + fidelity_coeff * dti * d)
                u = V @ a  # Project back into normal space
                fidelity_term = get_fidelity_term(u)
                # Project fidelity term into Hilbert space
                if normalized:
                    d = V.transpose() @ (degree_diag @ fidelity_term)
                else:
                    d = V.transpose() @ fidelity_term
            else:
                fidelity_term = get_fidelity_term(u)
                u += fidelity_coeff * dti * fidelity_term
                for i in range(num_communities):
                    u[:, i] = sp.linalg.lu_solve((lu, piv), u[:, i])

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
