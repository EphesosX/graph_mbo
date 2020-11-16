import numpy as np


def get_initial_state(
    num_nodes, num_communities, target_size, type="random", fidelity_type=None
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
        return get_fidelity_term(u, type=fidelity_type)
    return u


def get_fidelity_term(u, type="karate"):
    fidelity_term = np.zeros(u.shape)
    if type == "karate":
        fidelity_term[0, 0] = 1
        fidelity_term[-1, -1] = 1
    return fidelity_term
