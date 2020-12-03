from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt
from scipy import io

from graph_mbo import graph_mbo
from graph_mbo.utils import get_modularity, spectral_clustering


def run_congress_data():
    start_time = default_timer()
    adjs = io.loadmat("data/US_matricies/US_fraction.mat")
    party_lists = io.loadmat("data/US_matricies/US_partylists.mat")

    mbo_modularities = []
    spectral_modularities = []
    party_modularities = []
    comm_sizes_mbo = []
    comm_sizes_spectral = []

    # Index with the number of the congress in question
    congress_range = range(57, 116)
    # congress_range = [115]
    # congress_range = [81, 85, 86]
    # congress_range = range(110, 116)
    if False:
        for cnum in congress_range:
            print("Congress", str(cnum))
            adj = adjs[str(cnum)]
            party_list = party_lists[str(cnum)][0]
            party_codes = np.unique(party_list)

            mbo_modularity_max = 0
            mbo_n_comm_max = 0
            spectral_modularity_max = 0
            spectral_n_comm_max = 0
            # for num_communities in range(2, len(party_codes)+1):
            for num_communities in range(2, 5):
                # for num_communities in [2]:
                print("Number of communities:", num_communities)
                target_size = [
                    adj.shape[0] // num_communities // 2,
                    adj.shape[0] // num_communities // 2,
                ]
                for i in range(2, num_communities):
                    target_size.append(3)  # minimum independent size 3
                u = graph_mbo(
                    adj,
                    pseudospectral=False,
                    symmetric=False,
                    signless=False,
                    normalized=False,
                    modularity=True,
                    num_communities=num_communities,
                    dt=0.1,
                    min_dt=1e-8,
                    target_size=target_size,
                    initial_state_type="spectral",  # fidelity_avg",
                    thresh_type="max",  # auction",
                    n_inner=1000,
                    fidelity_type="spectral",
                    fidelity_coeff=100 * adj.shape[0],
                )

                for i in range(num_communities):
                    party_dict = {party_code: 0 for party_code in party_codes}
                    for row in range(u.shape[0]):
                        if u[row, i] == 1:
                            party_dict[party_list[row]] += 1
                    print(party_dict)
                mbo_modularity = get_modularity(adj, u)
                print("MBO modularity:", mbo_modularity)
                if mbo_modularity > mbo_modularity_max:
                    mbo_n_comm_max = num_communities
                mbo_modularity_max = max(mbo_modularity, mbo_modularity_max)

                u_spectral = spectral_clustering(adj, num_communities)
                spectral_modularity = get_modularity(adj, u_spectral)
                print("Spectral modularity:", spectral_modularity)
                if spectral_modularity > spectral_modularity_max:
                    spectral_n_comm_max = num_communities
                spectral_modularity_max = max(
                    spectral_modularity, spectral_modularity_max
                )
            mbo_modularities.append(mbo_modularity_max)
            spectral_modularities.append(spectral_modularity_max)

            u_party = np.zeros((adj.shape[0], len(party_codes)))
            for i in range(len(party_codes)):
                for row in range(adj.shape[0]):
                    if party_list[row] == party_codes[i]:
                        u_party[row, i] = 1
            party_modularity = get_modularity(adj, u_party)
            print("Party modularity:", party_modularity)
            party_modularities.append(party_modularity)
            print(
                "Best community numbers: MBO",
                mbo_n_comm_max,
                "spectral",
                spectral_n_comm_max,
            )
            comm_sizes_mbo.append(mbo_n_comm_max)
            comm_sizes_spectral.append(spectral_n_comm_max)

        print("Time taken:", default_timer() - start_time)
        np.save("mbo_modularities.npy", mbo_modularities)
        np.save("spectral_modularities.npy", spectral_modularities)
        np.save("party_modularities.npy", party_modularities)
        np.save("mbo_comm_sizes.npy", comm_sizes_mbo)
        np.save("spectral_comm_sizes.npy", comm_sizes_spectral)

    mbo_modularities = np.load("mbo_modularities.npy")
    spectral_modularities = np.load("spectral_modularities.npy")
    party_modularities = np.load("party_modularities.npy")
    comm_sizes_mbo = np.load("mbo_comm_sizes.npy")
    comm_sizes_spectral = np.load("spectral_comm_sizes.npy")

    plt.figure()
    plt.plot(congress_range, mbo_modularities, label="MBO")
    plt.plot(congress_range, spectral_modularities, label="Spectral")
    plt.plot(congress_range, party_modularities, label="Party")
    plt.legend()
    plt.xlabel("Congress Number")
    plt.ylabel("Modularity")
    plt.title("Congress Modularity Over Time")
    plt.show()

    opt_comm_sizes = []
    for i, cnum in enumerate(congress_range):
        if mbo_modularities[i] > spectral_modularities[i]:
            opt_comm_sizes.append(comm_sizes_mbo[i])
        else:
            opt_comm_sizes.append(comm_sizes_spectral[i])

    plt.figure()
    # plt.plot(congress_range, comm_sizes_mbo, label="MBO")
    # plt.plot(congress_range, comm_sizes_spectral, label="Spectral")
    plt.plot(congress_range, opt_comm_sizes)
    plt.legend()
    plt.xlabel("Congress Number")
    plt.ylabel("Optimal Number of Communities")
    plt.title("Congress Number of Communities Over Time")
    plt.show()


if __name__ == "__main__":
    run_congress_data()
