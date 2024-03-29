import numpy as np

def check_num_elements(main_group, other_groups):
    main_elements = []
    unique_elements = []
    total_others = 0

    for el in main_group:
        main_elements.append(el[0])
    for system in other_groups:
        for el in system:
            check = 0
            for e in unique_elements:
                if e == el[0]:
                    check = 1
                    break
            if check == 0:
                unique_elements.append(el[0])

    if len(unique_elements) == len(main_elements):
        return 1
    else:
        return 0

def get_structure_stoichiometry(atoms):
    stoichiometry = []
    symbols = atoms.get_chemical_symbols()

    n_atoms = len(symbols)
    unique_symbols = np.unique(symbols)
    for symbol in unique_symbols:
        stoichiometry.append([symbol,0.0])
        for element in symbols:
            if element == symbol:
                stoichiometry[-1][1] += 1.0
        stoichiometry[-1][1] = stoichiometry[-1][1] / float(n_atoms)
    return stoichiometry

def check_stoichiometry(main_group, other_groups,delta=0.2):
    real_ratios = []
    for mg1 in main_group:
        for mg2 in main_group:
            real_ratios.append([mg1[0]])
            real_ratios[-1].append(mg2[0])
            real_ratios[-1].append(abs(float(mg1[1]) - float(mg2[1])))

    delta = .15
    for i in range(len(other_groups)):
        for j in range(len(other_groups[i])):
            for k in range(len(other_groups[i])):
                ratios = [other_groups[i][j][0],other_groups[i][k][0],abs(float(other_groups[i][j][1]) - float(other_groups[i][k][1]))]
                for rr in real_ratios:
                    if rr[0] == ratios[0] and rr[1] == ratios[1]:
                        if rr[2] - delta < ratios[2] < rr[2] + delta:
                            pass
                        else:
                            return 0
    return 1
