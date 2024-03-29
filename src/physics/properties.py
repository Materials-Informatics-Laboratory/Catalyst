
def calc_reaction_enthalpy(energies,n_systems=2):
    '''
    mole fractions
    '''
    x = []
    for i in range(n_systems - 1):
        x.append(1.0 / float(n_systems - 1))
    '''
    detla_hmix
    '''
    delta_hmix = energies[0]
    for i,energy in enumerate(energies[1:]):
        delta_hmix -= energy * x[i]

    return delta_hmix