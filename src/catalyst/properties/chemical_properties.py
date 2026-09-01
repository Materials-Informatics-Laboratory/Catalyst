import numpy as np


def _element_names(group):
    if group is None:
        raise ValueError("Element group cannot be None.")
    names = []
    for entry in group:
        if not isinstance(entry, (list, tuple, np.ndarray)) or len(entry) < 1:
            raise ValueError(
                "Each composition entry must contain at least an element label."
            )
        names.append(entry[0])
    return names


def check_num_elements(main_group, other_groups):
    """Return 1 only when the compared systems contain the same element set."""
    main_elements = set(_element_names(main_group))
    if not main_elements:
        return 0

    unique_elements = set()
    for system in other_groups or []:
        unique_elements.update(_element_names(system))

    return int(unique_elements == main_elements)


def get_structure_stoichiometry(atoms):
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    if n_atoms == 0:
        raise ValueError("Cannot calculate stoichiometry for an empty structure.")

    unique_symbols, counts = np.unique(symbols, return_counts=True)
    return [
        [symbol, float(count) / float(n_atoms)]
        for symbol, count in zip(unique_symbols, counts)
    ]


def check_stoichiometry(main_group, other_groups, delta=0.2):
    """Check pairwise composition-difference agreement within ``delta``."""
    delta = float(delta)
    if delta < 0:
        raise ValueError("delta must be non-negative.")

    main_group = list(main_group)
    if not main_group:
        raise ValueError("main_group must contain at least one composition entry.")

    real_ratios = {}
    for mg1 in main_group:
        for mg2 in main_group:
            real_ratios[(mg1[0], mg2[0])] = abs(float(mg1[1]) - float(mg2[1]))

    for system in other_groups or []:
        for item1 in system:
            for item2 in system:
                key = (item1[0], item2[0])
                if key not in real_ratios:
                    return 0
                observed = abs(float(item1[1]) - float(item2[1]))
                if abs(observed - real_ratios[key]) >= delta:
                    return 0
    return 1


def calc_reaction_enthalpy(energies, n_systems=2):
    """Return reaction/mixing enthalpy for one product and equal-weight references."""
    energies = list(energies)
    n_systems = int(n_systems)
    if n_systems < 2:
        raise ValueError("n_systems must be at least 2.")
    if len(energies) != n_systems:
        raise ValueError(
            f"Expected {n_systems} energies, received {len(energies)}."
        )

    reference_weight = 1.0 / float(n_systems - 1)
    return float(energies[0]) - sum(
        float(energy) * reference_weight for energy in energies[1:]
    )
