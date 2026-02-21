import pickle
import gzip
import torch

def write_labelled_extxyz(filename,labels,atoms,cutoff):
    of = open(filename, 'w')
    of.write(str(len(labels)) + '\n')
    of.write('Lattice="')
    for cd in atoms.get_cell():
        for c in cd:
            of.write(str(c) + ' ')
    of.write(
        ' Properties=species:S:1:pos:R:3:y:R:1 cutoff ' + str(cutoff) + ' pbc="T T T"\n')
    for j, atom in enumerate(atoms):
        of.write(atom.symbol + ' ')
        for pos in atom.position:
            of.write(str(pos) + ' ')
        of.write(str(labels[j].item()) + '\n')
    of.close()


def save_dictionary(fname,data):
    with gzip.open(fname, "wb") as fp:
        pickle.dump(data, fp,protocol=pickle.HIGHEST_PROTOCOL)

def load_dictionary(fname):
    # Define a custom unpickler
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    with gzip.open(fname, "rb") as handle:
        dictionary = CPU_Unpickler(handle).load()
    return dictionary





