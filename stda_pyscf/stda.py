import numpy as np
from pyscf.lo import lowdin
from pyscf.lib import logger


def lowdin_pop(mol, dm, s, verbose=logger.DEBUG):
    log = logger.new_logger(mol, verbose)
    s_orth = np.linalg.inv(lowdin(s))
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        pop = np.einsum('qi,ij,jq->q', s_orth, dm, s_orth).real
    else:
        pop = np.einsum('qi,ij,jq->q', s_orth, dm[0]+dm[1], s_orth).real

    log.info(' ** Lowdin pop **')
    for i, s in enumerate(mol.ao_labels()):
        log.info('pop of  %s %10.5f', s, pop[i])

    log.note(' ** Lowdin atomic charges **')
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
    chg = mol.atom_charges() - chg
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg
