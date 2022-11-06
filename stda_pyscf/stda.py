import numpy as np
from scipy.spatial.distance import cdist
from pyscf.lo import lowdin
from pyscf.lib import logger
from .parameters import chemical_hardness, get_alpha_beta


def lowdin_pop(mol, dm, s, verbose=logger.DEBUG):
    log = logger.new_logger(mol, verbose)
    s_orth = np.linalg.inv(lowdin(s))
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        pop = np.einsum("qi,ij,jq->q", s_orth, dm, s_orth).real
    else:
        pop = np.einsum("qi,ij,jq->q", s_orth, dm[0] + dm[1], s_orth).real

    log.info(" ** Lowdin pop **")
    for i, s in enumerate(mol.ao_labels()):
        log.info("pop of  %s %10.5f", s, pop[i])

    log.note(" ** Lowdin atomic charges **")
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
    at_chg = mol.atom_charges() - chg
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note("charge of  %d%s =   %10.5f", ia, symb, at_chg[ia])
    return pop, at_chg, chg


def charge_density_monopoles(mol, mo_coeff, verbose=logger.DEBUG):
    s = mol.intor_symmetric("int1e_ovlp")
    s_orth = np.linalg.inv(lowdin(s))
    c_orth = np.dot(s_orth, mo_coeff)
    nmo = mo_coeff.shape[1]
    q = np.zeros((mol.natm, nmo, nmo))
    for i, (atidx, *_) in enumerate(mol.ao_labels(fmt=None)):
        q[atidx] += np.einsum("p,q->pq", c_orth[i], c_orth[i]).real
    return q


def distance_matrix(mol):
    coords = mol.atom_coords()
    R = cdist(coords, coords, metric="euclidean")
    return R


def hardness_matrix(mol):
    hrd = chemical_hardness
    sym = mol.atom_pure_symbol
    eta = np.array([hrd[sym(a)] for a in range(mol.natm)])
    eta = (eta[:, np.newaxis] + eta[np.newaxis, :]) / 2
    return eta


def gamma_J(mol, ax):
    R = distance_matrix(mol)
    eta = hardness_matrix(mol)
    _, beta = get_alpha_beta(ax)
    gamma = (1.0 / (R**beta + (ax * eta) ** (-beta))) ** (1.0 / beta)
    return gamma


def gamma_K(mol, ax):
    R = distance_matrix(mol)
    eta = hardness_matrix(mol)
    alpha, _ = get_alpha_beta(ax)
    gamma = (1.0 / (R**alpha + eta ** (-alpha))) ** (1.0 / alpha)
    return gamma
