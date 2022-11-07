import numpy as np
from scipy.spatial.distance import cdist
from pyscf.lo import lowdin
from pyscf import lib
from pyscf import scf
from .parameters import chemical_hardness, get_alpha_beta


# def lowdin_pop(mol, dm, s, verbose=logger.DEBUG):
#    log = logger.new_logger(mol, verbose)
#    s_orth = np.linalg.inv(lowdin(s))
#    if isinstance(dm, np.ndarray) and dm.ndim == 2:
#        pop = np.einsum("qi,ij,jq->q", s_orth, dm, s_orth).real
#    else:
#        pop = np.einsum("qi,ij,jq->q", s_orth, dm[0] + dm[1], s_orth).real
#
#    log.info(" ** Lowdin pop **")
#    for i, s in enumerate(mol.ao_labels()):
#        log.info("pop of  %s %10.5f", s, pop[i])
#
#    log.note(" ** Lowdin atomic charges **")
#    chg = np.zeros(mol.natm)
#    for i, s in enumerate(mol.ao_labels(fmt=None)):
#        chg[s[0]] += pop[i]
#    at_chg = mol.atom_charges() - chg
#    for ia in range(mol.natm):
#        symb = mol.atom_symbol(ia)
#        log.note("charge of  %d%s =   %10.5f", ia, symb, at_chg[ia])
#    return pop, at_chg, chg


def charge_density_monopole(mol, mo_coeff):
    s = mol.intor_symmetric("int1e_ovlp")
    s_orth = np.linalg.inv(lowdin(s))
    c_orth = np.dot(s_orth, mo_coeff)
    nmo = mo_coeff.shape[1]
    q = np.zeros((mol.natm, nmo, nmo))
    for i, (atidx, *_) in enumerate(mol.ao_labels(fmt=None)):
        q[atidx] += np.einsum("p,q->pq", c_orth[i], c_orth[i]).real
    return q


def distance_matrix(mol):
    coords = mol.atom_coords(unit="Bohr")
    R = cdist(coords, coords, metric="euclidean")
    return R


def hardness_matrix(mol):
    hrd = chemical_hardness
    sym = mol.atom_pure_symbol
    eta = np.array([hrd[sym(a)] for a in range(mol.natm)])
    eta = (eta[:, np.newaxis] + eta[np.newaxis, :]) / 2
    return eta


def gamma_J(mol, ax, beta=None):
    R = distance_matrix(mol)
    eta = hardness_matrix(mol)
    if beta is None:
        _, beta = get_alpha_beta(ax)
    denom = ((R * ax * eta)**beta + 1) ** (1./beta)
    gamma = ax * eta / denom
    # gamma = (1.0 / (R**beta + (ax * eta) ** (-beta))) ** (1.0 / beta)
    return gamma


def gamma_K(mol, ax, alpha=None):
    R = distance_matrix(mol)
    eta = hardness_matrix(mol)
    if alpha is None:
        alpha, _ = get_alpha_beta(ax)
    denom = ((R * eta)**alpha + 1) ** (1./alpha)
    gamma = eta / denom
    # gamma = (1.0 / (R**alpha + eta ** (-alpha))) ** (1.0 / alpha)
    return gamma


def get_hybrid_coeff(mf):
    mol = mf.mol
    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        *_, ax = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    elif isinstance(mf, scf.hf.RHF):
        ax = 1.0
    else:
        raise NotImplementedError(f"{type(mf)}")
    return ax


def eri_mo_monopole(mf, alpha=None, beta=None, ax=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    if ax is None:
        ax = get_hybrid_coeff(mf)
    print(ax)
    gam_J = gamma_J(mol, ax, beta)
    gam_K = gamma_K(mol, ax, alpha)
    q = charge_density_monopole(mol, mo_coeff)
    eri_J = lib.einsum("Apq,AB,Brs->pqrs", q, gam_J, q)
    eri_K = lib.einsum("Apq,AB,Brs->pqrs", q, gam_K, q)
    return eri_J, eri_K


def get_ab(
    mf, mo_energy=None, mo_coeff=None, mo_occ=None, alpha=None, beta=None, ax=None
):
    r"""A and B matrices for sTDA response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + 2(ia|jb)' - (ij|ab)'
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    assert mo_coeff.dtype == np.double

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = np.where(mo_occ == 2)[0]
    viridx = np.where(mo_occ == 0)[0]
    orbv = mo_coeff[:, viridx]
    orbo = mo_coeff[:, occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = np.hstack((orbo, orbv))
    nmo = nocc + nvir

    e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
    a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    b = np.zeros_like(a)

    eri_J, eri_K = eri_mo_monopole(mf, alpha, beta, ax)
    a += np.einsum("iajb->iajb", eri_K[:nocc, nocc:, :nocc, nocc:]) * 2
    a -= np.einsum("ijab->iajb", eri_J[:nocc, :nocc, nocc:, nocc:])

    return a, b
