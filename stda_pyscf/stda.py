import numpy as np
from scipy.spatial.distance import cdist
from pyscf.lo import lowdin
from pyscf import lib
from pyscf import scf
from pyscf import dft
from pyscf.lib import logger
from .parameters import chemical_hardness, get_alpha_beta


# Constants
AU_TO_EV = 27.211324570273


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


def charge_density_monopole(mol, mo_coeff, mo_coeff2=None):
    s = mol.intor_symmetric("int1e_ovlp")
    s_orth = np.linalg.inv(lowdin(s))
    c_orth = np.dot(s_orth, mo_coeff)
    c_orth2 = np.dot(s_orth, mo_coeff2) if mo_coeff2 is not None else c_orth
    nmo = c_orth.shape[1]
    nmo2 = c_orth2.shape[1]
    q = np.zeros((mol.natm, nmo, nmo2))
    for i, (atidx, *_) in enumerate(mol.ao_labels(fmt=None)):
        q[atidx] += np.einsum("p,q->pq", c_orth[i], c_orth2[i]).real
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
    denom = ((R * ax * eta) ** beta + 1) ** (1.0 / beta)
    gamma = ax * eta / denom
    # gamma = (1.0 / (R**beta + (ax * eta) ** (-beta))) ** (1.0 / beta)
    return gamma


def gamma_K(mol, ax, alpha=None):
    R = distance_matrix(mol)
    eta = hardness_matrix(mol)
    if alpha is None:
        alpha, _ = get_alpha_beta(ax)
    denom = ((R * eta) ** alpha + 1) ** (1.0 / alpha)
    gamma = eta / denom
    # gamma = (1.0 / (R**alpha + eta ** (-alpha))) ** (1.0 / alpha)
    return gamma


def get_hybrid_coeff(mf):
    mol = mf.mol
    if isinstance(mf, dft.rks.RKS):
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
    gam_J = gamma_J(mol, ax, beta)
    gam_K = gamma_K(mol, ax, alpha)
    q = charge_density_monopole(mol, mo_coeff)
    eri_J = lib.einsum("Apq,AB,Brs->pqrs", q, gam_J, q)
    eri_K = lib.einsum("Apq,AB,Brs->pqrs", q, gam_K, q)
    return eri_J, eri_K


def _select_active_space(a, eri_J, eri_K, e_max, tp):
    nocc, nvir, _, _ = a.shape
    # (1) select the CSFs by energy
    # E_u = A_ia,ia = (e_a - e_i) + 2*(ia|ia) - (ii|aa) ; u = ia
    diag_a = np.diag(a.reshape(nocc * nvir, nocc * nvir)).copy()
    diag_a += np.einsum("iaia->ia", eri_K).reshape(-1) * 2
    diag_a -= np.einsum("iaia->ia", eri_J).reshape(-1)
    # Select P-CSF and N-CSF
    idx_pcsf = np.where(diag_a <= e_max)[0]
    idx_ncsf = np.where(diag_a > e_max)[0]
    pcsf = np.concatenate([[divmod(i, nvir)] for i in idx_pcsf])
    ncsf = np.concatenate([[divmod(i, nvir)] for i in idx_ncsf])
    # (2) select the S-CSFs by perturbation
    eri_J = eri_J[:, :, pcsf[:, 0], pcsf[:, 1]].reshape(nocc * nvir, pcsf.shape[0])
    eri_K = eri_K[:, :, pcsf[:, 0], pcsf[:, 1]].reshape(nocc * nvir, pcsf.shape[0])
    a_uv = 2 * eri_K - eri_J
    denom = diag_a[:, None] - diag_a[idx_pcsf]
    # P-CSF have e_perturb = np.infty
    e_perturb = np.sum(
        np.divide(
            np.abs(a_uv) ** 2,
            denom,
            out=np.full(a_uv.shape, np.infty),
            where=denom != 0,
        ),
        axis=1,
    )
    # P-CSF + S-CSF
    idx_pcsf = np.where(e_perturb >= tp)[0]
    idx_ncsf = np.where(e_perturb < tp)[0]
    pcsf = np.concatenate([[divmod(i, nvir)] for i in idx_pcsf])
    ncsf = np.concatenate([[divmod(i, nvir)] for i in idx_ncsf])
    # Get perturbative contribution to sum to P-CSF
    e_ncsf = e_perturb[idx_ncsf].sum()
    return idx_pcsf, pcsf, e_ncsf


def select_active_space(
    mf,
    a=None,
    eri_J=None,
    eri_K=None,
    e_max=7.0,
    tp=1e-4,
    alpha=None,
    beta=None,
    ax=None,
):
    if a is None:
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    if eri_J is None or eri_K is None:
        _eri_J, _eri_K = eri_mo_monopole(mf, alpha=alpha, beta=beta, ax=ax)
        _eri_J = np.einsum("ijab->iajb", _eri_J[:nocc, :nocc, nocc:, nocc:])
        _eri_K = np.einsum("iajb->iajb", _eri_K[:nocc, nocc:, :nocc, nocc:])
        if eri_J is None:
            eri_J = _eri_J
        if eri_K is None:
            eri_K = _eri_K
    e_max = e_max / AU_TO_EV
    idx_pcsf, pcsf, e_ncsf = _select_active_space(
        a=a, eri_J=eri_J, eri_K=eri_K, e_max=e_max, tp=tp
    )
    return idx_pcsf, pcsf, e_ncsf


def get_ab(
    mf,
    mo_energy=None,
    mo_coeff=None,
    mo_occ=None,
    alpha=None,
    beta=None,
    ax=None,
    e_max=7.0,
    tp=1e-4,
    mode="active",
):
    r"""A and B matrices for sTDA response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + 2(ia|jb)' - (ij|ab)'
    B[i,a,j,b] = 0
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    assert mo_coeff.dtype == np.double
    if mode != 'active' and mode != 'full':
        raise RuntimeError(f"mode is either 'full' or 'active', given '{mode}'")

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
    eri_J = np.einsum("ijab->iajb", eri_J[:nocc, :nocc, nocc:, nocc:])
    eri_K = np.einsum("iajb->iajb", eri_K[:nocc, nocc:, :nocc, nocc:])

    if mode == "full":
        pass
    elif mode == "active":
        idx_pcsf, _, _ = select_active_space(
            mf, a=a, eri_J=eri_J, eri_K=eri_K, alpha=alpha, beta=beta, ax=ax
        )
        pcsf_block = np.ix_(idx_pcsf, idx_pcsf)
        eri_J = eri_J.reshape(nocc * nvir, nocc * nvir)[pcsf_block]
        eri_K = eri_K.reshape(nocc * nvir, nocc * nvir)[pcsf_block]
        a = a.reshape(nocc * nvir, nocc * nvir)[pcsf_block]
        b = b.reshape(nocc * nvir, nocc * nvir)[pcsf_block]

    a += eri_K * 2 - eri_J

    return a, b


def direct_diagonalization(a, nstates=3):
    if a.ndim == 4:
        nocc, nvir, _, _ = a.shape
        a = a.reshape(nocc * nvir, nocc * nvir)
    elif a.ndim == 2:
        pass
    else:
        raise RuntimeError(f'a.ndim={a.ndim} not supported')
    e, v = np.linalg.eig(a)
    e = np.sort(e[e > 0])[:nstates]
    e *= AU_TO_EV
    return e
