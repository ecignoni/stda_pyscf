import sys
import numpy as np
from scipy.spatial.distance import cdist
from pyscf.lo import lowdin
from pyscf import lib
from pyscf import scf
from pyscf import dft
from pyscf.lib import logger
from pyscf import tdscf
from pyscf.tdscf.rhf import TDMixin
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


def eri_mo_monopole(mf, alpha=None, beta=None, ax=None, mode="full"):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    if ax is None:
        ax = get_hybrid_coeff(mf)
    if mode != "stda" and mode != "full":
        raise RuntimeError(f"mode is either 'stda' or 'full', given '{mode}'")
    gam_J = gamma_J(mol, ax, beta)
    gam_K = gamma_K(mol, ax, alpha)
    if mode == "full":
        q = charge_density_monopole(mol, mo_coeff)
        eri_J = lib.einsum("Apq,AB,Brs->pqrs", q, gam_J, q)
        eri_K = lib.einsum("Apq,AB,Brs->pqrs", q, gam_K, q)
    elif mode == "stda":
        occidx = np.where(mf.mo_occ == 2)[0]
        nocc = len(occidx)
        q_oo = charge_density_monopole(mol, mo_coeff[:, :nocc], mo_coeff[:, :nocc])
        q_ov = charge_density_monopole(mol, mo_coeff[:, :nocc], mo_coeff[:, nocc:])
        q_vv = charge_density_monopole(mol, mo_coeff[:, nocc:], mo_coeff[:, nocc:])
        eri_J = lib.einsum("Aij,AB,Bab->iajb", q_oo, gam_J, q_vv)
        eri_K = lib.einsum("Aia,AB,Bjb->iajb", q_ov, gam_K, q_ov)
    return eri_J, eri_K


def _select_active_space(a, eri_J, eri_K, e_max, tp, verbose=None):

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    nocc, nvir, _, _ = a.shape

    # (1) select the CSFs by energy
    # E_u = A_ia,ia = (e_a - e_i) + 2*(ia|ia) - (ii|aa) ; u = ia
    diag_a = np.diag(a.reshape(nocc * nvir, nocc * nvir)).copy()
    diag_a += np.einsum("iaia->ia", eri_K).reshape(-1) * 2
    diag_a -= np.einsum("iaia->ia", eri_J).reshape(-1)

    # Select P-CSF and N-CSF
    idx_pcsf = np.where(diag_a <= e_max)[0]
    idx_ncsf = np.where(diag_a > e_max)[0]
    try:
        pcsf = np.concatenate([[divmod(i, nvir)] for i in idx_pcsf])
    except ValueError as e:
        if idx_pcsf.size == 0:
            errmsg = 'No CSF below the energy threshold,'
            errmsg += ' you may want to increase it'
            raise ValueError(errmsg) from None
        else:
            raise e

    log.info('%d CSF included by energy' % len(idx_pcsf))
    log.info('%d considered in PT2' % len(idx_ncsf))

    # (2) select the S-CSFs by perturbation
    eri_J = eri_J[:, :, pcsf[:, 0], pcsf[:, 1]].reshape(nocc * nvir, pcsf.shape[0])
    eri_K = eri_K[:, :, pcsf[:, 0], pcsf[:, 1]].reshape(nocc * nvir, pcsf.shape[0])
    a_uv = 2 * eri_K[idx_ncsf] - eri_J[idx_ncsf]
    denom = diag_a[idx_ncsf, None] - diag_a[idx_pcsf]
    e_pt = np.divide(
        np.abs(a_uv) ** 2,
        denom,
    )
    e_u = np.sum(e_pt, axis=1)

    # S-CSF and N-CSF
    idx_scsf = idx_ncsf[e_u >= tp]
    idx_ncsf = idx_ncsf[e_u < tp]
    scsf = np.concatenate([[divmod(i, nvir)] for i in idx_scsf])
    ncsf = np.concatenate([[divmod(i, nvir)] for i in idx_ncsf])

    log.info('%d CSF included by PT' % len(idx_scsf))
    log.info('%d CSF in total' % (len(idx_scsf) + len(idx_pcsf)))

    # (3) Get perturbative contribution to sum to P-CSF
    a_uv = 2 * eri_K[idx_ncsf] - eri_J[idx_ncsf]
    denom = denom[e_u < tp]
    e_pt = np.divide(
        np.abs(a_uv) ** 2,
        denom,
    )
    # This is a stabilizing contrib.
    e_ncsf = -np.sum(e_pt, axis=0)

    return idx_pcsf, idx_scsf, idx_ncsf, pcsf, scsf, ncsf, e_ncsf


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
    verbose=None,
):

    if a is None:
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = np.where(mo_occ == 2)[0]
        viridx = np.where(mo_occ == 0)[0]
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)

    if eri_J is None or eri_K is None:
        _eri_J, _eri_K = eri_mo_monopole(mf, alpha=alpha, beta=beta, ax=ax, mode="stda")
        if eri_J is None:
            eri_J = _eri_J
        if eri_K is None:
            eri_K = _eri_K

    e_max = e_max / AU_TO_EV

    idx_pcsf, idx_scsf, idx_ncsf, pcsf, scsf, ncsf, e_ncsf = _select_active_space(
        a=a, eri_J=eri_J, eri_K=eri_K, e_max=e_max, tp=tp, verbose=verbose,
    )

    return idx_pcsf, idx_scsf, idx_ncsf, pcsf, scsf, ncsf, e_ncsf


def screen_mo(mf, mo_energy=None, ax=None, e_max=7.0, verbose=None):
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if ax is None:
        ax = get_hybrid_coeff(mf)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    log.info('%-40s = %15.8f' % ('spectral range up to (eV)', e_max))

    mo_occ = mf.mo_occ
    occidx = np.where(mo_occ == 2)[0]
    viridx = np.where(mo_occ == 0)[0]

    window = 2 * (1.0 + 0.8 * ax) * (e_max / AU_TO_EV)
    vthr = np.max(mo_energy[occidx]) + window
    othr = np.min(mo_energy[viridx]) - window

    log.info('%-40s = %15.8f' % ('occ MO cut-off (eV)', othr * AU_TO_EV))
    log.info('%-40s = %15.8f' % ('virtMO cut-off (eV)', vthr * AU_TO_EV))

    mask_occ = np.where(mo_energy[occidx] > othr)[0]
    mask_vir = np.where(mo_energy[viridx] < vthr)[0]
    nocc = len(mask_occ)
    nvir = len(mask_vir)
    mask = np.ix_(mask_occ, mask_vir, mask_occ, mask_vir)

    return mask


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
    verbose=None,
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
    if mode != "active" and mode != "full":
        raise RuntimeError(f"mode is either 'full' or 'active', given '{mode}'")

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

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

    eri_J, eri_K = eri_mo_monopole(mf, alpha=alpha, beta=beta, ax=ax, mode="stda")

    if mode == "full":
        idx_pcsf = np.arange(nocc * nvir)
        idx_scsf = None
        idx_ncsf = None
        e_ncsf = 0

    elif mode == "active":

        log.info('******** sTDA ********')

        # Screen MOs based on scaled energy gap
        screen_mask = screen_mo(mf, ax=ax, e_max=e_max, verbose=log)
        a = a[screen_mask]
        b = b[screen_mask]
        eri_J = eri_J[screen_mask]
        eri_K = eri_K[screen_mask]
        nocc, nvir, _, _ = a.shape

        log.info('%-40s = %15.8E' % ('perturbation thr', tp))
        log.info('%-40s = F' % ('triplet'))
        log.info('%s %12d' % ('MOs in TDA :', nocc+nvir))
        log.info('%s %12d' % ('oMOs in TDA:', nocc))
        log.info('%s %12d' % ('vMOs in TDA:', nvir))

        # MO selection based on energy (P-CSF) and perturbation theory (S-CSF, N-CSF)
        idx_pcsf, idx_scsf, idx_ncsf, pcsf, scsf, ncsf, e_ncsf = select_active_space(
            mf,
            a=a,
            eri_J=eri_J,
            eri_K=eri_K,
            alpha=alpha,
            beta=beta,
            ax=ax,
            e_max=e_max,
            tp=tp,
            verbose=log,
        )

        # Reconstruct A, B in the set of active CSFs
        idx_active = np.concatenate((idx_pcsf, idx_scsf))
        active_block = np.ix_(idx_active, idx_active)

        eri_J = eri_J.reshape(nocc * nvir, nocc * nvir)[active_block]
        eri_K = eri_K.reshape(nocc * nvir, nocc * nvir)[active_block]
        a = a.reshape(nocc * nvir, nocc * nvir)
        # Perturbative contrib. on P-CSF only
        a[idx_pcsf, idx_pcsf] += e_ncsf
        a = a[active_block]

        b = b.reshape(nocc * nvir, nocc * nvir)[active_block]

    a += eri_K * 2 - eri_J

    return a, b, (idx_pcsf, idx_scsf, idx_ncsf, pcsf, scsf, ncsf, e_ncsf)


def direct_diagonalization(a, nstates=3):
    if a.ndim == 4:
        nocc, nvir, _, _ = a.shape
        a = a.reshape(nocc * nvir, nocc * nvir)
    elif a.ndim == 2:
        pass
    else:
        raise RuntimeError(f"a.ndim={a.ndim} not supported")
    e, v = np.linalg.eig(a)
    # trick to get `idx` of same length of e
    e[e < 0] = np.infty
    idx = np.argsort(e)
    e = e[idx][:nstates]
    v = v[:, idx][:, :nstates]
    # e *= AU_TO_EV
    return e, v


# def _contract_multipole_active(tdobj, ints, hermi=True, xy=None):
#    if xy is None:
#        xy = tdobj.xy
#    mo_coeff = tdobj._scf.mo_coeff
#    mo_occ = tdobj._scf.mo_occ
#    orbo = mo_coeff[:, mo_occ == 2]
#    orbv = mo_coeff[:, mo_occ == 0]
#    nocc = orbo.shape[1]
#    nvir = orbv.shape[1]
#
#    nstates = len(xy)
#    pol_shape = ints.shape[:-2]
#    nao = ints.shape[-1]
#
#    ints = lib.einsum(
#        "xpq,pi,qj->xij", ints.reshape(-1, nao, nao), orbo.conj(), orbv
#    )  # [:, nocc, nvir]
#    ints = ints.reshape(ints.shape[0], nocc * nvir)  # [:, nocc*nvir]
#    ints = ints[:, tdobj.idx_pcsf]  # [:, num P-CSF]
#    pol = np.array([np.einsum("xp,p->x", ints, x) * 2 for x, y in xy])
#    if isinstance(xy[0][1], np.ndarray):
#        if hermi:
#            pol += [np.einsum("xp,p->x", ints, y) * 2 for x, y in xy]
#        else:
#            pol -= [np.einsum("xp,p->x", ints, y) * 2 for x, y in xy]
#    pol = pol.reshape((tdobj.nstates,) + pol_shape)
#    return pol

# def _contract_multipole(tdobj, ints, hermi=True, xy=None):
#    if tdobj.mode == "full":
#        # xy is in the same format of pyscf: delegate
#        return tdscf.rhf._contract_multipole(tdobj, ints, hermi=hermi, xy=xy)
#    elif tdobj.mode == "active":
#        # x is [num_PCSF], use different contraction method
#        return _contract_multipole_active(tdobj, ints, hermi=hermi, xy=xy)
#    else:
#        raise RuntimeError(f"mode is either 'full' or 'active', given {mode}")


class sTDA(TDMixin):
    """simplified Tamm-Dancoff approximation"""

    def __init__(self, mf, ax=None, alpha=None, beta=None, e_max=7.5, tp=1e-4):
        super().__init__(mf)
        self.ax = ax
        self.alpha = alpha
        self.beta = beta
        self.e_max = e_max
        self.tp = tp

        keys = set(("ax", "alpha", "beta", "e_max", "tp"))
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, x):
        self._ax = get_hybrid_coeff(self._scf) if x is None else x

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, x):
        assert self.ax is not None
        self._alpha = get_alpha_beta(self.ax)[0] if x is None else x

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, x):
        assert self.ax is not None
        self._beta = get_alpha_beta(self.ax)[1] if x is None else x

    def check_restricted(self):
        mf = self._scf
        is_rks = isinstance(mf, dft.rks.RKS)
        is_rhf = isinstance(mf, scf.hf.RHF)
        if not is_rks and not is_rhf:
            raise NotImplementedError(f"{type(mf)}. Only RKS and RHF are supported")

    def check_singlet(self):
        if self.singlet == False:
            raise NotImplementedError(f"Only singlet excitations are supported.")

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("Davidson diagonalization currently not supported by sTDA.")
        log.info(
            "'conv_tol', 'eigh lindep', 'eigh level_shift', 'eigh max_space'. and 'eigh max_cycle' are ignored."
        )
        log.info("")
        log.info("******** sTDA specific parameters ********")
        log.info("ax = %s", self.ax)
        log.info("alpha = %s", self.alpha)
        log.info("beta = %s", self.beta)
        log.info("e_max = %g (eV), %g (a.u.)", self.e_max, self.e_max / AU_TO_EV)
        log.info("tp = %g (a.u.)", self.tp)
        log.info('\n')

    def parse_active_space(self, active_space):
        self.idx_pcsf, self.idx_scsf, self.idx_ncsf = active_space[0:3]
        self.pcsf, self.scsf, self.ncsf = active_space[3:6]
        self.e_ncsf = active_space[-1]
        return self

    def kernel(self, nstates=None, mode="active"):
        """sTDA diagonalization solver"""
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_restricted()
        self.check_singlet()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        self.mode = mode

        log = logger.Logger(self.stdout, self.verbose)

        a, b, active_space = get_ab(
            self._scf,
            alpha=self.alpha,
            beta=self.beta,
            ax=self.ax,
            e_max=self.e_max,
            tp=self.tp,
            mode=mode,
            verbose=log,
        )
        self.parse_active_space(active_space)
        self.e, x1 = direct_diagonalization(a, nstates=nstates)
        self.converged = [True]

        nocc = (self._scf.mo_occ > 0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc

        if mode == "full":
            # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
            self.xy = [(xi.reshape(nocc, nvir) * np.sqrt(0.5), 0) for xi in x1.T]
        elif mode == "active":
            # Here xi is [num P-CSF], we transform back to (nocc, nvir)
            xy = [(xi * np.sqrt(0.5), 0) for xi in x1.T]
            new_xy = []
            active = np.concatenate((self.pcsf, self.scsf), axis=0)
            for x, y in xy:
                new_x = np.zeros((nocc, nvir))
                new_y = np.zeros((nocc, nvir))
                new_x[active[:, 0], active[:, 1]] = x.copy()
                new_xy.append((new_x, new_y))
            self.xy = new_xy

        if self.chkfile:
            lib.chkfile.save(self.chkfile, "tddft/e", self.e)
            lib.chkfile.save(self.chkfile, "tddft/xy", self.xy)

        log.info('\n')
        log.timer("sTDA", *cpu0)
        self._finalize()
        return self.e, self.xy
