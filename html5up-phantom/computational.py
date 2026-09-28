"""Computational chemistry/physics module for LabLogbook's Computational category.

Two techniques live here:
  - Monte Carlo: 2D Ising model, Monte Carlo integration, random-walk diffusion.
    All real simulations run live in NumPy — nothing here is precomputed or faked.
  - DFT (small molecule): real single-point DFT via PySCF, on a geometry built from
    a SMILES string (RDKit, MMFF-optimized) or pasted XYZ coordinates.

Honesty notes (mirrors the same policy already applied to SEM/EDS/EBSD elsewhere
in this app): DFT here is single-point energy only, on small molecules with small
basis sets — this is genuine electronic structure theory, not a toy, but it is not
a substitute for a properly converged production calculation (larger basis sets,
geometry optimization, solvation models, etc. all change the numbers). Runtimes
are kept to roughly a minute or two by capping atom count and default basis size;
pushing those limits is possible but turns a web request into a background job,
which this page does not attempt.
"""
import io
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hard caps so a single web request can't hang the server for a large system.
MAX_ISING_L = 128
MAX_MC_SAMPLES = 2_000_000
MAX_WALK_STEPS = 200_000
MAX_WALKERS = 2_000
MAX_DFT_ATOMS = 30

DFT_BASIS_OPTIONS = ['sto-3g', '6-31g', '6-31g*', '6-31+g*']
DFT_FUNCTIONAL_OPTIONS = ['b3lyp', 'pbe', 'pbe0', 'blyp', 'hf']


# ---------------------------------------------------------------------------
# Monte Carlo: 2D Ising model
# ---------------------------------------------------------------------------

def run_ising_2d(L=32, temperature=2.269, n_sweeps=400, J=1.0, seed=None, out_dir=None, tag=''):
    """Metropolis Monte Carlo on an LxL periodic 2D Ising lattice (spins +/-1).

    Returns per-sweep magnetization/energy traces (after an internal equilibration
    burn-in that isn't plotted) plus a snapshot of the final spin configuration.
    T_c for the 2D Ising model is exactly 2/ln(1+sqrt(2)) ~= 2.269 (J=k_B=1) —
    the default temperature sits right at that critical point.
    """
    L = int(max(4, min(MAX_ISING_L, L)))
    n_sweeps = int(max(10, min(4000, n_sweeps)))
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    def local_energy_delta(s, i, j):
        neighbors = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
        return 2.0 * J * s[i, j] * neighbors

    def total_energy(s):
        right = np.roll(s, -1, axis=1)
        down = np.roll(s, -1, axis=0)
        return -J * np.sum(s * right + s * down)

    burn_in = max(20, n_sweeps // 5)
    mags, energies = [], []
    beta = 1.0 / max(temperature, 1e-6)
    n_sites = L * L

    for sweep in range(n_sweeps + burn_in):
        # one Metropolis sweep = N attempted single-spin flips
        ii = rng.integers(0, L, size=n_sites)
        jj = rng.integers(0, L, size=n_sites)
        rand_vals = rng.random(n_sites)
        for k in range(n_sites):
            i, j = ii[k], jj[k]
            dE = local_energy_delta(spins, i, j)
            if dE <= 0 or rand_vals[k] < np.exp(-beta * dE):
                spins[i, j] *= -1
        if sweep >= burn_in:
            mags.append(abs(np.sum(spins)) / n_sites)
            energies.append(total_energy(spins) / n_sites)

    mags = np.array(mags)
    energies = np.array(energies)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title(f'Final spin configuration (T={temperature:.3f})')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].plot(mags, color='#2b6cb0', linewidth=1)
    axes[1].set_xlabel('Sweep (post burn-in)'); axes[1].set_ylabel('|Magnetization| / site')
    axes[1].set_title('Magnetization')
    axes[2].plot(energies, color='#b3261e', linewidth=1)
    axes[2].set_xlabel('Sweep (post burn-in)'); axes[2].set_ylabel('Energy / site (J units)')
    axes[2].set_title('Energy')
    fig.tight_layout()

    fname = f"ising_{tag}{int(time.time()*1000)}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)

    return {
        'plot_filename': fname,
        'L': L, 'temperature': temperature, 'n_sweeps': n_sweeps, 'J': J,
        'mean_abs_magnetization': round(float(np.mean(mags)), 5),
        'std_magnetization': round(float(np.std(mags)), 5),
        'mean_energy_per_site': round(float(np.mean(energies)), 5),
        'T_c_reference': round(float(2.0 / np.log(1 + np.sqrt(2))), 5),
    }


# ---------------------------------------------------------------------------
# Monte Carlo: numerical integration
# ---------------------------------------------------------------------------

MC_INTEGRALS = {
    'circle': {
        'label': 'Area of unit circle (-> pi), 2D',
        'dim': 2, 'bounds': (-1.0, 1.0),
        'indicator': lambda pts: (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0,
        'domain_volume': 4.0,
        'exact': np.pi,
    },
    'sphere': {
        'label': 'Volume of unit sphere, 3D',
        'dim': 3, 'bounds': (-1.0, 1.0),
        'indicator': lambda pts: (pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2) <= 1.0,
        'domain_volume': 8.0,
        'exact': 4.0 / 3.0 * np.pi,
    },
    'gaussian_2d': {
        'label': 'Integral of exp(-(x^2+y^2)) over [-3,3]^2 (-> pi, as bound->inf)',
        'dim': 2, 'bounds': (-3.0, 3.0),
        'func': lambda pts: np.exp(-(pts[:, 0] ** 2 + pts[:, 1] ** 2)),
        'domain_volume': 36.0,
        'exact': np.pi,  # true value on [-3,3]^2 is pi minus a tiny tail correction
    },
}


def run_mc_integration(target='circle', n_samples=100000, seed=None, out_dir=None, tag=''):
    if target not in MC_INTEGRALS:
        target = 'circle'
    spec = MC_INTEGRALS[target]
    n_samples = int(max(100, min(MAX_MC_SAMPLES, n_samples)))
    rng = np.random.default_rng(seed)
    lo, hi = spec['bounds']
    pts = rng.uniform(lo, hi, size=(n_samples, spec['dim']))

    if 'indicator' in spec:
        inside = spec['indicator'](pts)
        values = inside.astype(float)
    else:
        values = spec['func'](pts)

    # running estimate, sampled at log-spaced checkpoints, to show MC convergence
    checkpoints = np.unique(np.logspace(np.log10(20), np.log10(n_samples), 60).astype(int))
    running = np.cumsum(values) / np.arange(1, n_samples + 1)
    estimates = spec['domain_volume'] * running[checkpoints - 1]

    final_estimate = spec['domain_volume'] * np.mean(values)
    std_err = spec['domain_volume'] * np.std(values) / np.sqrt(n_samples)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(checkpoints, estimates, color='#2b6cb0', linewidth=1.3, label='MC estimate')
    if spec.get('exact') is not None:
        ax.axhline(spec['exact'], color='#b3261e', linestyle='--', linewidth=1, label=f"Reference = {spec['exact']:.6f}")
    ax.set_xscale('log')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Estimate')
    ax.set_title(spec['label'])
    ax.legend()
    fig.tight_layout()

    fname = f"mcint_{tag}{int(time.time()*1000)}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)

    return {
        'plot_filename': fname, 'target': target, 'label': spec['label'],
        'n_samples': n_samples, 'estimate': round(float(final_estimate), 6),
        'std_error': round(float(std_err), 6),
        'exact': spec.get('exact'),
        'abs_error': round(abs(float(final_estimate) - spec['exact']), 6) if spec.get('exact') is not None else None,
    }


# ---------------------------------------------------------------------------
# Monte Carlo: random walk / diffusion
# ---------------------------------------------------------------------------

def run_random_walk(n_steps=2000, n_walkers=200, dim=2, step_size=1.0, seed=None, out_dir=None, tag=''):
    n_steps = int(max(10, min(MAX_WALK_STEPS, n_steps)))
    n_walkers = int(max(1, min(MAX_WALKERS, n_walkers)))
    dim = int(max(1, min(3, dim)))
    rng = np.random.default_rng(seed)

    # isotropic random step directions of fixed length step_size
    steps = rng.normal(size=(n_walkers, n_steps, dim))
    norms = np.linalg.norm(steps, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    steps = steps / norms * step_size

    positions = np.cumsum(steps, axis=1)  # (n_walkers, n_steps, dim)
    positions = np.concatenate([np.zeros((n_walkers, 1, dim)), positions], axis=1)

    sq_disp = np.sum(positions ** 2, axis=2)  # (n_walkers, n_steps+1)
    msd = np.mean(sq_disp, axis=0)
    t = np.arange(n_steps + 1)

    # linear fit of MSD vs t (skip the first few points) -> diffusion coefficient
    # theory: MSD = 2*dim*D*t for an unbiased random walk
    fit_start = max(1, n_steps // 20)
    slope, intercept = np.polyfit(t[fit_start:], msd[fit_start:], 1)
    D_est = slope / (2 * dim)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    n_show = min(30, n_walkers)
    if dim >= 2:
        for w in range(n_show):
            axes[0].plot(positions[w, :, 0], positions[w, :, 1], linewidth=0.6, alpha=0.7)
        axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        axes[0].set_title(f'{n_show} of {n_walkers} walker paths (2D projection)')
        axes[0].set_aspect('equal', adjustable='datalim')
    else:
        for w in range(n_show):
            axes[0].plot(t, positions[w, :, 0], linewidth=0.6, alpha=0.7)
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('Position')
        axes[0].set_title(f'{n_show} of {n_walkers} walker traces (1D)')

    axes[1].plot(t, msd, color='#2b6cb0', linewidth=1.3, label='MSD (simulated)')
    axes[1].plot(t, slope * t + intercept, color='#b3261e', linestyle='--', linewidth=1,
                 label=f'Linear fit -> D = {D_est:.4f}')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Mean squared displacement')
    axes[1].set_title('MSD vs. step (diffusion coefficient from slope)')
    axes[1].legend()
    fig.tight_layout()

    fname = f"randwalk_{tag}{int(time.time()*1000)}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)

    return {
        'plot_filename': fname, 'n_steps': n_steps, 'n_walkers': n_walkers, 'dim': dim,
        'step_size': step_size, 'diffusion_coefficient': round(float(D_est), 6),
        'final_msd': round(float(msd[-1]), 4),
    }


# ---------------------------------------------------------------------------
# DFT: real single-point calculation via PySCF
# ---------------------------------------------------------------------------

class DFTInputError(ValueError):
    pass


def _geometry_from_smiles(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise DFTInputError('RDKit is not installed on the server — SMILES input needs it (pip install rdkit). '
                             'You can paste XYZ coordinates directly instead.')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DFTInputError(f'Could not parse SMILES: {smiles!r}')
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_DFT_ATOMS:
        raise DFTInputError(f'Molecule has {n_atoms} atoms (incl. H); this page caps single-point DFT at '
                             f'{MAX_DFT_ATOMS} atoms to keep runtimes reasonable for a web request.')
    embed_result = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE, useRandomCoords=True)
    if embed_result != 0:
        raise DFTInputError('RDKit could not generate a 3D geometry for this molecule.')
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # fall back to the raw embedded geometry if MMFF parameters are missing
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return '; '.join(lines), n_atoms


def _validate_xyz(xyz_text):
    lines = [ln.strip() for ln in xyz_text.replace(';', '\n').splitlines() if ln.strip()]
    if not lines:
        raise DFTInputError('No atoms given.')
    if len(lines) > MAX_DFT_ATOMS:
        raise DFTInputError(f'{len(lines)} atoms given; this page caps single-point DFT at {MAX_DFT_ATOMS} atoms.')
    for ln in lines:
        parts = ln.split()
        if len(parts) != 4:
            raise DFTInputError(f"Bad geometry line (expected 'Element x y z'): {ln!r}")
        try:
            [float(p) for p in parts[1:]]
        except ValueError:
            raise DFTInputError(f'Non-numeric coordinate in line: {ln!r}')
    return '; '.join(lines), len(lines)


def run_dft(structure_input, input_type='smiles', basis='6-31g', functional='b3lyp',
            charge=0, spin=0, label=''):
    """Real single-point restricted (or unrestricted, if spin != 0) Kohn-Sham DFT via PySCF.

    Returns total energy, HOMO/LUMO orbital energies and gap, and dipole moment —
    all computed for real from the given geometry, not looked up or approximated.
    """
    from pyscf import gto, dft

    if basis not in DFT_BASIS_OPTIONS:
        basis = '6-31g'
    if functional not in DFT_FUNCTIONAL_OPTIONS:
        functional = 'b3lyp'

    if input_type == 'smiles':
        atom_str, n_atoms = _geometry_from_smiles(structure_input)
    else:
        atom_str, n_atoms = _validate_xyz(structure_input)

    t0 = time.time()
    mol = gto.M(atom=atom_str, basis=basis, charge=int(charge), spin=int(spin), verbose=0)

    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = functional
    energy_hartree = mf.kernel()
    elapsed = time.time() - t0

    if not mf.converged:
        raise DFTInputError('SCF did not converge for this molecule/basis/functional combination — '
                             'try a different basis set or check the input geometry and charge/spin.')

    HARTREE_TO_EV = 27.211386245988

    if spin == 0:
        n_occ = mol.nelectron // 2
        mo = mf.mo_energy
        homo_ev = float(mo[n_occ - 1]) * HARTREE_TO_EV
        lumo_ev = float(mo[n_occ]) * HARTREE_TO_EV if n_occ < len(mo) else None
    else:
        mo_a, mo_b = mf.mo_energy
        n_occ_a, n_occ_b = mf.nelec
        homo_a = mo_a[n_occ_a - 1] if n_occ_a > 0 else None
        homo_b = mo_b[n_occ_b - 1] if n_occ_b > 0 else None
        homo_ev = float(max(x for x in (homo_a, homo_b) if x is not None)) * HARTREE_TO_EV
        lumo_candidates = []
        if n_occ_a < len(mo_a):
            lumo_candidates.append(mo_a[n_occ_a])
        if n_occ_b < len(mo_b):
            lumo_candidates.append(mo_b[n_occ_b])
        lumo_ev = float(min(lumo_candidates)) * HARTREE_TO_EV if lumo_candidates else None

    gap_ev = (lumo_ev - homo_ev) if (homo_ev is not None and lumo_ev is not None) else None

    try:
        dip = mf.dip_moment(unit='Debye', verbose=0)
        dipole_debye = round(float(np.linalg.norm(dip)), 4)
    except Exception:
        dipole_debye = None

    return {
        'label': label or 'Molecule',
        'n_atoms': n_atoms,
        'basis': basis,
        'functional': functional,
        'charge': int(charge),
        'spin': int(spin),
        'energy_hartree': round(float(energy_hartree), 8),
        'energy_ev': round(float(energy_hartree) * HARTREE_TO_EV, 4),
        'homo_ev': round(homo_ev, 4) if homo_ev is not None else None,
        'lumo_ev': round(lumo_ev, 4) if lumo_ev is not None else None,
        'gap_ev': round(gap_ev, 4) if gap_ev is not None else None,
        'dipole_debye': dipole_debye,
        'converged': bool(mf.converged),
        'elapsed_seconds': round(elapsed, 2),
        'geometry_xyz': atom_str.replace('; ', '\n'),
    }


def run_dft_job(**kwargs):
    """Entry point for the background compute worker (RQ). Never raises: returns
    {'ok': True, 'result': ...} or {'ok': False, 'error': <message for the user>}, so the
    web app can show the same inline error it would for a synchronous run instead of a
    worker traceback."""
    try:
        return {'ok': True, 'result': run_dft(**kwargs)}
    except DFTInputError as e:
        return {'ok': False, 'error': str(e)}
    except ImportError:
        return {'ok': False, 'error': 'PySCF is not installed on this server, so DFT is unavailable here.'}
    except Exception as e:
        return {'ok': False, 'error': f'DFT calculation failed: {e}'}
