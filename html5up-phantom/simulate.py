"""Synthetic-data generators behind the "Simulate" tab.

Pure functions: form values in, CSV text plus an "answer key" out — no Flask, no database.
Anything that needs the app's own reference data (tryptic digest, adduct table, EDS line
table, curve models) is passed in through `refs`, so this module never imports app.py.
Every generator returns:
    {'csv': str, 'label': str, 'key': [str, ...], 'next_tab': str|None, 'prominence': float|None}
or {'error': str}.
"""
import re
import numpy as np

FWHM_TO_SIGMA = 1 / 2.3548
SQRT_2PI = np.sqrt(2 * np.pi)


# ---- form helpers ----------------------------------------------------------------------

def _num(form, name, default, cast=float, lo=None, hi=None):
    try:
        v = cast(str(form.get(name, '')).strip())
    except (ValueError, TypeError):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _rng(form):
    seed = str(form.get('seed', '')).strip()
    return np.random.default_rng(int(seed) if re.fullmatch(r'-?\d+', seed) else None)


def _rows(text, min_cols, max_rows=200):
    rows = []
    for line in (text or '').splitlines():
        parts = [p.strip() for p in re.split(r'[,\t;]+', line.strip()) if p.strip() != '']
        if len(parts) < min_cols:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return rows[:max_rows]


def _to_csv(headers, columns):
    lines = [','.join(headers)]
    for row in zip(*columns):
        lines.append(','.join(f'{v:.6g}' if isinstance(v, (float, np.floating)) else str(v) for v in row))
    return '\n'.join(lines) + '\n'


def _add_noise(y, rng, noise_pct):
    if noise_pct <= 0:
        return y
    return y + rng.normal(0, noise_pct / 100 * max(np.max(np.abs(y)), 1e-12), len(y))


def _prominence(y_clean, noise_pct, frac=0.05):
    """A Peak Picking prominence that clears the noise we just added. Over a few thousand
    points the tallest noise spikes reach roughly 7σ of prominence, so anything lower would
    pick up noise as peaks."""
    peak = float(np.max(np.abs(y_clean)))
    return max(frac * peak, 7 * noise_pct / 100 * peak)


def peak_profile(x, center, height, fwhm, shape='gaussian', tail=0.0):
    """One peak on an ascending x grid. `tail` > 0 convolves with an exponential decay (an
    exponentially-modified Gaussian) — the peak tailing seen on real chromatography columns;
    the convolution preserves area but lowers and shifts the apex."""
    fwhm = max(fwhm, 1e-9)
    if shape == 'lorentzian':
        y = height / (1 + ((x - center) / (fwhm / 2)) ** 2)
    else:
        sigma = fwhm * FWHM_TO_SIGMA
        y = height * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    if tail > 0 and len(x) > 2:
        dx = float(np.median(np.diff(x)))
        n = int(min(len(x), 10 * tail / dx)) + 1
        kernel = np.exp(-np.arange(n) * dx / tail)
        y = np.convolve(y, kernel / kernel.sum())[:len(x)]
    return y


# ---- per-technique UI specs (drive the Simulate tab form) -------------------------------

def _peak_field(default, help_text):
    return {'name': 'peaks', 'label': 'Peaks — one per line: position, height, FWHM',
            'kind': 'textarea', 'default': default, 'help': help_text}


COMMON_FIELDS = [
    {'name': 'noise_pct', 'label': 'Noise (% of tallest signal)', 'kind': 'number', 'default': 1.5, 'step': 0.1},
    {'name': 'seed', 'label': 'Random seed (optional — same seed, same data)', 'kind': 'text', 'default': ''},
]


def _spectrum_mode(x_label, x_start, x_end, points, peaks, shape='gaussian', tail=False, extra_help=''):
    fields = [
        {'name': 'x_start', 'label': f'{x_label} — start', 'kind': 'number', 'default': x_start, 'step': 'any'},
        {'name': 'x_end', 'label': f'{x_label} — end', 'kind': 'number', 'default': x_end, 'step': 'any'},
        {'name': 'points', 'label': 'Data points', 'kind': 'number', 'default': points, 'step': 1},
        _peak_field(peaks, 'Height can be negative (e.g. CD bands). FWHM = full width at half maximum. ' + extra_help),
        {'name': 'shape', 'label': 'Peak shape', 'kind': 'select', 'default': shape,
         'options': [('gaussian', 'Gaussian'), ('lorentzian', 'Lorentzian')]},
        {'name': 'baseline', 'label': 'Baseline offset', 'kind': 'number', 'default': 0, 'step': 'any'},
    ]
    if tail:
        fields.append({'name': 'tail', 'label': 'Tailing time constant (0 = symmetric peaks)', 'kind': 'number',
                       'default': 0, 'step': 0.01, 'help': 'Same units as the x axis. ~0.05–0.3 gives visible tailing.'})
    return fields


SIM_SPECS = {
    'FTIR': {'x': ('Wavenumber_cm-1', 'Absorbance'), 'modes': [
        {'key': 'spectrum', 'label': 'Spectrum', 'fields': _spectrum_mode(
            'Wavenumber (cm⁻¹)', 400, 4000, 1800, '3300, 0.6, 180\n2920, 0.5, 60\n1710, 0.9, 40\n1600, 0.4, 50\n1050, 0.7, 80')}]},
    'UV-Vis': {'x': ('Wavelength_nm', 'Absorbance'), 'modes': [
        {'key': 'spectrum', 'label': 'Spectrum', 'fields': _spectrum_mode(
            'Wavelength (nm)', 200, 800, 1200, '260, 0.8, 30\n340, 0.4, 45')}]},
    'Fluorescence': {'x': ('Emission_nm', 'Intensity'), 'modes': [
        {'key': 'spectrum', 'label': 'Emission spectrum', 'fields': _spectrum_mode(
            'Emission wavelength (nm)', 400, 700, 900, '520, 1000, 40\n560, 400, 50')}]},
    'Raman': {'x': ('Raman_shift_cm-1', 'Intensity'), 'modes': [
        {'key': 'spectrum', 'label': 'Spectrum', 'fields': _spectrum_mode(
            'Raman shift (cm⁻¹)', 100, 3500, 1700, '1350, 600, 60\n1580, 900, 40\n2700, 500, 70',
            extra_help='Default mimics carbon: D, G and 2D bands.')}]},
    'NMR (1H, 13C)': {'x': ('Shift_ppm', 'Intensity'), 'modes': [
        {'key': 'spectrum', 'label': '1D spectrum', 'fields': _spectrum_mode(
            'Chemical shift (ppm)', 0, 12, 3000, '7.3, 1.0, 0.05\n3.7, 0.6, 0.04\n1.2, 0.9, 0.04', shape='lorentzian')}]},
    'CD (Circular Dichroism)': {'x': ('Wavelength_nm', 'Ellipticity'), 'modes': [
        {'key': 'spectrum', 'label': 'Spectrum', 'fields': _spectrum_mode(
            'Wavelength (nm)', 190, 260, 700, '208, -12000, 10\n222, -11000, 12\n192, 20000, 8',
            extra_help='Default is an α-helix-like signature.')}]},
    'HPLC / GC': {'x': ('RT_min', 'Signal'), 'modes': [
        {'key': 'chromatogram', 'label': 'Chromatogram', 'fields': _spectrum_mode(
            'Retention time (min)', 0.5, 8, 3000, '2.0, 3200, 0.20\n3.2, 1800, 0.24\n4.5, 2600, 0.20', tail=True,
            extra_help='Default: three compounds.')}]},
    'LC-MS': {'x': ('RT_min', 'Intensity'), 'modes': [
        {'key': 'chromatogram', 'label': 'Chromatogram', 'fields': _spectrum_mode(
            'Retention time (min)', 0.5, 8, 3000, '1.5, 9500, 0.15\n3.05, 4200, 0.18', tail=True)},
        {'key': 'features', 'label': 'm/z feature table (for Formula ID)', 'fields': [
            {'name': 'compounds', 'label': 'Compounds — one per line: neutral mass, retention time, intensity',
             'kind': 'textarea', 'default': '180.0634, 1.52, 95000\n342.1162, 3.05, 42000\n194.0804, 4.48, 15000',
             'help': 'Defaults are glucose, sucrose and caffeine (monoisotopic neutral masses).'},
            {'name': 'adduct', 'label': 'Ionization / adduct', 'kind': 'select', 'default': 'M+H', 'options': 'ADDUCTS'},
            {'name': 'ppm_error', 'label': 'Mass error (ppm, 1 σ)', 'kind': 'number', 'default': 2, 'step': 0.5}]}]},
    'EDS/EDX': {'x': ('Energy_keV', 'Counts'), 'modes': [
        {'key': 'eds', 'label': 'EDS spectrum from composition', 'fields': [
            {'name': 'composition', 'label': 'Elements — symbol:amount, comma-separated', 'kind': 'text',
             'default': 'C:30, O:50, Si:20',
             'help': 'Only elements the app has a line table for: C N O F Na Mg Al Si P S Cl K Ca Ti Cr Mn Fe Co Ni Cu Zn Ag Pt Au.'},
            {'name': 'points', 'label': 'Data points', 'kind': 'number', 'default': 1000, 'step': 1}]}]},
    'MALDI': {'x': ('m/z', 'Intensity'), 'modes': [
        {'key': 'polymer', 'label': 'Polymer distribution (Mn / Mw / PDI)', 'fields': [
            {'name': 'repeat_mass', 'label': 'Repeat-unit mass (Da)', 'kind': 'number', 'default': 44.026, 'step': 'any',
             'help': '44.026 = ethylene glycol (PEG).'},
            {'name': 'end_mass', 'label': 'End groups (Da)', 'kind': 'number', 'default': 18.011, 'step': 'any'},
            {'name': 'cation_mass', 'label': 'Cation adduct (Da)', 'kind': 'number', 'default': 22.989, 'step': 'any',
             'help': '22.989 = Na⁺.'},
            {'name': 'n_mean', 'label': 'Mean repeat count', 'kind': 'number', 'default': 25, 'step': 1},
            {'name': 'n_sd', 'label': 'Spread of repeat count (σ)', 'kind': 'number', 'default': 5, 'step': 0.5},
            {'name': 'fwhm', 'label': 'Peak width (Da)', 'kind': 'number', 'default': 0.6, 'step': 0.1}]},
        {'key': 'pmf', 'label': 'Peptide digest (for PMF)', 'fields': [
            {'name': 'sequence', 'label': 'Protein sequence (single-letter code)', 'kind': 'textarea',
             'default': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR',
             'help': 'Enter this same sequence on the PMF tab to score the match.'},
            {'name': 'missed', 'label': 'Missed cleavages in the digest', 'kind': 'number', 'default': 1, 'step': 1},
            {'name': 'fraction', 'label': 'Fraction of peptides observed (%)', 'kind': 'number', 'default': 65, 'step': 5},
            {'name': 'mass_sd', 'label': 'Mass error (Da, 1 σ)', 'kind': 'number', 'default': 0.05, 'step': 0.01},
            {'name': 'n_noise', 'label': 'Unrelated noise peaks', 'kind': 'number', 'default': 6, 'step': 1}]},
        {'key': 'imaging', 'label': 'Ion image (for Imaging)', 'fields': [
            {'name': 'width', 'label': 'Grid width (pixels)', 'kind': 'number', 'default': 30, 'step': 1},
            {'name': 'height', 'label': 'Grid height (pixels)', 'kind': 'number', 'default': 24, 'step': 1},
            {'name': 'hotspots', 'label': 'Hotspots — one per line: x, y, amplitude, radius (σ, pixels)', 'kind': 'textarea',
             'default': '10, 8, 900, 3\n22, 16, 600, 4'},
            {'name': 'background', 'label': 'Background intensity', 'kind': 'number', 'default': 80, 'step': 'any'}]}]},
}


# ---- generators ------------------------------------------------------------------------

def _spectrum(technique, form, rng):
    spec = SIM_SPECS[technique]
    x_name, y_name = spec['x']
    mode = spec['modes'][0]
    defaults = {f['name']: f['default'] for f in mode['fields']}
    x0 = _num(form, 'x_start', float(defaults['x_start']))
    x1 = _num(form, 'x_end', float(defaults['x_end']))
    if x0 == x1:
        return {'error': 'Start and end must differ.'}
    n = _num(form, 'points', int(defaults['points']), int, 50, 20000)
    x = np.linspace(min(x0, x1), max(x0, x1), n)
    peaks = _rows(form.get('peaks', ''), 3)
    if not peaks:
        return {'error': 'Add at least one peak as "position, height, FWHM" (numbers only).'}
    shape = form.get('shape', 'gaussian')
    tail = _num(form, 'tail', 0.0, float, 0.0, 1e6)
    y = np.full(n, _num(form, 'baseline', 0.0))
    key = []
    is_chrom = 'tail' in defaults
    for row in peaks:
        c, h, w = row[0], row[1], row[2]
        y = y + peak_profile(x, c, h, w, shape, tail)
        line = f"Peak at {c:g}: height {h:g}, FWHM {w:g}"
        if is_chrom and shape == 'gaussian' and h > 0 and w > 0:
            sigma = w * FWHM_TO_SIGMA
            line += f" → true area ≈ {h * sigma * SQRT_2PI:.4g}"
            if tail == 0:
                line += f", plate count N ≈ {5.54 * (c / w) ** 2:.0f}"
        key.append(line)
    if is_chrom and tail > 0:
        key.append(f"Tailing time constant {tail:g} — expect a USP tailing factor above 1 and the apex shifted later than the listed position.")
    noise_pct = _num(form, 'noise_pct', 1.5, float, 0, 50)
    prominence = _prominence(y, noise_pct)
    if prominence > 0.05 * float(np.max(np.abs(y))) * 1.01:
        key.append(f"At {noise_pct:g}% noise, Peak Picking prominence was set to ~{prominence:.3g} to stay above the noise — peaks smaller than that won't be picked. Lower the noise to resolve smaller peaks.")
    y = _add_noise(y, rng, noise_pct)
    return {
        'csv': _to_csv([x_name, y_name], [x, y]),
        'label': f"Simulated {technique} " + ('chromatogram' if is_chrom else 'spectrum'),
        'key': key, 'next_tab': None, 'prominence': prominence,
    }


def _eds(form, rng, refs):
    line_table = {}
    for lo, hi, label in refs['EDS_TABLE']:
        line_table[label.split()[0]] = (lo + hi) / 2
    parts = [p for p in re.split(r'[,;\n]+', form.get('composition', '')) if p.strip()]
    comp = {}
    for p in parts:
        m = re.fullmatch(r'\s*([A-Za-z]{1,2})\s*[:=]\s*([0-9.]+)\s*', p)
        if not m:
            return {'error': f'Could not read "{p.strip()}" — use symbol:amount, e.g. Fe:40.'}
        sym = m.group(1).capitalize()
        if sym not in line_table:
            return {'error': f'No characteristic line table for "{sym}".'}
        comp[sym] = comp.get(sym, 0) + float(m.group(2))
    if not comp:
        return {'error': 'Enter at least one element, e.g. C:30, O:50, Si:20.'}
    n = _num(form, 'points', 1000, int, 100, 20000)
    x = np.linspace(0.1, 10, n)
    y = 40 * np.exp(-x / 3) + 5     # smooth bremsstrahlung-like background
    total = sum(comp.values())
    key = []
    for sym, amount in comp.items():
        y = y + peak_profile(x, line_table[sym], amount * 60, 0.13)
        key.append(f"{sym}: amount {amount:g} → peak near {line_table[sym]:.2f} keV; expected share of identified signal ≈ {100 * amount / total:.1f}%")
    noise_pct = _num(form, 'noise_pct', 1.5, float, 0, 50)
    prominence = _prominence(y, noise_pct, 0.04)
    y = np.clip(_add_noise(y, rng, noise_pct), 0, None)
    return {'csv': _to_csv(['Energy_keV', 'Counts'], [x, y]), 'label': 'Simulated EDS/EDX spectrum',
            'key': key, 'next_tab': None, 'prominence': prominence}


def _lcms_features(form, rng, refs):
    rows = _rows(form.get('compounds', ''), 3)
    if not rows:
        return {'error': 'Add compounds as "neutral mass, retention time, intensity".'}
    adduct = form.get('adduct', 'M+H')
    if adduct not in refs['LC_MS_ADDUCTS']:
        adduct = 'M+H'
    shift = refs['LC_MS_ADDUCTS'][adduct][1]
    ppm = _num(form, 'ppm_error', 2.0, float, 0, 50)
    rt, mz, inten, key = [], [], [], []
    for mass, r, i in ((r[0], r[1], r[2]) for r in rows):
        observed = (mass + shift) * (1 + rng.normal(0, ppm) / 1e6)
        rt.append(r); mz.append(observed); inten.append(i)
        key.append(f"Neutral mass {mass:g} at RT {r:g} → observed m/z {observed:.4f} ({refs['LC_MS_ADDUCTS'][adduct][0].strip()}); Formula ID (same adduct) should list a formula within a few ppm of {mass:g}")
    return {'csv': _to_csv(['RT', 'm/z', 'Intensity'], [rt, mz, inten]), 'label': 'Simulated LC-MS feature table',
            'key': key, 'next_tab': 'Formula ID', 'prominence': None}


def _maldi_polymer(form, rng):
    rep = _num(form, 'repeat_mass', 44.026, float, 1, 5000)
    end = _num(form, 'end_mass', 18.011)
    cat = _num(form, 'cation_mass', 22.989)
    nm = _num(form, 'n_mean', 25.0, float, 2, 500)
    nsd = _num(form, 'n_sd', 5.0, float, 0.3, 100)
    fwhm = _num(form, 'fwhm', 0.6, float, 0.05, 50)
    ns = np.arange(max(1, int(nm - 4 * nsd)), int(nm + 4 * nsd) + 1)
    weights = np.exp(-((ns - nm) ** 2) / (2 * nsd ** 2))
    masses = ns * rep + end + cat
    heights = 10000 * weights
    x = np.arange(masses.min() - 4 * rep, masses.max() + 4 * rep, fwhm / 8)
    if len(x) > 40000:
        return {'error': 'That range is too large — use a smaller spread or repeat count.'}
    y = np.full(len(x), 10.0)
    for m, h in zip(masses, heights):
        y = y + peak_profile(x, m, h, fwhm)
    noise_pct = _num(form, 'noise_pct', 1.5, float, 0, 50)
    prominence = _prominence(y, noise_pct, 0.02)
    y = _add_noise(y, rng, noise_pct)
    mn = float(np.sum(heights * masses) / np.sum(heights))
    mw = float(np.sum(heights * masses ** 2) / np.sum(heights * masses))
    key = [f"{len(ns)} oligomer peaks, n = {ns.min()}–{ns.max()}, spaced {rep:g} Da apart; peak m/z = n × {rep:g} + {end + cat:g}",
           f"Expected from the peak list (what the Analysis tab computes): Mn ≈ {mn:.1f}, Mw ≈ {mw:.1f}, PDI ≈ {mw / mn:.4f}",
           f"Peak Picking prominence was set to ~{prominence:.3g}; oligomers in the far tails (below that height) won't be picked, which is normal for real spectra too."]
    return {'csv': _to_csv(['m/z', 'Intensity'], [x, y]), 'label': 'Simulated MALDI polymer spectrum',
            'key': key, 'next_tab': None, 'prominence': prominence}


def _maldi_pmf(form, rng, refs):
    peptides, seq = refs['tryptic_digest'](form.get('sequence', ''), _num(form, 'missed', 1, int, 0, 3))
    if not seq:
        return {'error': 'Enter a protein sequence.'}
    unknown = sorted(set(seq) - set(refs['AA_MONO_MASS']))
    if unknown:
        return {'error': f"Unsupported residue letter(s): {', '.join(unknown)}."}
    theo = [(p, refs['peptide_mono_mass'](p['sequence']) + refs['PROTON_MASS']) for p in peptides]
    theo = [(p, mz) for p, mz in theo if 500 <= mz <= 4000]
    if not theo:
        return {'error': 'No peptides in the 500–4000 m/z window — use a longer sequence.'}
    frac = _num(form, 'fraction', 65.0, float, 5, 100) / 100
    picked = [t for t in theo if rng.random() < frac] or theo[:1]
    sd = _num(form, 'mass_sd', 0.05, float, 0, 1)
    obs = [(p, mz + rng.normal(0, sd), float(rng.uniform(1500, 10000))) for p, mz in picked]
    noise_mz = rng.uniform(600, min(4000, max(mz for _, mz in theo) + 100), _num(form, 'n_noise', 6, int, 0, 60))
    x = np.arange(500, min(4000, max(mz for _, mz in theo) + 200), 0.1)
    y = np.full(len(x), 15.0)
    for _, mz, h in obs:
        y = y + peak_profile(x, mz, h, 0.5)
    for mz in noise_mz:
        y = y + peak_profile(x, mz, float(rng.uniform(300, 1200)), 0.5)
    noise_pct = _num(form, 'noise_pct', 1.5, float, 0, 50)
    prominence = max(400.0, _prominence(y, noise_pct, 0.0))
    y = _add_noise(y, rng, noise_pct)
    covered = set()
    key = []
    for p, mz, h in obs:
        covered.update(range(p['start'], p['end'] + 1))
        key.append(f"Residues {p['start']}–{p['end']} {p['sequence']}: [M+H]⁺ ≈ {mz:.3f}")
    key.append(f"{len(obs)} of {len(theo)} theoretical peptides present, plus {len(noise_mz)} unrelated noise peaks; expected sequence coverage ≈ {100 * len(covered) / len(seq):.1f}%")
    key.append("Use the same sequence on the PMF tab (matching missed cleavages, tolerance ≥ 0.2 Da).")
    return {'csv': _to_csv(['m/z', 'Intensity'], [x, y]), 'label': 'Simulated MALDI peptide spectrum',
            'key': key, 'next_tab': 'PMF', 'prominence': prominence}


def _maldi_imaging(form, rng):
    w = _num(form, 'width', 30, int, 4, 100)
    h = _num(form, 'height', 24, int, 4, 100)
    spots = _rows(form.get('hotspots', ''), 4, 20)
    if not spots:
        return {'error': 'Add at least one hotspot as "x, y, amplitude, radius".'}
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    z = np.full((h, w), _num(form, 'background', 80.0), dtype=float)
    key = []
    for sx, sy, amp, rad in spots:
        z += amp * np.exp(-(((gx - sx) ** 2 + (gy - sy) ** 2) / (2 * max(rad, 0.3) ** 2)))
        key.append(f"Hotspot centred at ({sx:g}, {sy:g}), amplitude {amp:g}, radius σ {rad:g} px")
    z = np.clip(_add_noise(z.ravel(), rng, _num(form, 'noise_pct', 1.5, float, 0, 50)), 0, None)
    return {'csv': _to_csv(['x', 'y', 'intensity'], [gx.ravel(), gy.ravel(), z]),
            'label': 'Simulated MALDI ion image', 'key': key, 'next_tab': 'Imaging', 'prominence': None}


def run_simulation(technique, form, refs):
    """Entry point: pick the generator for this technique + chosen mode."""
    spec = SIM_SPECS.get(technique)
    if not spec:
        return {'error': f'No simulator for {technique} yet.'}
    mode = form.get('mode') or spec['modes'][0]['key']
    rng = _rng(form)
    if technique == 'EDS/EDX':
        return _eds(form, rng, refs)
    if technique == 'LC-MS' and mode == 'features':
        return _lcms_features(form, rng, refs)
    if technique == 'MALDI':
        return {'polymer': lambda: _maldi_polymer(form, rng),
                'pmf': lambda: _maldi_pmf(form, rng, refs),
                'imaging': lambda: _maldi_imaging(form, rng)}.get(mode, lambda: {'error': 'Unknown mode.'})()
    return _spectrum(technique, form, rng)


# ---- generic curves (Data Interpretation workspace) -------------------------------------

CURVE_SPECS = {
    'linear': {'label': 'Linear', 'params': [('m', 2.0), ('c', 1.0)], 'x': (0, 10)},
    'quadratic': {'label': 'Quadratic', 'params': [('a', 0.5), ('b', -2.0), ('c', 3.0)], 'x': (0, 10)},
    'exponential': {'label': 'Exponential growth / decay', 'params': [('a', 5.0), ('b', -0.4), ('c', 1.0)], 'x': (0, 10)},
    'power': {'label': 'Power law', 'params': [('a', 2.0), ('b', 1.5)], 'x': (0.5, 10)},
    'logarithmic': {'label': 'Logarithmic', 'params': [('a', 3.0), ('b', 1.0)], 'x': (0.5, 20)},
    'langmuir': {'label': 'Langmuir isotherm (saturation)', 'params': [('a (max)', 9.0), ('b (half-sat.)', 1.5)], 'x': (0.2, 35)},
    'sigmoid': {'label': 'Sigmoidal (S-curve)', 'params': [('amplitude', 10.0), ('k', 1.2), ('x0', 5.0), ('offset', 0.5)], 'x': (0, 10)},
    'gaussian': {'label': 'Gaussian peak', 'params': [('amplitude', 10.0), ('mean', 5.0), ('sigma', 1.0), ('offset', 0.5)], 'x': (0, 10)},
}


def run_curve_simulation(form, curves):
    """`curves` maps shape name -> the app's own model function, so the simulated data is
    generated by exactly the functions the fitting and shape-recognition code uses."""
    shape = form.get('shape', 'langmuir')
    if shape not in CURVE_SPECS:
        return {'error': 'Unknown curve shape.'}
    spec = CURVE_SPECS[shape]
    params = [_num(form, f'p{i}', default) for i, (_, default) in enumerate(spec['params'])]
    x0 = _num(form, 'x_start', float(spec['x'][0]))
    x1 = _num(form, 'x_end', float(spec['x'][1]))
    if x0 >= x1:
        return {'error': 'X start must be smaller than X end.'}
    n = _num(form, 'points', 40, int, 8, 5000)
    x = np.linspace(x0, x1, n)
    with np.errstate(all='ignore'):
        y = np.asarray(curves[shape](x, *params), dtype=float)
    if not np.all(np.isfinite(y)):
        return {'error': 'Those parameters give undefined values over this range (e.g. log/power of a non-positive x).'}
    y = _add_noise(y, _rng(form), _num(form, 'noise_pct', 2.0, float, 0, 50))
    names = ', '.join(f"{name} = {p:g}" for (name, _), p in zip(spec['params'], params))
    key = [f"{spec['label']} with {names}, {n} points over x = {x0:g}…{x1:g}"]
    if shape != 'gaussian':
        key.append(f"The Analysis tab's Shape suggestions should point to this shape ({spec['label']}), though close relatives (e.g. sigmoid vs Langmuir on a short range) can outscore it when noise is high.")
    return {'csv': _to_csv(['X', 'Y'], [x, y]), 'label': f"Simulated {spec['label']}", 'key': key, 'next_tab': 'plot', 'prominence': None}
