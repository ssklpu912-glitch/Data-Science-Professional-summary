"""Best-effort parsers for native AFM vendor file formats.

Every loader here either returns a real, correctly-scaled 2D numpy array pulled
straight out of the instrument's own file, or returns None. Callers must always
treat None as "couldn't parse this" and fall back to storing the raw file
unparsed — never guess or fabricate a number when a parse fails.

Verified this session (installed + import-tested):
  .ibw (Asylum/Igor)     -> igor2 + AFMReader.ibw
  .spm (Bruker Nanoscope) -> pySPM (installed --no-deps, its numpy<2 pin is stale
                              and unnecessary on numpy 2.x) + AFMReader.spm
  .nid (Nanosurf)         -> NSFopen

.mdt (NT-MDT) is supported only if the user has separately installed the GPLv3
PyMDT module (github.com/symartin/PyMDT) themselves — it is NOT vendored into
this repo, since embedding GPL-licensed source into an otherwise-unlicensed
project is a licensing decision for the project owner, not something to do
silently. Without it, .mdt files are stored unparsed, same as .par/.wip/.pfm/.sfm.
"""

import numpy as np

# Candidate channel names to try, in priority order, when looking for the real
# height/topography channel in a multi-channel native file.
TOPOGRAPHY_CHANNEL_CANDIDATES = [
    'HeightTracee', 'HeightRetrace', 'Height', 'HeightSensor',
    'ZSensor', 'Height Sensor', 'Z-Axis',
]


def _load_ibw(filepath):
    from igor2 import binarywave
    from AFMReader.ibw import load_ibw

    scan = binarywave.load(filepath)
    labels = []
    for label_list in scan['wave']['labels']:
        for label in label_list:
            if label:
                labels.append(label.decode())
    if not labels:
        return None
    channel = next((c for c in TOPOGRAPHY_CHANNEL_CANDIDATES if c in labels), labels[0])

    image, pixel_size_nm = load_ibw(filepath, channel)
    return {'data': image, 'pixel_size_nm': pixel_size_nm, 'channel_name': channel, 'units': 'nm'}


def _load_spm(filepath):
    import pySPM
    from AFMReader.spm import load_spm

    scan = pySPM.Bruker(filepath)
    labels = []
    for layer in scan.layers:
        try:
            channel_name = layer[b'@2:Image Data'][0].decode('latin1').split('"')[1]
            labels.append(channel_name)
        except Exception:
            continue
    if not labels:
        return None
    channel = next((c for c in TOPOGRAPHY_CHANNEL_CANDIDATES if c in labels), labels[0])

    image, pixel_size_nm = load_spm(filepath, channel)
    return {'data': image, 'pixel_size_nm': pixel_size_nm, 'channel_name': channel, 'units': 'nm'}


def _load_nid(filepath):
    from NSFopen.read import read as nsf_read

    afm = nsf_read(filepath)
    if afm.data is None:
        return None

    forward = afm.data.get('Image', {}).get('Forward', {})
    channel_keys = list(forward.keys())
    if not channel_keys:
        return None
    channel = next(
        (c for c in channel_keys if 'z-axis' in c.lower() or 'height' in c.lower()),
        channel_keys[0],
    )

    is_height = 'z-axis' in channel.lower() or 'height' in channel.lower()
    array = np.array(forward[channel], dtype=float)
    array = array * 1e9 if is_height else array  # metres -> nm for real height data only
    units = 'nm' if is_height else 'a.u.'

    pixel_size_nm = None
    try:
        x_range_m = afm.param['X']['range'][0]
        pixel_size_nm = (x_range_m * 1e9) / array.shape[1]
    except Exception:
        pass

    return {'data': array, 'pixel_size_nm': pixel_size_nm, 'channel_name': channel, 'units': units}


def _load_mdt(filepath):
    try:
        import MDTfile  # noqa: F401  — optional, GPLv3, not vendored; user installs separately if wanted
    except ImportError:
        return None

    mdt = MDTfile.MDTFile(filepath)
    for frame in mdt:
        data = getattr(frame, 'data', None)
        if data is not None and np.ndim(data) == 2:
            return {'data': np.array(data, dtype=float), 'pixel_size_nm': None,
                     'channel_name': 'mdt_frame_0', 'units': 'a.u.'}
    return None


_LOADERS = {
    '.ibw': _load_ibw,
    '.spm': _load_spm,
    '.nid': _load_nid,
    '.mdt': _load_mdt,
}

NATIVE_EXTENSIONS = set(_LOADERS.keys())

# Vendor formats with no available open-source Python parser — accepted on
# upload and stored, but never decoded into numbers.
UNPARSED_EXTENSIONS = {'.par', '.wip', '.pfm', '.sfm'}


def try_parse_afm_native(filepath, ext):
    """Attempts to load real numeric data from a native AFM file.

    Returns {'data': 2D np.ndarray, 'pixel_size_nm': float|None,
    'channel_name': str, 'units': str} on success, or None on any failure —
    corrupt file, unexpected internal structure, missing optional dependency,
    or unrecognized extension. Callers must treat None as "not parseable",
    never as a signal to estimate a value some other way.
    """
    loader = _LOADERS.get(ext.lower())
    if loader is None:
        return None
    try:
        result = loader(filepath)
        if result is None:
            return None
        if result['data'] is None or result['data'].ndim != 2:
            return None
        return result
    except Exception:
        return None
