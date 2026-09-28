from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, abort
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from cachelib import FileSystemCache
import redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError
from sqlalchemy import or_, func, inspect as sa_inspect
import simulate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import io
import re
import time
import secrets
import json
import shutil
import hashlib
import smtplib
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import logging
from logging.handlers import RotatingFileHandler
import requests
import pdfplumber
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import skew as scipy_skew
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.restoration import unwrap_phase
from afm_formats import try_parse_afm_native, NATIVE_EXTENSIONS as AFM_NATIVE_EXTENSIONS, UNPARSED_EXTENSIONS as AFM_UNPARSED_EXTENSIONS
import computational
import confocal

def _load_secret_key():
    """Prefer a real deployment secret from the environment. Falls back to a
    key persisted alongside the app so local/dev sessions survive restarts
    without ever hardcoding a secret in source."""
    env_key = os.environ.get('FLASK_SECRET_KEY')
    if env_key:
        return env_key

    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.flask_secret_key')
    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            key = f.read().strip()
            if key:
                return key

    key = secrets.token_hex(32)
    with open(key_path, 'w') as f:
        f.write(key)
    return key


# Debug mode exposes Werkzeug's interactive debugger, which allows arbitrary code
# execution from any page that errors — it must never be on in production. Defaults
# to off; opt in locally with FLASK_DEBUG=1. Read once at module level since it also
# gates logging setup below, not just the dev-server call at the bottom of this file.
DEBUG_MODE = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')

def _database_url():
    """DATABASE_URL (Postgres in production) or the local SQLite file. Hosts commonly
    hand out postgres:// URLs; SQLAlchemy needs the dialect+driver spelled out."""
    url = os.environ.get('DATABASE_URL', '').strip()
    if not url:
        return 'sqlite:///users.db'
    for prefix in ('postgres://', 'postgresql://'):
        if url.startswith(prefix):
            return 'postgresql+psycopg://' + url[len(prefix):]
    return url


# Redis backs sessions, rate-limit counters and the compute job queue in production.
# Unset locally (no Redis on Windows), where each falls back to a single-machine
# equivalent: sessions on disk, limits in memory, compute jobs run inline.
REDIS_URL = os.environ.get('REDIS_URL', '').strip() or None
redis_conn = redis.Redis.from_url(REDIS_URL) if REDIS_URL else None

app = Flask(__name__)
app.secret_key = _load_secret_key()
# Behind a reverse proxy (Caddy/Nginx) every request otherwise appears to come from the
# proxy itself — which breaks per-IP rate limits and makes url_for(_external=True) emit
# http:// links. Trust exactly one proxy hop's X-Forwarded-* headers, and only when told to.
if os.environ.get('BEHIND_PROXY', '0').lower() in ('1', 'true', 'yes'):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config['SQLALCHEMY_DATABASE_URI'] = _database_url()
# Drop pooled connections the database closed while idle instead of erroring on first use.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Caps a single request's total upload size (all files in a multi-file upload combined) —
# without this, an unbounded upload can fill the disk. 100 MB comfortably covers a batch
# of microscopy images or a stack of AFM height maps.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True   # JS can't read the session cookie (mitigates XSS cookie theft)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # not sent on cross-site requests (mitigates CSRF)
# Secure requires HTTPS — off by default so local dev over plain HTTP still works.
# Set SESSION_COOKIE_SECURE=1 in production once the app is served over HTTPS.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', '0').lower() in ('1', 'true', 'yes')

# Server-side sessions: the cookie holds only a random session id. Flask's default
# cookie session is capped at ~4 KB and silently stops saving past that — and this app
# keeps per-technique plot state, compute results and DFT geometries in the session.
# JSON (not msgpack) keeps the old cookie session's behaviour of turning int dict keys
# into strings, which existing session-reading code relies on.
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_KEY_PREFIX'] = 'lablogbook:session:'
app.config['SESSION_SERIALIZATION_FORMAT'] = 'json'
if redis_conn is not None:
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_REDIS'] = redis_conn
else:
    app.config['SESSION_TYPE'] = 'cachelib'
    app.config['SESSION_CACHELIB'] = FileSystemCache(
        cache_dir=os.path.join(app.instance_path, 'sessions'), threshold=0,
    )
Session(app)

db = SQLAlchemy(app)
# render_as_batch: SQLite can't ALTER most column properties in place, so autogenerated
# migrations use Alembic's copy-and-swap batch mode (a plain ALTER on Postgres).
migrate = Migrate(app, db, render_as_batch=True)
csrf = CSRFProtect(app)


def _rate_limit_key():
    """Signed-in users are limited per account (so a shared lab/campus IP doesn't
    throttle everyone together); anonymous requests like login are limited per IP."""
    uid = session.get('user_id')
    return f'user:{uid}' if uid else get_remote_address()


limiter = Limiter(
    _rate_limit_key,
    app=app,
    storage_uri=REDIS_URL or 'memory://',
    key_prefix='lablogbook:limits',
    strategy='fixed-window',
)


def _login_email_key():
    """Second login limit keyed on the account being tried, so one account can't be
    brute-forced from many IPs at once."""
    return 'login-email:' + (request.form.get('email') or '').strip().lower()


# Compute routes are the expensive ones — and the obvious target for someone trying to
# tie up the server. Limits are per account (see _rate_limit_key).
MC_RATE_LIMIT = '20 per minute;200 per hour'
DFT_RATE_LIMIT = '5 per minute;30 per hour'
# DFT runs for up to about a minute — too long to hold a web thread, since a handful of
# concurrent runs would stall every other page. With Redis configured it's queued for the
# separate compute worker (`rq worker compute`) and the page polls until it's done; the
# number of worker processes is then the hard cap on concurrent DFT runs. Without Redis
# (local development) it still runs inline in the request.
compute_queue = Queue('compute', connection=redis_conn) if redis_conn is not None else None
DFT_JOB_TIMEOUT = 600            # seconds a worker may spend on one DFT run before it's killed
COMP_JOB_KEEP_SECONDS = 86400    # how long a finished job's result waits to be picked up

COMP_ENDPOINT_TABS = {
    'comp_run_ising': 'Ising Model',
    'comp_run_mc_integration': 'Monte Carlo Integration',
    'comp_run_random_walk': 'Random Walk',
    'comp_run_dft': 'Run Calculation',
}

# In debug mode Flask's reloader already prints everything to the console. Outside
# debug mode (i.e. any real deployment), nothing is logged anywhere by default —
# so route errors and warnings need somewhere durable to land.
if not DEBUG_MODE:
    log_dir = os.path.join(app.root_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=1_000_000, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('LabLogbook startup')

APP_VERSION = "1.1.0"
APP_LAST_UPDATED = "2026-09-16"


@app.context_processor
def inject_footer_context():
    return {
        'app_version': APP_VERSION,
        'app_last_updated': APP_LAST_UPDATED,
        'current_year': datetime.now().year,
    }


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class LogEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    category = db.Column(db.String(50), nullable=False)   # 'experiments', 'characterizations', 'miscellaneous'

    # Core identification
    title = db.Column(db.String(200), nullable=False)
    entry_datetime = db.Column(db.DateTime, nullable=False, default=datetime.now)
    exp_type = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(50), nullable=True)

    # Scientific content
    objective = db.Column(db.Text, nullable=True)
    materials = db.Column(db.Text, nullable=True)
    procedure = db.Column(db.Text, nullable=True)
    input_params = db.Column(db.Text, nullable=True)
    output_results = db.Column(db.Text, nullable=True)
    observations = db.Column(db.Text, nullable=True)
    conclusion = db.Column(db.Text, nullable=True)


class CharEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    technique_name = db.Column(db.String(150), nullable=False)
    date_scheduled = db.Column(db.Date, nullable=False, default=datetime.now)
    num_samples = db.Column(db.Integer, nullable=True)
    sample_prep = db.Column(db.String(50), nullable=True)     # powder/film/liquid/frozen/other
    outcome = db.Column(db.String(50), nullable=True)         # repeat needed/good data/synthesize again
    interpretation = db.Column(db.Text, nullable=True)


class DataFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    original_filename = db.Column(db.String(300), nullable=False)
    stored_filename = db.Column(db.String(300), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)   # 'tabular' or 'image'
    label = db.Column(db.String(200), nullable=True)        # optional note, e.g. "TEM sample 3"
    technique_name = db.Column(db.String(150), nullable=True)   # e.g. "SEM", "AFM" — None for files uploaded outside a technique workspace

    # chosen after upload, for tabular files
    x_column = db.Column(db.String(100), nullable=True)
    y_column = db.Column(db.String(100), nullable=True)
    fit_type = db.Column(db.String(30), nullable=True)      # linear/quadratic/exponential/gaussian/none
    fit_params = db.Column(db.Text, nullable=True)          # JSON-encoded list of fitted parameters
    r_squared = db.Column(db.Float, nullable=True)
    plot_filename = db.Column(db.String(300), nullable=True)

    # manual overrides for file parsing, when auto-detection guesses wrong
    parse_delimiter = db.Column(db.String(20), nullable=True)   # 'comma'/'semicolon'/'tab'/'whitespace', or None = auto
    parse_header_row = db.Column(db.Integer, nullable=True)     # 0-indexed line number, or None = auto

    # for images: nm-per-pixel scale, auto-extracted from instrument metadata when available
    pixel_size_nm = db.Column(db.Float, nullable=True)
    image_crop_bottom = db.Column(db.Integer, nullable=True)    # row where the instrument info bar starts
    preview_filename = db.Column(db.String(300), nullable=True) # browser-viewable PNG, for formats like TIFF that <img> can't render directly

    # AFM: which physical quantity this file represents (topography, a specialized-mode
    # channel map, or a force curve) and how much of it is real vs. just a picture
    channel_type = db.Column(db.String(50), nullable=True)      # 'topography'/'kpfm_potential'/'cafm_current'/'scm_capacitance'/'pfm_amplitude'/'pfm_phase'/'mfm_phase'/'lfm_friction'/'force_curve'
    data_units = db.Column(db.String(20), nullable=True)        # e.g. 'nm', 'V', 'nA', 'fF', 'deg'
    height_data_filename = db.Column(db.String(300), nullable=True)  # .npy of real parsed numeric data, if any
    parse_status = db.Column(db.String(20), nullable=True)      # 'parsed_native'/'image_only'/'unparsed_raw'
    imaging_notes = db.Column(db.Text, nullable=True)           # free-text context, e.g. liquid environment/buffer/temperature

    # AFM force curves (Mechanical / Biological SMFS): approach curve reuses x_column/y_column above
    spring_constant = db.Column(db.Float, nullable=True)        # cantilever spring constant, N/m
    y_column_retract = db.Column(db.String(100), nullable=True) # retract curve's force/deflection column
    adhesion_force = db.Column(db.Float, nullable=True)         # computed: min(retract force)

    # generated by the Simulate tab rather than uploaded — always badged as such in the UI
    is_simulated = db.Column(db.Boolean, nullable=False, default=False)
    simulation_key = db.Column(db.Text, nullable=True)          # JSON list of "what was simulated / what to expect" lines


class ImageAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    pixel_size_nm = db.Column(db.Float, nullable=True)
    sizes_nm_json = db.Column(db.Text, nullable=False)   # JSON list of measured diameters (nm, or px if no scale)
    unit = db.Column(db.String(10), nullable=False, default='nm')
    histogram_filename = db.Column(db.String(300), nullable=True)
    analysis_text = db.Column(db.Text, nullable=True)
    measurement_type = db.Column(db.String(20), nullable=False, default='particle_size')  # 'particle_size'/'step_height'/'layer_thickness'/'saed_spacing'/'lattice_fringe'/'strain_map'
    config_json = db.Column(db.Text, nullable=True)         # structured per-measurement data that doesn't fit the scalar columns above (SAED spot coords, GPA g-vectors/reference rect)
    extra_images_json = db.Column(db.Text, nullable=True)   # JSON list of {label, filename} — for measurements producing more than one result image (e.g. strain map's exx/eyy/exy)


class DefectAnnotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    x = db.Column(db.Float, nullable=False)   # pixel coordinates on the full-resolution image
    y = db.Column(db.Float, nullable=False)
    defect_type = db.Column(db.String(30), nullable=False)  # dislocation/stacking_fault/grain_boundary/vacancy/twin_boundary/other
    note = db.Column(db.Text, nullable=True)


class OverlayPlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    label = db.Column(db.String(200), nullable=True)
    config_json = db.Column(db.Text, nullable=False)   # JSON list of {file_id, x_column, y_column, fit_type}
    fit_results_json = db.Column(db.Text, nullable=True)   # JSON list of {label, fit_type, fit_params, r_squared}
    plot_filename = db.Column(db.String(300), nullable=True)


class AnalysisSnapshot(db.Model):
    """A frozen copy of a Data Interpretation plot and its analysis, captured exactly as it
    looked at save time — including its own copy of the plot image — so it stays reproducible
    even if the underlying data files are later edited, reparsed, or deleted. Mirrors Protocol
    versioning: a named, point-in-time record you can revisit or restore into a live, editable
    session again, rather than a live view that silently changes underneath you."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    title = db.Column(db.String(200), nullable=False)
    note = db.Column(db.Text, nullable=True)
    plot_type = db.Column(db.String(30), nullable=False)
    state_json = db.Column(db.Text, nullable=False)        # dp_state at capture time (file_ids, plot_type, derivative, format)
    results_json = db.Column(db.Text, nullable=False)      # per-series stats/analysis/shape at capture time
    overall_analysis = db.Column(db.Text, nullable=True)
    plot_filename = db.Column(db.String(300), nullable=True)


class MiscEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    entry_date = db.Column(db.Date, nullable=False, default=datetime.now)
    research_topic = db.Column(db.String(200), nullable=False)
    journal = db.Column(db.String(50), nullable=True)     # ACS/RSC/Wiley/Elsevier/IEEE/Springer/Nature/Other
    paper_title = db.Column(db.String(300), nullable=True)
    paper_link = db.Column(db.String(500), nullable=True)
    pdf_filename = db.Column(db.String(300), nullable=True)
    key_facts = db.Column(db.Text, nullable=True)


class Protocol(db.Model):
    """A reusable, versioned standard procedure or equipment SOP. Full step text lives on
    ProtocolVersion so editing a protocol keeps its history instead of overwriting it."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    equipment = db.Column(db.String(150), nullable=True)    # set for equipment-specific operating instructions
    category = db.Column(db.String(100), nullable=True)     # freeform grouping, e.g. "Sample prep", "Safety"
    current_version = db.Column(db.Integer, nullable=False, default=1)
    cloned_from_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=True)
    linked_experiment_id = db.Column(db.Integer, db.ForeignKey('log_entry.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    versions = db.relationship(
        'ProtocolVersion', backref='protocol',
        order_by='ProtocolVersion.version_number', cascade='all, delete-orphan',
    )
    cloned_from = db.relationship('Protocol', remote_side=[id])
    linked_experiment = db.relationship('LogEntry', foreign_keys=[linked_experiment_id])

    @property
    def latest_version(self):
        return self.versions[-1] if self.versions else None


class ProtocolVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    protocol_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=False)
    version_number = db.Column(db.Integer, nullable=False)
    steps = db.Column(db.Text, nullable=False)           # one step per line
    change_note = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


class Project(db.Model):
    """A funded project/grant — its timeline and milestones/deliverables live alongside it."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    grant_name = db.Column(db.String(200), nullable=True)
    funding_start = db.Column(db.Date, nullable=True)
    funding_end = db.Column(db.Date, nullable=True)
    funding_amount = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    milestones = db.relationship(
        'Milestone', backref='project',
        order_by='Milestone.due_date', cascade='all, delete-orphan',
    )
    tasks = db.relationship('Task', backref='project', cascade='all, delete-orphan')


class Milestone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    due_date = db.Column(db.Date, nullable=True)
    deliverable = db.Column(db.String(300), nullable=True)
    is_done = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


class Task(db.Model):
    """A to-do item, optionally scoped to a project and/or linked to a specific experiment."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    due_date = db.Column(db.Date, nullable=True)
    is_done = db.Column(db.Boolean, nullable=False, default=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=True)
    linked_experiment_id = db.Column(db.Integer, db.ForeignKey('log_entry.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    linked_experiment = db.relationship('LogEntry', foreign_keys=[linked_experiment_id])


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    category = db.Column(db.String(20), nullable=False, default='bug')   # 'bug' or 'feedback'
    message = db.Column(db.Text, nullable=False)
    email = db.Column(db.String(150), nullable=True)
    page_url = db.Column(db.String(300), nullable=True)


class Sample(db.Model):
    """A physical sample, as the one thing every technique's data about it has in
    common. Nothing else in the app ties AFM/TEM/XPS/etc. records together by what
    they're actually measurements *of* — this is what makes cross-technique
    correlation possible at all."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    properties = db.relationship(
        'SampleProperty', backref='sample',
        order_by='SampleProperty.technique_name', cascade='all, delete-orphan',
    )


class SampleProperty(db.Model):
    """One named scalar result for a sample from a given technique — e.g.
    (AFM, "Ra roughness", 2.4, "nm") or (XPS, "O:C ratio", 0.31, None). The
    (technique_name, property_name) pair is the axis label used when correlating
    two properties across every sample that has both."""
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.Integer, db.ForeignKey('sample.id'), nullable=False)
    technique_name = db.Column(db.String(100), nullable=False)
    property_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(30), nullable=True)
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


class LabGroup(db.Model):
    """A named set of people (a lab, a collaboration) an owner can share items with at once.
    Members are identified by email, not user id, so someone can be added before they've
    registered and simply gains access once they sign up with that email."""
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    members = db.relationship(
        'GroupMember', backref='group', order_by='GroupMember.added_at', cascade='all, delete-orphan',
    )


class GroupMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('lab_group.id'), nullable=False)
    email = db.Column(db.String(150), nullable=False)   # stored lowercase
    added_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


class Share(db.Model):
    """One grant of access to one item. Exactly one of grantee_email / group_id / link_token
    identifies who the grant is for: a specific email, a lab group, or anyone holding an
    unguessable link (read-only, no account needed)."""
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    item_type = db.Column(db.String(30), nullable=False)   # experiment/characterization/paper/protocol/sample/snapshot
    item_id = db.Column(db.Integer, nullable=False)
    grantee_email = db.Column(db.String(150), nullable=True)
    group_id = db.Column(db.Integer, db.ForeignKey('lab_group.id'), nullable=True)
    link_token = db.Column(db.String(64), unique=True, nullable=True)
    note = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    expires_at = db.Column(db.DateTime, nullable=True)


class ShareComment(db.Model):
    """A comment on a shared item, visible to the owner and everyone the item is shared with
    by email or group (never to guest-link viewers)."""
    id = db.Column(db.Integer, primary_key=True)
    item_type = db.Column(db.String(30), nullable=False)
    item_id = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


# Schema changes go through Flask-Migrate (Alembic): edit a model, then
#   flask --app app db migrate -m "what changed"
# review the generated file in migrations/versions/, commit it, and it's applied on the
# next start (ensure_schema below) — never by deleting the database.
#
# BASELINE_REVISION is the first migration, matching the schema as it stood when
# migrations were introduced. Databases created before then have tables but no
# alembic_version table; they get the old hand-rolled column patches one last time and
# are then stamped at the baseline, so Alembic takes over from exactly that point.
BASELINE_REVISION = 'c1a7b0d4e2f1'


def _legacy_add_missing_columns():
    """Pre-Alembic additive patches (SQLite only), kept solely to bring an old local
    database up to the baseline before it's stamped. Don't add to this — write a migration."""
    def _add_missing_columns(table, columns):
        existing = {row[1] for row in db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()}
        if not existing:
            # table doesn't exist (e.g. no model defines it any more) — nothing to migrate
            return
        for col_name, col_def in columns:
            if col_name not in existing:
                db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))
        db.session.commit()

    _add_missing_columns('data_file', [
        ('technique_name', 'VARCHAR(150)'),
        ('channel_type', 'VARCHAR(50)'),
        ('data_units', 'VARCHAR(20)'),
        ('height_data_filename', 'VARCHAR(300)'),
        ('parse_status', 'VARCHAR(20)'),
        ('imaging_notes', 'TEXT'),
        ('spring_constant', 'FLOAT'),
        ('y_column_retract', 'VARCHAR(100)'),
        ('adhesion_force', 'FLOAT'),
        ('user_id', 'INTEGER'),
        ('is_simulated', 'BOOLEAN NOT NULL DEFAULT 0'),
        ('simulation_key', 'TEXT'),
    ])
    _add_missing_columns('image_analysis', [
        ('measurement_type', "VARCHAR(20) NOT NULL DEFAULT 'particle_size'"),
        ('config_json', 'TEXT'),
        ('extra_images_json', 'TEXT'),
    ])
    # user_id: added so each person only sees and can act on their own data.
    # Nullable — existing rows predate accounts and simply won't match any
    # session's user_id, which is the intended (invisible-to-everyone) outcome.
    _add_missing_columns('log_entry', [('user_id', 'INTEGER')])
    _add_missing_columns('char_entry', [('user_id', 'INTEGER')])
    _add_missing_columns('overlay_plot', [('user_id', 'INTEGER')])
    _add_missing_columns('custom_plot', [('user_id', 'INTEGER')])
    _add_missing_columns('misc_entry', [('user_id', 'INTEGER')])
    _add_missing_columns('protocol', [('user_id', 'INTEGER')])
    _add_missing_columns('project', [('user_id', 'INTEGER')])
    _add_missing_columns('task', [('user_id', 'INTEGER')])


def ensure_schema():
    """Bring the database up to the latest migration. Called once at startup from the
    single web process (see __main__) — not at import time, so the compute worker and
    `flask db ...` commands don't race it."""
    from flask_migrate import upgrade, stamp
    with app.app_context():
        tables = set(sa_inspect(db.engine).get_table_names())
        if tables and 'alembic_version' not in tables:
            app.logger.info('Pre-migrations database found; patching and stamping at baseline.')
            db.create_all()                 # any tables added since this database was made
            if db.engine.dialect.name == 'sqlite':
                _legacy_add_missing_columns()
            stamp(revision=BASELINE_REVISION)
        upgrade()


# Every route requires a signed-in session except these — the pages you need
# before you can have one, plus Flask's own static file server and the favicon
# every browser requests automatically regardless of what page is loaded.
PUBLIC_ENDPOINTS = {'index', 'register', 'login', 'static', 'favicon', 'forgot_password', 'reset_password',
                    'shared_link_view'}


PRODUCTION_MODE = os.environ.get('PRODUCTION', '0').lower() in ('1', 'true', 'yes')

# Local-development convenience only: DEV_AUTOLOGIN=<email> signs every browser in as that
# user (created on first use) so the developer can skip the sign-in screen. Hard-disabled
# whenever PRODUCTION is set, since it bypasses authentication entirely.
DEV_AUTOLOGIN_EMAIL = os.environ.get('DEV_AUTOLOGIN', '').strip().lower() or None
if DEV_AUTOLOGIN_EMAIL and PRODUCTION_MODE:
    DEV_AUTOLOGIN_EMAIL = None
if DEV_AUTOLOGIN_EMAIL:
    app.logger.warning(f'DEV_AUTOLOGIN is ON — every visitor is signed in as {DEV_AUTOLOGIN_EMAIL}. Local development only.')


@app.before_request
def require_login():
    if DEV_AUTOLOGIN_EMAIL and 'user_id' not in session and request.endpoint not in (None, 'static'):
        dev_user = User.query.filter_by(email=DEV_AUTOLOGIN_EMAIL).first()
        if not dev_user:
            dev_user = User(
                name='Dev User', email=DEV_AUTOLOGIN_EMAIL,
                password=generate_password_hash(secrets.token_urlsafe(32)),
            )
            db.session.add(dev_user)
            try:
                db.session.commit()
            except Exception:
                # another concurrent request created it first
                db.session.rollback()
                dev_user = User.query.filter_by(email=DEV_AUTOLOGIN_EMAIL).first()
        session['user_id'] = dev_user.id
        session['user_name'] = dev_user.name
        session['user_email'] = dev_user.email

    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return
    if 'user_id' not in session:
        return redirect(url_for('index'))


def get_owned_or_404(model, obj_id, owner_field='user_id'):
    """Fetch a row the signed-in user actually owns — 404s (not 403) on
    someone else's id, so a guessed/enumerated id can't even confirm it exists."""
    obj = model.query.get_or_404(obj_id)
    if getattr(obj, owner_field) != session.get('user_id'):
        abort(404)
    return obj


@app.errorhandler(413)
def too_large(e):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return render_template(
        'error.html',
        title='Upload too large',
        message=f"That upload is over the {max_mb} MB limit for a single request. Try uploading fewer files at once, or smaller ones.",
    ), 413


@app.errorhandler(429)
def too_many_requests(e):
    # Compute forms redirect back to their tab with an inline error, like any other
    # compute failure, instead of dropping the user on a bare error page.
    if request.endpoint and request.endpoint.startswith('comp_run_') and request.view_args:
        session['comp_error'] = f'Too many calculations in a short time ({e.description}). Please wait a bit and try again.'
        return redirect(url_for('technique_workspace', slug=request.view_args.get('slug'),
                                tab=COMP_ENDPOINT_TABS.get(request.endpoint)))
    return render_template(
        'error.html',
        title='Too many attempts',
        message=f"You've made too many requests in a short time ({e.description}). Please wait a few minutes and try again.",
    ), 429


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='images/lablogbook-logo1.svg'))


@app.route('/')
def index():
    return render_template('index.html', banner_image='images/homepage-banner.png')


def validate_password(password):
    """Sign-up password rules — kept in step with the checklist in static/js/main.js.
    Applies to new accounts only; existing passwords are never re-checked at sign-in."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r'[A-Za-z]', password):
        return "Password must include at least one letter."
    if not re.search(r'[0-9]', password):
        return "Password must include at least one number."
    return None


@app.route('/register', methods=['GET', 'POST'])
@limiter.limit('10 per hour', methods=['POST'], key_func=get_remote_address)
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="An account with this email already exists.")

        password_error = validate_password(password)
        if password_error:
            return render_template('register.html', error=password_error)

        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit('10 per minute;50 per hour', methods=['POST'], key_func=get_remote_address)
@limiter.limit('20 per hour', methods=['POST'], key_func=_login_email_key)
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            return redirect(url_for('profile'))

        return render_template('index.html', error="Invalid email or password.", banner_image='images/homepage-banner.png')

    return render_template('index.html', banner_image='images/homepage-banner.png')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('user_email', None)
    return redirect(url_for('index'))


# ---- Password reset -------------------------------------------------------------------
# Stateless signed tokens (no schema change): each token embeds the user id plus a
# fingerprint of their *current* password hash, so it stops working the moment the password
# changes — a reset link is single-use — and it also expires after an hour.

PASSWORD_RESET_MAX_AGE = 3600
_reset_serializer = URLSafeTimedSerializer(app.secret_key, salt='password-reset')


def _password_fingerprint(user):
    return hashlib.sha256(user.password.encode('utf-8')).hexdigest()[:16]


def make_reset_token(user):
    return _reset_serializer.dumps({'uid': user.id, 'fp': _password_fingerprint(user)})


def user_from_reset_token(token):
    try:
        data = _reset_serializer.loads(token, max_age=PASSWORD_RESET_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    user = db.session.get(User, data.get('uid'))
    if user and data.get('fp') == _password_fingerprint(user):
        return user
    return None


def send_email(to_addr, subject, body):
    """Sends a plain-text email via SMTP when SMTP_HOST is configured (SMTP_PORT, SMTP_USER,
    SMTP_PASSWORD, MAIL_FROM). Returns True only if it was actually sent. With no SMTP
    configured the message is written to the server log instead — fine for local development."""
    host = os.environ.get('SMTP_HOST')
    if not host:
        app.logger.warning(f'SMTP not configured — email to {to_addr} ({subject}):\n{body}')
        return False
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = os.environ.get('MAIL_FROM') or os.environ.get('SMTP_USER') or 'no-reply@lablogbook.local'
    msg['To'] = to_addr
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get('SMTP_PORT', 587)), timeout=10) as smtp:
            smtp.starttls()
            if os.environ.get('SMTP_USER'):
                smtp.login(os.environ['SMTP_USER'], os.environ.get('SMTP_PASSWORD', ''))
            smtp.send_message(msg)
        return True
    except Exception:
        app.logger.exception(f'Sending email to {to_addr} failed')
        return False


def send_reset_email(user, link):
    return send_email(
        user.email, 'Reset your LabLogbook password',
        f"Hi {user.name},\n\nUse this link to choose a new LabLogbook password "
        f"(valid for {PASSWORD_RESET_MAX_AGE // 60} minutes, one use):\n\n{link}\n\n"
        "If you didn't ask for this, you can ignore this email — your password hasn't changed."
    )


@app.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit('5 per hour', methods=['POST'], key_func=get_remote_address)
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        user = User.query.filter_by(email=email).first()
        dev_link = None
        if user:
            link = url_for('reset_password', token=make_reset_token(user), _external=True)
            sent = send_reset_email(user, link)
            # Only ever shown on-page for local development (no SMTP, dev mode on) — showing it
            # in production would let anyone reset any account just by knowing its email.
            if not sent and not PRODUCTION_MODE and (DEBUG_MODE or DEV_AUTOLOGIN_EMAIL):
                dev_link = link
        # Same response whether or not the email is registered, so this page can't be used
        # to discover who has an account.
        return render_template('forgot_password.html', submitted=True, dev_link=dev_link)
    return render_template('forgot_password.html', submitted=False)


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@limiter.limit('10 per hour', methods=['POST'], key_func=get_remote_address)
def reset_password(token):
    user = user_from_reset_token(token)
    if not user:
        return render_template('reset_password.html', invalid=True), 400
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password != request.form.get('confirm_password', ''):
            return render_template('reset_password.html', error="The two passwords don't match.")
        password_error = validate_password(password)
        if password_error:
            return render_template('reset_password.html', error=password_error)
        user.password = generate_password_hash(password)
        db.session.commit()
        return render_template('reset_password.html', done=True)
    return render_template('reset_password.html')


STATUS_OPTIONS = ["In progress", "Completed", "Failed", "Repeat needed"]
EXP_TYPE_OPTIONS = ["Chemical", "Physical", "Biological", "Characterization", "Other"]

SAMPLE_PREP_OPTIONS = ["Powder", "Film", "Liquid", "Frozen", "Other"]
OUTCOME_OPTIONS = ["Repeat needed", "Good data", "Synthesize again"]

JOURNAL_OPTIONS = ["ACS", "RSC", "Wiley", "Elsevier", "IEEE", "Springer", "Nature", "Other"]


PROFILE_CATEGORIES = {
    'Experiments': '#7ecaf6',
    'Characterizations': '#7bd0c1',
    'Data Interpretation': '#ae85ca',
    'Research Papers': '#f2849e',
}


@app.route('/profile')
def profile():
    uid = session['user_id']
    experiments_entries = LogEntry.query.filter_by(category='experiments', user_id=uid).order_by(LogEntry.entry_datetime.desc()).all()
    char_entries = CharEntry.query.filter_by(user_id=uid).order_by(CharEntry.date_scheduled.desc()).all()
    data_files = DataFile.query.filter_by(user_id=uid).order_by(DataFile.uploaded_at.desc()).all()
    paper_entries = MiscEntry.query.filter_by(user_id=uid).order_by(MiscEntry.entry_date.desc()).all()

    # a single combined feed, newest first, so the dashboard can show "recent activity"
    # across all four log types without a separate tab per type. Entries that belong to a
    # specific characterization technique (a scheduled run, or an uploaded data file) show
    # that technique as their category chip, rather than the generic bucket name.
    activity = []
    for e in experiments_entries:
        activity.append({'date': e.entry_datetime, 'title': e.title, 'category': 'Experiments', 'tag': e.status, 'link': url_for('experiments')})
    for e in char_entries:
        title_bits = []
        if e.num_samples:
            title_bits.append(f"{e.num_samples} sample{'s' if e.num_samples != 1 else ''}")
        if e.sample_prep:
            title_bits.append(e.sample_prep)
        activity.append({
            'date': datetime.combine(e.date_scheduled, datetime.min.time()),
            'title': ' · '.join(title_bits) or 'Characterization run',
            'category': e.technique_name,
            'tag': e.outcome,
            'link': url_for('characterizations'),
        })
    for f in data_files:
        activity.append({
            'date': f.uploaded_at,
            'title': f.original_filename,
            'category': f.technique_name or 'Data Interpretation',
            'tag': f.file_type.capitalize() if f.file_type else None,
            'link': url_for('view_data_file', file_id=f.id),
        })
    for p in paper_entries:
        activity.append({'date': datetime.combine(p.entry_date, datetime.min.time()), 'title': p.paper_title or p.research_topic, 'category': 'Research Papers', 'tag': p.journal, 'link': url_for('miscellaneous')})
    activity.sort(key=lambda a: a['date'], reverse=True)

    category_counts = {
        'Experiments': len(experiments_entries),
        'Characterizations': len(char_entries),
        'Data Interpretation': len(data_files),
        'Research Papers': len(paper_entries),
    }
    total_entries = sum(category_counts.values())
    max_count = max(category_counts.values()) if total_entries else 0
    category_breakdown = [
        {'name': name, 'count': count, 'color': PROFILE_CATEGORIES[name], 'pct': round(count / max_count * 100) if max_count else 0}
        for name, count in sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    top_category = category_breakdown[0]['name'] if total_entries else '—'

    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    entries_this_month = sum(1 for a in activity if a['date'] >= month_start)

    earliest_date = min((a['date'] for a in activity), default=None)

    user_name = session.get('user_name', 'Demo User')
    initials = ''.join(part[0].upper() for part in user_name.split()[:2]) or 'U'

    return render_template(
        'profile.html',
        page_title='My Profile',
        user_name=user_name,
        user_initials=initials,
        user_email=session.get('user_email', 'demo@lablogbook.com'),
        member_since=earliest_date.strftime('%d %b %Y') if earliest_date else None,
        total_entries=total_entries,
        entries_this_month=entries_this_month,
        top_category=top_category,
        recent_activity=activity[:8],
        category_breakdown=category_breakdown,
        banner_image='images/header-banner.png',
    )


def fill_entry_from_form(entry):
    entry.title = request.form['title']
    entry_datetime_str = request.form.get('entry_datetime')
    entry.entry_datetime = (
        datetime.fromisoformat(entry_datetime_str) if entry_datetime_str else datetime.now()
    )
    entry.exp_type = request.form.get('exp_type', '')
    entry.status = request.form.get('status', '')
    entry.objective = request.form.get('objective', '')
    entry.materials = request.form.get('materials', '')
    entry.procedure = request.form.get('procedure', '')
    entry.input_params = request.form.get('input_params', '')
    entry.output_results = request.form.get('output_results', '')
    entry.observations = request.form.get('observations', '')
    entry.conclusion = request.form.get('conclusion', '')


def build_inference(entry_a, entry_b):
    """Very simple rule-based comparison notes between two entries."""
    notes = []

    if entry_a.status and entry_b.status:
        if entry_a.status != entry_b.status:
            notes.append(f"Status changed from \"{entry_a.status}\" to \"{entry_b.status}\".")
        else:
            notes.append(f"Status stayed the same (\"{entry_a.status}\").")

    if entry_a.input_params and entry_b.input_params:
        if entry_a.input_params.strip() == entry_b.input_params.strip():
            notes.append("Input parameters were unchanged between the two runs.")
        else:
            notes.append("Input parameters were changed — review whether that explains any difference in results.")

    if entry_a.output_results and entry_b.output_results:
        if entry_a.output_results.strip() == entry_b.output_results.strip():
            notes.append("Output/results came out identical.")
        else:
            notes.append("Output/results differ between the two entries — worth a closer look at what changed.")

    if not notes:
        notes.append("Not enough data in both entries to compare yet — fill in status, input parameters, or output/results.")

    return notes


def log_page(category, page_title):
    """Shared handler for experiments/characterizations/miscellaneous pages."""
    if request.method == 'POST':
        new_entry = LogEntry(category=category, user_id=session['user_id'])
        fill_entry_from_form(new_entry)
        db.session.add(new_entry)
        db.session.commit()
        return redirect(url_for(category))

    entries = LogEntry.query.filter_by(category=category, user_id=session['user_id']).order_by(LogEntry.entry_datetime.desc()).all()

    # Which two entries to compare — default to the two most recent, unless the user picked specific ones
    compare_a_id = request.args.get('compare_a', type=int)
    compare_b_id = request.args.get('compare_b', type=int)

    compare_a = None
    compare_b = None
    inference = []

    if len(entries) >= 2:
        compare_a = next((e for e in entries if e.id == compare_a_id), entries[0])
        compare_b = next((e for e in entries if e.id == compare_b_id), entries[1])
        if compare_a.id != compare_b.id:
            inference = build_inference(compare_a, compare_b)

    return render_template(
        'log.html',
        page_title=page_title,
        category=category,
        entries=entries,
        compare_a=compare_a,
        compare_b=compare_b,
        inference=inference,
        now=datetime.now().strftime('%Y-%m-%dT%H:%M'),
        status_options=STATUS_OPTIONS,
        exp_type_options=EXP_TYPE_OPTIONS,
    )


@app.route('/experiments', methods=['GET', 'POST'])
def experiments():
    return log_page('experiments', 'Experiments')


TABULAR_EXTENSIONS = {'.csv', '.txt', '.xlsx', '.xls', '.tsv', '.dat', '.asc', '.prn', '.out', '.mpt', '.cor', '.tab'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
KNOWN_NON_DATA_EXTENSIONS = {'.exe', '.zip', '.rar', '.7z', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.mp3', '.mp4', '.avi', '.mov', '.dll', '.iso'}


def extract_pixel_size_from_tiff(filepath):
    """Many SEM instruments (e.g. Hitachi SU-series) embed calibration metadata as
    UTF-16 text within the TIFF file itself. Look for a 'PixelSize=<value>' entry,
    which gives nm-per-pixel directly from the instrument — far more reliable than
    trying to detect a scale bar visually."""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        for encoding in ('utf-16-le', 'latin-1'):
            try:
                text = raw.decode(encoding, errors='ignore')
            except Exception:
                continue
            m = re.search(r'PixelSize\s*=\s*([\d.]+)', text)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


def detect_image_info_bar(gray_array):
    """Detects an instrument info bar/banner at the bottom of an SEM/TEM image by finding
    the largest sudden brightness drop in the bottom quarter of the image. Returns the row
    index where the real image content ends (safe crop boundary), or full height if none found."""
    h = gray_array.shape[0]
    row_means = gray_array.mean(axis=1)
    search_start = int(h * 0.75)
    if search_start >= h - 2:
        return h
    diffs = np.diff(row_means[search_start:])
    if len(diffs) == 0:
        return h
    worst_idx = int(np.argmin(diffs))
    drop_size = diffs[worst_idx]
    if drop_size < -1.6 * np.std(row_means):
        return search_start + worst_idx + 1
    return h


def compute_intensity_roughness(gray_array, crop_bottom=None):
    """A grayscale-intensity-based roughness PROXY for AFM images — NOT a substitute for
    true Ra/Rq computed from real height-map data (.ibw, .gwy, etc.), which this tool doesn't
    parse. Useful only as a rough, comparative texture metric between images of the same type."""
    region = gray_array[:crop_bottom, :] if crop_bottom else gray_array
    region = region.astype(float)
    mean_intensity = float(np.mean(region))
    ra_proxy = float(np.mean(np.abs(region - mean_intensity)))     # mean absolute deviation, like Ra
    rq_proxy = float(np.sqrt(np.mean((region - mean_intensity) ** 2)))  # RMS deviation, like Rq
    return {'mean_intensity': mean_intensity, 'ra_proxy': ra_proxy, 'rq_proxy': rq_proxy}


def compute_porosity(gray_array, crop_bottom=None, pores_are_dark=True):
    """Estimates pore/void area fraction via Otsu thresholding — genuinely computable from a
    plain SEM image, unlike composition/crystallography. Returns stats plus an overlay image
    (pores highlighted) so you can visually verify the segmentation rather than trust a number blindly."""
    region = gray_array[:crop_bottom, :] if crop_bottom else gray_array
    smoothed = gaussian(region.astype(float), sigma=1.0)
    thresh = threshold_otsu(smoothed)

    pore_mask = smoothed < thresh if pores_are_dark else smoothed > thresh
    pore_mask = remove_small_objects(pore_mask, min_size=20)
    pore_mask = remove_small_holes(pore_mask, area_threshold=20)

    total_px = pore_mask.size
    pore_px = int(np.sum(pore_mask))
    pore_fraction = pore_px / total_px if total_px else 0.0

    # simple crack indicator: thin, elongated dark regions have high perimeter-to-area ratio
    labeled = label(pore_mask)
    regions = regionprops(labeled)
    elongated_count = sum(1 for r in regions if r.area >= 15 and r.perimeter > 0 and (r.perimeter ** 2) / (4 * np.pi * r.area) > 4)

    return {
        'pore_fraction': pore_fraction,
        'pore_count': len(regions),
        'elongated_count': elongated_count,
        'mask': pore_mask,
        'region': region,
    }


def generate_porosity_overlay(data_file):
    """Runs porosity detection on a stored image file and saves a visual overlay showing
    exactly what was classified as pore/void, so results can be sanity-checked visually."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    gray = np.array(Image.open(filepath).convert('L'))
    crop = data_file.image_crop_bottom or gray.shape[0]

    stats = compute_porosity(gray, crop)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(stats['region'], cmap='gray')
    overlay = np.zeros((*stats['mask'].shape, 4))
    overlay[stats['mask']] = [1, 0.2, 0.2, 0.45]  # translucent red over detected pores/voids
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(f"Pore/void area: {stats['pore_fraction']*100:.1f}% · {stats['pore_count']} region(s) detected")
    fig.tight_layout()

    overlay_filename = f"porosity_{data_file.id}_{int(datetime.now().timestamp())}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename), dpi=130)
    plt.close(fig)

    return {
        'pore_fraction': stats['pore_fraction'],
        'pore_count': stats['pore_count'],
        'elongated_count': stats['elongated_count'],
        'overlay_filename': overlay_filename,
    }


# AFM specialized-mode channel maps (Electrical/Magnetic/Chemical-Frictional tabs) and,
# when a native file was parsed, real topography — all share this same compute/render
# pair. channel_type only affects the colormap and display label; the stats themselves
# are always genuine when computed at all (only ever run on real height_data_filename
# arrays — an image-only file never gets fabricated numbers, see generate_channel_overlay).
CHANNEL_TYPE_INFO = {
    'topography': {'label': 'Topography (height)', 'cmap': 'afmhot'},
    'kpfm_potential': {'label': 'Surface potential (KPFM)', 'cmap': 'RdBu_r'},
    'cafm_current': {'label': 'Current (Conductive AFM)', 'cmap': 'inferno'},
    'scm_capacitance': {'label': 'Capacitance (SCM)', 'cmap': 'viridis'},
    'pfm_amplitude': {'label': 'Piezoresponse amplitude (PFM)', 'cmap': 'viridis'},
    'pfm_phase': {'label': 'Piezoresponse phase (PFM)', 'cmap': 'RdBu_r'},
    'mfm_phase': {'label': 'Magnetic phase shift (MFM)', 'cmap': 'RdBu_r'},
    'lfm_friction': {'label': 'Lateral force / friction (LFM)', 'cmap': 'inferno'},
    'se': {'label': 'Secondary electron (SE)', 'cmap': 'gray'},
    'bse': {'label': 'Backscattered electron (BSE)', 'cmap': 'gray'},
}


def compute_channel_stats(array, channel_type=None, units=None):
    """Genuine mean/std/min/max/range of a real parsed data channel. Only ever call this
    on an actual numeric array pulled from a native file (height_data_filename) — never
    on a grayscale rendering of a plain photo, which has no calibrated physical meaning."""
    arr = np.asarray(array, dtype=float)
    stats = {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
        'units': units or '',
        'label': CHANNEL_TYPE_INFO.get(channel_type, {}).get('label', channel_type or 'Channel'),
    }
    if channel_type == 'mfm_phase':
        # rough proxy for magnetic domain-wall density: fraction of pixels that fall on a
        # detected edge in the phase map. Not a substitute for proper domain segmentation —
        # useful only as a relative, comparative indicator between images of the same sample.
        norm = (arr - arr.min()) / (arr.ptp() or 1)
        edges = canny(norm, sigma=1.5)
        stats['domain_boundary_density_proxy'] = float(np.mean(edges))
    return stats


def generate_channel_overlay(data_file):
    """Renders whatever data exists for this file: a real colored channel map with genuine
    stats when height_data_filename holds parsed numeric data, or just the plain image with
    no stats (parse_status != 'parsed_native') — the caller must not treat a missing 'stats'
    key as zero/unknown, it means no calibrated data was available, full stop."""
    channel_type = data_file.channel_type or 'topography'
    info = CHANNEL_TYPE_INFO.get(channel_type, {'label': channel_type, 'cmap': 'viridis'})

    if data_file.parse_status == 'parsed_native' and data_file.height_data_filename:
        arr = np.load(os.path.join(app.config['UPLOAD_FOLDER'], data_file.height_data_filename))
        stats = compute_channel_stats(arr, channel_type, data_file.data_units)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(arr, cmap=info['cmap'])
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(f"{info['label']}" + (f" ({data_file.data_units})" if data_file.data_units else ''))
        ax.axis('off')
        ax.set_title(info['label'])
        fig.tight_layout()

        overlay_filename = f"channel_{data_file.id}_{int(datetime.now().timestamp())}.png"
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename), dpi=130)
        plt.close(fig)
        return {'stats': stats, 'overlay_filename': overlay_filename, 'label': info['label']}

    # no real numeric data for this channel — show the picture, report no stats rather
    # than guessing something from pixel brightness for a channel type we can't proxy
    return {'stats': None, 'overlay_filename': None, 'label': info['label']}


# TEM: crystallographic analysis (SAED d-spacings, HRTEM lattice-fringe spacing, and
# Hytch geometric phase analysis for strain mapping). All three read genuine spot
# positions/phase information out of a real 2D FFT — nothing here is estimated from
# pixel brightness the way the SEM/AFM intensity-based proxies are.

def compute_fft_power_spectrum(gray_array):
    """Real 2D FFT of an image. Returns (log-magnitude array for display,
    fft-shifted complex array for later masked-region extraction)."""
    window = np.outer(np.hanning(gray_array.shape[0]), np.hanning(gray_array.shape[1]))
    f = np.fft.fft2(gray_array.astype(float) * window)
    f_shifted = np.fft.fftshift(f)
    log_mag = np.log1p(np.abs(f_shifted))
    return log_mag, f_shifted


def fft_spot_to_spacing(spot_x, spot_y, img_w, img_h, pixel_size_nm):
    """Converts a clicked FFT spot's absolute pixel position to a real-space lattice
    spacing (nm), accounting for non-square images/crops correctly (x and y frequency
    axes are scaled independently by their own dimension before combining)."""
    cx, cy = img_w / 2.0, img_h / 2.0
    freq_x = (spot_x - cx) / (img_w * pixel_size_nm)
    freq_y = (spot_y - cy) / (img_h * pixel_size_nm)
    freq = np.hypot(freq_x, freq_y)
    if freq == 0:
        return None
    return 1.0 / freq


def _gaussian_aperture(shape, center_xy, sigma):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-(((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2) / (2 * sigma ** 2)))


def compute_gpa_strain_map(gray_array, g1_xy, g2_xy, ref_rect):
    """Hytch geometric phase analysis. g1_xy/g2_xy: absolute (x, y) pixel coordinates of
    two non-collinear reciprocal-lattice spots in this image's own FFT. ref_rect: (x0,
    y0, x1, y1) real-space pixel box assumed defect-free/unstrained — every strain value
    is reported relative to this region, so a badly-chosen reference silently shifts
    everything. Strain components are dimensionless by construction (displacement
    gradient), so no physical calibration is needed here. Raises ValueError if g1/g2 are
    degenerate (collinear) or the reference box is empty."""
    h, w = gray_array.shape
    x0, y0, x1, y1 = [int(round(v)) for v in ref_rect]
    x0, x1 = sorted((max(0, min(x0, w - 1)), max(0, min(x1, w - 1))))
    y0, y1 = sorted((max(0, min(y0, h - 1)), max(0, min(y1, h - 1))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Reference region is empty — pick two distinct corners on the image.")

    log_mag, f_shifted = compute_fft_power_spectrum(gray_array)
    sigma = max(min(h, w) * 0.02, 3)
    center = np.array([w / 2.0, h / 2.0])
    yy, xx = np.mgrid[0:h, 0:w]

    def geometric_phase(g_xy):
        # physical spatial frequency of this g-spot, in cycles/pixel along each axis —
        # must match units with the ramp below (bin-offset alone is dimensionally wrong
        # once combined with a second, possibly differently-scaled, g-vector)
        g_vec = (np.array(g_xy, dtype=float) - center) / np.array([w, h])
        mask = _gaussian_aperture((h, w), g_xy, sigma)
        complex_img = np.fft.ifft2(np.fft.ifftshift(f_shifted * mask))
        raw_phase = np.angle(complex_img)
        ramp = 2 * np.pi * (g_vec[0] * xx + g_vec[1] * yy)
        p = np.angle(np.exp(1j * (raw_phase - ramp)))
        ref_mean = np.angle(np.mean(np.exp(1j * p[y0:y1, x0:x1])))
        p = np.angle(np.exp(1j * (p - ref_mean)))
        return unwrap_phase(p), g_vec

    p1, g1_vec = geometric_phase(g1_xy)
    p2, g2_vec = geometric_phase(g2_xy)

    a_matrix = np.array([[g1_vec[0], g1_vec[1]], [g2_vec[0], g2_vec[1]]])
    if abs(np.linalg.det(a_matrix)) < 1e-9:
        raise ValueError("The two g-vectors are collinear — pick spots from two different lattice directions.")
    a_inv = np.linalg.inv(a_matrix)

    dp1_dy, dp1_dx = np.gradient(p1)
    dp2_dy, dp2_dx = np.gradient(p2)

    dudx = (a_inv[0, 0] * dp1_dx + a_inv[0, 1] * dp2_dx) / (2 * np.pi)
    dudy = (a_inv[0, 0] * dp1_dy + a_inv[0, 1] * dp2_dy) / (2 * np.pi)
    dvdx = (a_inv[1, 0] * dp1_dx + a_inv[1, 1] * dp2_dx) / (2 * np.pi)
    dvdy = (a_inv[1, 0] * dp1_dy + a_inv[1, 1] * dp2_dy) / (2 * np.pi)

    exx, eyy, exy = dudx, dvdy, 0.5 * (dudy + dvdx)

    # phase near the image border is unreliable (the Hann window drives the signal to
    # ~0 there, and np.gradient falls back to one-sided differences) — this is a known
    # GPA edge effect, not a real strain signal, so mask it out rather than let a border
    # artifact dominate the displayed range or the reported max
    margin = max(int(round(3 * min(h, w) / (2 * np.pi * sigma))), 8)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin, :] = border_mask[-margin:, :] = True
    border_mask[:, :margin] = border_mask[:, -margin:] = True
    for arr in (exx, eyy, exy):
        arr[border_mask] = np.nan

    return {
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'ref_rect': (x0, y0, x1, y1),
        'margin': margin,
        'stats': {
            label: {
                'mean': float(np.nanmean(arr)), 'max': float(np.nanmax(np.abs(arr))),
                'ref_residual': float(np.nanmean(arr[y0:y1, x0:x1])),
            }
            for label, arr in (('exx', exx), ('eyy', eyy), ('exy', exy))
        },
    }


def generate_strain_map_overlay(data_file, gray_array, g1_xy, g2_xy, ref_rect):
    result = compute_gpa_strain_map(gray_array, g1_xy, g2_xy, ref_rect)
    images = []
    vmax = max(0.01, max(result['stats'][k]['max'] for k in ('exx', 'eyy', 'exy')))
    for label, key in (('εxx', 'exx'), ('εyy', 'eyy'), ('εxy', 'exy')):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(result[key], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.85, label='strain')
        rect = plt.Rectangle((result['ref_rect'][0], result['ref_rect'][1]),
                              result['ref_rect'][2] - result['ref_rect'][0],
                              result['ref_rect'][3] - result['ref_rect'][1],
                              fill=False, edgecolor='black', linewidth=1.2, linestyle='--')
        ax.add_patch(rect)
        ax.axis('off')
        ax.set_title(f"{label} (dashed box = reference region, blank border = unreliable edge)")
        fig.tight_layout()
        filename = f"strain_{key}_{data_file.id}_{int(datetime.now().timestamp())}.png"
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], filename), dpi=130)
        plt.close(fig)
        images.append({'label': label, 'filename': filename})
    return images, result['stats']


def linear_fn(x, m, c):
    return m * x + c


def quadratic_fn(x, a, b, c):
    return a * x**2 + b * x + c


def exponential_fn(x, a, b, c):
    return a * np.exp(b * x) + c


def gaussian_fn(x, a, mu, sigma, offset):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


FIT_FUNCTIONS = {
    'linear': (linear_fn, ['m (slope)', 'c (intercept)']),
    'quadratic': (quadratic_fn, ['a', 'b', 'c']),
    'exponential': (exponential_fn, ['a', 'b', 'c']),
    'gaussian': (gaussian_fn, ['amplitude', 'mean', 'sigma', 'offset']),
}


def power_fn(x, a, b):
    return a * np.power(x, b)


def logarithmic_fn(x, a, b):
    return a * np.log(x) + b


def langmuir_fn(x, a, b):
    return (a * x) / (b + x)


def sigmoid_fn(x, amplitude, k, x0, offset):
    return amplitude / (1 + np.exp(-k * (x - x0))) + offset


def suggest_curve_shape(x, y):
    """Quietly tries a handful of canonical curve shapes against a raw X/Y series and
    returns the best-fitting one, if any fits convincingly (R^2 >= 0.9) — a shape hint
    the user never asked for and would otherwise only get by manually trying fit types
    one at a time. Deliberately separate from FIT_FUNCTIONS (the explicit, user-selected
    fit used elsewhere) so this stays a read-only suggestion, never changing what those
    already-tested fit workflows do."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 6:
        return None

    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())
    if x_range == 0 or y_range == 0:
        return None

    candidates = []

    def try_fit(shape, func, p0, friendly, needs_positive_x=False):
        if needs_positive_x and np.any(x <= 0):
            return
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=6000)
            y_pred = func(x, *popt)
            if not np.all(np.isfinite(y_pred)):
                return
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
            if r_squared is not None and np.isfinite(r_squared):
                # BIC (lower is better) charges for extra parameters, so a nested model such as a
                # quadratic can't win over a line just by having one more knob to turn.
                floor = 1e-12 * ss_tot
                bic = len(x) * np.log(max(ss_res, floor) / len(x)) + len(popt) * np.log(len(x))
                candidates.append({'shape': shape, 'friendly': friendly, 'r_squared': float(r_squared), 'bic': float(bic)})
        except (RuntimeError, ValueError, TypeError, OverflowError):
            return

    slope_guess = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
    try_fit('linear', linear_fn, [slope_guess, y[0]], "a linear relationship")
    try_fit('quadratic', quadratic_fn, [0.0, slope_guess, y[0]], "a quadratic (parabolic) relationship")
    try_fit('exponential', exponential_fn, [y_range or 1.0, 1.0 / x_range, y.min()], "exponential growth or decay")
    try_fit('power', power_fn, [1.0, 1.0], "a power-law relationship (y ∝ x^b)", needs_positive_x=True)
    try_fit('logarithmic', logarithmic_fn, [y_range or 1.0, y.min()], "a logarithmic relationship", needs_positive_x=True)
    try_fit('langmuir', langmuir_fn, [y.max(), x_range / 2 or 1.0],
            "a Langmuir-type saturation curve — common for adsorption, binding, or surface-coverage data leveling off toward a plateau")
    try_fit('sigmoid', sigmoid_fn, [y_range, 4.0 / x_range, np.median(x), y.min()],
            "a sigmoidal (S-shaped) curve — common for dose-response, growth, or phase-transition data")

    if not candidates:
        return None

    good = [c for c in candidates if c['r_squared'] >= 0.9]
    if not good:
        return None
    return min(good, key=lambda c: c['bic'])


# AFM force-distance curve models (Mechanical tab: Hertz/DMT contact mechanics; Biological
# tab: worm-like-chain for single-molecule force spectroscopy). Separate from FIT_FUNCTIONS
# above because these need physical constants (tip radius) baked in via closures, not just
# x — curve_fit itself works the same way underneath, same as generate_plot_and_fit.

def _make_hertz_fn(tip_radius_nm):
    R = tip_radius_nm
    def hertz_fn(x, E_star, x0):
        delta = np.maximum(x0 - x, 0.0)  # indentation depth; zero (no contact) beyond x0
        return (4.0 / 3.0) * E_star * np.sqrt(R) * delta ** 1.5
    return hertz_fn


def _make_dmt_fn(tip_radius_nm):
    R = tip_radius_nm
    def dmt_fn(x, E_star, x0, F_adhesion):
        delta = np.maximum(x0 - x, 0.0)
        return (4.0 / 3.0) * E_star * np.sqrt(R) * delta ** 1.5 - F_adhesion
    return dmt_fn


def wlc_fn(x, Lp, Lc):
    """Marko-Siggia interpolation formula for the worm-like chain model, at room temperature
    (kT ≈ 4.1 pN·nm). Lp = persistence length, Lc = contour length, both in nm; returns force
    in pN. Standard model for polymer/protein/DNA unfolding force-extension curves."""
    kT = 4.1
    ratio = np.clip(x / Lc, 0, 0.999)
    return (kT / Lp) * (0.25 / (1 - ratio) ** 2 - 0.25 + ratio)


def fit_force_curve(x, y, fit_type, tip_radius_nm=None):
    """Fits a force-distance/extension curve to a genuine contact-mechanics or polymer model.
    Returns (fit_params [(label, value), ...], r_squared, fitted_func) or (None, None, None)
    if the fit didn't converge — never returns a plausible-looking but meaningless fit."""
    try:
        if fit_type in ('hertz', 'dmt'):
            if not tip_radius_nm or tip_radius_nm <= 0:
                return None, None, None
            contact_guess = x[np.argmax(y)]
            e_guess = max(np.ptp(y), 1.0) / (max(np.ptp(x), 1.0) ** 1.5)
            if fit_type == 'hertz':
                func = _make_hertz_fn(tip_radius_nm)
                popt, _ = curve_fit(func, x, y, p0=[e_guess, contact_guess],
                                     bounds=([0, min(x)], [np.inf, max(x)]), maxfev=8000)
                labels = ['E* (effective modulus, Pa)', 'x0 (contact point)']
            else:
                func = _make_dmt_fn(tip_radius_nm)
                adh_guess = max(-min(y), 0.0) + 1e-9
                popt, _ = curve_fit(func, x, y, p0=[e_guess, contact_guess, adh_guess],
                                     bounds=([0, min(x), 0], [np.inf, max(x), np.inf]), maxfev=8000)
                labels = ['E* (effective modulus, Pa)', 'x0 (contact point)', 'F_adhesion']
        elif fit_type == 'wlc':
            lc_guess = max(x) * 1.2 if max(x) > 0 else 1.0
            popt, _ = curve_fit(wlc_fn, x, y, p0=[0.4, lc_guess],
                                 bounds=([0.01, max(x) * 0.5], [50, max(x) * 5]), maxfev=8000)
            func = wlc_fn
            labels = ['Lp (persistence length, nm)', 'Lc (contour length, nm)']
        else:
            return None, None, None

        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
        fit_params = list(zip(labels, [round(float(p), 6) for p in popt]))
        return fit_params, r_squared, func
    except (RuntimeError, ValueError):
        return None, None, None


def detect_unfolding_events(x, y, prominence=None):
    """Finds force peaks on a retract curve — the sawtooth pattern characteristic of
    single-molecule unfolding events. Each event is visually verifiable on the rendered
    plot, same principle as the porosity overlay: check it before trusting the count."""
    if len(y) < 5:
        return []
    prominence = prominence if prominence else 0.1 * (np.ptp(y) or 1)
    peak_idx, props = find_peaks(y, prominence=prominence)
    events = [{'position': float(x[i]), 'force': float(y[i])} for i in peak_idx]
    return events


def generate_force_curve_plot(x, y_approach, y_retract, fit_type, tip_radius_nm, out_path, find_unfolding=False):
    """Plots the approach curve (and retract, if present), fits the requested contact-
    mechanics or WLC model to whichever curve the model applies to, and — for SMFS — marks
    detected unfolding events. Returns a results dict; any field that couldn't be computed
    from real data is left out entirely rather than set to a placeholder."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y_approach, color='#2b6cb0', linewidth=1.6, label='Approach')
    if y_retract is not None:
        ax.plot(x, y_retract, color='#e53e3e', linewidth=1.6, label='Retract')

    results = {}

    if fit_type in ('hertz', 'dmt'):
        fit_params, r_squared, func = fit_force_curve(np.array(x), np.array(y_approach), fit_type, tip_radius_nm)
        if fit_params:
            x_smooth = np.linspace(min(x), max(x), 300)
            ax.plot(x_smooth, func(x_smooth, *[p[1] for p in fit_params]), '--', color='#2f855a',
                     linewidth=1.8, label=f'{fit_type.upper()} fit')
            results['fit_params'] = fit_params
            results['r_squared'] = r_squared
            results['elastic_modulus_pa'] = fit_params[0][1]
    elif fit_type == 'wlc':
        target_y = y_retract if y_retract is not None else y_approach
        fit_params, r_squared, func = fit_force_curve(np.array(x), np.array(target_y), 'wlc', None)
        if fit_params:
            x_smooth = np.linspace(min(x), max(x) * 0.98, 300)
            ax.plot(x_smooth, func(x_smooth, *[p[1] for p in fit_params]), '--', color='#2f855a',
                     linewidth=1.8, label='WLC fit')
            results['fit_params'] = fit_params
            results['r_squared'] = r_squared

    if y_retract is not None:
        results['adhesion_force'] = float(np.min(y_retract))
        if find_unfolding:
            events = detect_unfolding_events(np.array(x), np.array(y_retract))
            if events:
                ax.scatter([e['position'] for e in events], [e['force'] for e in events],
                           color='#dd6b20', zorder=5, s=40, marker='v', label='Unfolding event')
                results['unfolding_events'] = events

    ax.set_xlabel('Distance / extension')
    ax.set_ylabel('Force / deflection')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return results


NUMERIC_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def _numeric_fraction(fields):
    if not fields:
        return 0.0
    stripped = [f.strip().strip('"').strip("'").strip() for f in fields]
    hits = sum(1 for f in stripped if NUMERIC_RE.match(f))
    return hits / len(fields)


DELIMITER_MAP = {'auto': None, 'comma': ',', 'semicolon': ';', 'tab': '\t', 'whitespace': r'\s+'}


def pick_xy_columns(df):
    """Picks X and Y columns automatically based on which columns actually contain usable
    numeric data — not by trusting header names (which instrument exports often get wrong,
    e.g. a stray title line becoming the header, or a trailing delimiter creating an
    'Unnamed' column). Returns (x_col, y_col) or (None, None) if nothing usable is found."""
    numeric_counts = {}
    for col in df.columns:
        numeric_counts[col] = pd.to_numeric(df[col], errors='coerce').notna().sum()

    # keep columns with at least 2 real numeric values, in their original left-to-right order
    valid_cols = [c for c in df.columns if numeric_counts[c] >= 2]

    if len(valid_cols) >= 2:
        return valid_cols[0], valid_cols[1]
    elif len(df.columns) >= 2:
        return df.columns[0], df.columns[1]
    return None, None


def read_tabular_file(filepath, ext, delimiter_override=None, header_row_override=None):
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath, header=header_row_override if header_row_override is not None else 0)

    # manual override path — honor the user's explicit choice instead of guessing
    if delimiter_override is not None or header_row_override is not None:
        sep = delimiter_override if delimiter_override else ','
        skip = header_row_override if header_row_override is not None else 0
        return pd.read_csv(filepath, sep=sep, engine='python', skiprows=skip, header=0)

    return _auto_read_tabular_file(filepath)


def _auto_read_tabular_file(filepath):

    # Read raw lines so we can detect and skip instrument-export metadata
    # (many exporters — e.g. electrochemistry/EC-Lab, XRD instruments — prepend
    # several lines like "Technique: CV" or "Nb header lines : 5" before the real table).
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = [ln.rstrip('\n\r') for ln in fh.readlines()]
    lines = [ln for ln in lines if ln.strip() != '']

    if not lines:
        raise ValueError("File appears to be empty.")

    candidate_delimiters = [',', ';', '\t', r'\s+']

    best_delim = None
    best_score = (0, 1)   # (consistency = how many lines hit the mode, field count)
    for delim in candidate_delimiters:
        counts = [len(re.split(delim, ln.strip())) for ln in lines[:min(40, len(lines))]]
        counts = [c for c in counts if c > 1]
        if not counts:
            continue
        mode_count = max(set(counts), key=counts.count)
        consistency = counts.count(mode_count)
        score = (consistency, mode_count)
        # prefer the delimiter that splits the MOST lines consistently into the same field
        # count — not just whichever delimiter happens to produce the highest count on one line
        # (e.g. a stray space inside a header cell can fool a naive "most fields wins" approach)
        if score > best_score:
            best_score = score
            best_delim = delim

    if best_delim is None:
        # nothing split into multiple fields with any delimiter — fall back to pandas' own sniffing
        return pd.read_csv(filepath, sep=None, engine='python')

    # find the first line that looks like real numeric data using this delimiter
    first_data_idx = None
    for i, ln in enumerate(lines):
        fields = [f for f in re.split(best_delim, ln.strip()) if f != '']
        first_field_numeric = bool(fields) and bool(NUMERIC_RE.match(fields[0].strip().strip('"').strip("'")))
        if len(fields) >= 2 and _numeric_fraction(fields) >= 0.6 and first_field_numeric:
            first_data_idx = i
            break

    if first_data_idx is None:
        # no clearly-numeric row found — just let pandas try from the top
        return pd.read_csv(filepath, sep=best_delim, engine='python')

    # is there a header row immediately above the data?
    header_idx = None
    if first_data_idx > 0:
        prev_fields = [f for f in re.split(best_delim, lines[first_data_idx - 1].strip()) if f != '']
        data_fields = [f for f in re.split(best_delim, lines[first_data_idx].strip()) if f != '']
        if len(prev_fields) == len(data_fields):
            mostly_non_numeric = _numeric_fraction(prev_fields) < 0.5
            # wide-format matrix files often have numeric column labels (e.g. wavelengths) for
            # every column except the first — catch that case even though the row is mostly numeric
            first_cell_is_label = (
                not NUMERIC_RE.match(prev_fields[0].strip().strip('"').strip("'"))
                and NUMERIC_RE.match(data_fields[0].strip().strip('"').strip("'") or 'x')
            )
            if mostly_non_numeric or first_cell_is_label:
                header_idx = first_data_idx - 1

    if header_idx is not None:
        return pd.read_csv(filepath, sep=best_delim, engine='python', skiprows=header_idx, header=0)
    else:
        # no usable header line — read from the first data row and assign generic column names
        df = pd.read_csv(filepath, sep=best_delim, engine='python', skiprows=first_data_idx, header=None)
        df.columns = [f"Col{i+1}" for i in range(df.shape[1])]
        return df


def generate_fit_analysis(fit_type, fit_params, r_squared, x_col='X', y_col='Y'):
    """Turns fit results into a plain-language interpretation. Rule-based, not AI-generated —
    useful for a quick read of what the numbers actually mean."""
    if not fit_type or fit_type == 'none':
        return f"No curve fit was applied — this is a raw plot of {y_col} vs {x_col}."

    if not fit_params:
        return f"A {fit_type} fit was attempted but did not converge — the data may not follow this shape, or may need cleaning (outliers, noise, insufficient range)."

    params = dict(fit_params)
    notes = []

    # fit quality, in plain language
    if r_squared is not None:
        if r_squared >= 0.98:
            notes.append(f"The {fit_type} fit is excellent (R² = {r_squared:.4f}), meaning {y_col} follows this relationship very closely.")
        elif r_squared >= 0.9:
            notes.append(f"The {fit_type} fit is good (R² = {r_squared:.4f}), capturing most of the trend with some scatter.")
        elif r_squared >= 0.7:
            notes.append(f"The {fit_type} fit is moderate (R² = {r_squared:.4f}) — the general trend holds, but there's noticeable deviation from the model.")
        else:
            notes.append(f"The {fit_type} fit is weak (R² = {r_squared:.4f}) — this model may not be the right shape for this data; consider trying a different fit type.")

    # fit-specific interpretation
    if fit_type == 'linear':
        m = params.get('m (slope)')
        c = params.get('c (intercept)')
        if m is not None:
            direction = "increases" if m > 0 else "decreases"
            notes.append(f"{y_col} {direction} by about {abs(m):.4g} for every 1-unit increase in {x_col}.")
        if c is not None:
            notes.append(f"At {x_col} = 0, {y_col} is predicted to be about {c:.4g}.")

    elif fit_type == 'quadratic':
        a = params.get('a')
        if a is not None:
            shape = "curves upward (concave up)" if a > 0 else "curves downward (concave down)"
            notes.append(f"The relationship {shape}, suggesting a changing rate rather than a constant one.")

    elif fit_type == 'exponential':
        b = params.get('b')
        if b is not None:
            trend = "grows" if b > 0 else "decays"
            notes.append(f"{y_col} {trend} exponentially with {x_col} (rate constant b = {b:.4g}).")

    elif fit_type == 'gaussian':
        mu = params.get('mean')
        sigma = params.get('sigma')
        if mu is not None:
            notes.append(f"The peak is centered around {x_col} = {mu:.4g}.")
        if sigma is not None:
            notes.append(f"The peak width (σ) is about {abs(sigma):.4g}, indicating how sharp or broad the feature is.")

    return " ".join(notes)



def generate_plot_and_fit(x, y, fit_type, out_path):
    """Plots the data and, if requested, overlays a fitted curve. Returns (fit_params, r_squared)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=20, color='#2b6cb0', label='Data', zorder=3)

    fit_params = None
    r_squared = None

    if fit_type and fit_type != 'none' and fit_type in FIT_FUNCTIONS:
        func, param_names = FIT_FUNCTIONS[fit_type]
        try:
            if fit_type == 'gaussian':
                p0 = [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 4, min(y)]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=8000)
            else:
                popt, _ = curve_fit(func, x, y, maxfev=8000)

            x_smooth = np.linspace(min(x), max(x), 300)
            y_smooth = func(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, color='#e53e3e', linewidth=2, label=f'{fit_type.capitalize()} fit', zorder=2)

            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None

            fit_params = list(zip(param_names, [round(float(p), 5) for p in popt]))
        except (RuntimeError, ValueError):
            fit_params = None
            r_squared = None

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return fit_params, r_squared


PLOT_COLORS = ['#2b6cb0', '#e53e3e', '#38a169', '#d69e2e', '#805ad5', '#dd6b20', '#0987a0', '#b83280']

COLORMAP_OPTIONS = ['default', 'tab10', 'Set1', 'Set2', 'Dark2', 'Paired', 'Accent', 'viridis', 'plasma', 'coolwarm', 'rainbow']


def get_series_colors(colormap_name, n):
    """Returns n hex colors from the chosen colormap (or the default hand-picked palette)."""
    if not colormap_name or colormap_name == 'default':
        return [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(n)]
    try:
        cmap = plt.get_cmap(colormap_name)
        if n <= 1:
            positions = [0.5]
        else:
            positions = np.linspace(0, 1, n)
        return [matplotlib.colors.to_hex(cmap(p)) for p in positions]
    except (ValueError, KeyError):
        return [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(n)]


def compute_series_stats(y):
    """Basic descriptive stats for the analysis panel — not AI-generated, just arithmetic."""
    return {
        'count': int(len(y)),
        'mean': round(float(np.mean(y)), 5),
        'std': round(float(np.std(y)), 5),
        'min': round(float(np.min(y)), 5),
        'max': round(float(np.max(y)), 5),
    }


def build_stats_analysis(label, stats, plot_type):
    parts = [f"{label}: {stats['count']} points, mean = {stats['mean']:.4g}, std dev = {stats['std']:.4g}, range [{stats['min']:.4g}, {stats['max']:.4g}]."]
    if stats['std'] > 0 and abs(stats['mean']) > 0:
        cv = abs(stats['std'] / stats['mean'])
        if cv < 0.05:
            parts.append("Very low variability — the values are tightly clustered.")
        elif cv > 0.5:
            parts.append("High variability — the values are quite spread out.")
    return " ".join(parts)


def generate_custom_plot(series_list, plot_type, options, out_path):
    """series_list: list of {x, y, label} (x may be None for histogram/box).
    options: dict with title, xlabel, ylabel, show_grid, show_legend, log_x, log_y, fit_type.
    Returns list of per-series result dicts: {label, stats, fit_type, fit_params, r_squared, analysis}."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    results = []

    if plot_type == 'histogram':
        for i, s in enumerate(series_list):
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            ax.hist(s['y'], bins=20, alpha=0.65, color=color, label=s['label'], edgecolor='white')
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'fit_type': None, 'fit_params': None, 'r_squared': None,
                             'analysis': build_stats_analysis(s['label'], stats, plot_type)})

    elif plot_type == 'box':
        ax.boxplot([s['y'] for s in series_list], labels=[s['label'] for s in series_list], patch_artist=True)
        for i, s in enumerate(series_list):
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'fit_type': None, 'fit_params': None, 'r_squared': None,
                             'analysis': build_stats_analysis(s['label'], stats, plot_type)})

    elif plot_type == 'bar':
        x_positions = np.arange(len(series_list))
        heights = [np.mean(s['y']) for s in series_list]
        colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(series_list))]
        ax.bar(x_positions, heights, color=colors)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([s['label'] for s in series_list], rotation=20, ha='right')
        for i, s in enumerate(series_list):
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'fit_type': None, 'fit_params': None, 'r_squared': None,
                             'analysis': build_stats_analysis(s['label'], stats, plot_type)})

    else:  # scatter or line
        for i, s in enumerate(series_list):
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            x, y, label = s['x'], s['y'], s['label']

            if plot_type == 'line':
                order = np.argsort(x)
                ax.plot(x[order], y[order], color=color, linewidth=1.6, marker='o', markersize=3, label=label)
            else:
                ax.scatter(x, y, s=18, color=color, alpha=0.75, label=label)

            fit_type = options.get('fit_type', 'none')
            fit_params, r_squared = None, None
            if fit_type and fit_type != 'none' and fit_type in FIT_FUNCTIONS:
                func, param_names = FIT_FUNCTIONS[fit_type]
                try:
                    if fit_type == 'gaussian':
                        p0 = [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 4, min(y)]
                        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=8000)
                    else:
                        popt, _ = curve_fit(func, x, y, maxfev=8000)
                    x_smooth = np.linspace(min(x), max(x), 300)
                    ax.plot(x_smooth, func(x_smooth, *popt), color=color, linewidth=1.6, linestyle='--')
                    y_pred = func(x, *popt)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
                    fit_params = list(zip(param_names, [round(float(p), 5) for p in popt]))
                except (RuntimeError, ValueError):
                    pass

            stats = compute_series_stats(y)
            analysis = generate_fit_analysis(fit_type, fit_params, r_squared, x_col=options.get('xlabel', 'X'), y_col=label) \
                if fit_type and fit_type != 'none' else build_stats_analysis(label, stats, plot_type)
            results.append({'label': label, 'stats': stats, 'fit_type': fit_type, 'fit_params': fit_params, 'r_squared': r_squared, 'analysis': analysis})

    if options.get('title'):
        ax.set_title(options['title'])
    ax.set_xlabel(options.get('xlabel') or 'X')
    ax.set_ylabel(options.get('ylabel') or 'Y')
    if options.get('log_x'):
        ax.set_xscale('log')
    if options.get('log_y'):
        ax.set_yscale('log')
    if options.get('show_grid', True):
        ax.grid(alpha=0.25)
    if options.get('show_legend', True) and plot_type not in ('bar',):
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return results


def generate_overlay_plot(series_list, out_path):
    """series_list: list of dicts with keys x, y, label, fit_type. Draws all series on one figure."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    results = []

    for i, series in enumerate(series_list):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        x, y, label, fit_type = series['x'], series['y'], series['label'], series['fit_type']

        ax.scatter(x, y, s=18, color=color, alpha=0.75, label=label, zorder=3)

        fit_params, r_squared = None, None
        if fit_type and fit_type != 'none' and fit_type in FIT_FUNCTIONS:
            func, param_names = FIT_FUNCTIONS[fit_type]
            try:
                if fit_type == 'gaussian':
                    p0 = [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 4, min(y)]
                    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=8000)
                else:
                    popt, _ = curve_fit(func, x, y, maxfev=8000)

                x_smooth = np.linspace(min(x), max(x), 300)
                y_smooth = func(x_smooth, *popt)
                ax.plot(x_smooth, y_smooth, color=color, linewidth=1.6, linestyle='--', zorder=2)

                y_pred = func(x, *popt)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
                fit_params = list(zip(param_names, [round(float(p), 5) for p in popt]))
            except (RuntimeError, ValueError):
                pass

        results.append({'label': label, 'fit_type': fit_type, 'fit_params': fit_params, 'r_squared': r_squared})

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return results



def fill_char_entry_from_form(entry):
    entry.technique_name = request.form['technique_name']
    date_str = request.form.get('date_scheduled')
    entry.date_scheduled = datetime.fromisoformat(date_str).date() if date_str else datetime.now().date()
    num_samples_str = request.form.get('num_samples')
    entry.num_samples = int(num_samples_str) if num_samples_str else None
    entry.sample_prep = request.form.get('sample_prep', '')
    entry.outcome = request.form.get('outcome', '')
    entry.interpretation = request.form.get('interpretation', '')


@app.route('/characterizations', methods=['GET', 'POST'])
def characterizations():
    if request.method == 'POST':
        new_entry = CharEntry(user_id=session['user_id'])
        fill_char_entry_from_form(new_entry)
        db.session.add(new_entry)
        db.session.commit()
        return redirect(url_for('characterizations'))

    entries = CharEntry.query.filter_by(user_id=session['user_id']).order_by(CharEntry.technique_name.asc(), CharEntry.date_scheduled.desc()).all()

    # group entries by technique name for the "log of previous data collections" view
    grouped = {}
    for e in entries:
        grouped.setdefault(e.technique_name, []).append(e)

    return render_template(
        'characterizations.html',
        page_title='Characterizations',
        grouped_entries=grouped,
        today=datetime.now().strftime('%Y-%m-%d'),
        sample_prep_options=SAMPLE_PREP_OPTIONS,
        outcome_options=OUTCOME_OPTIONS,
        banner_image='images/characterizations-banner.png',
    )


@app.route('/char-entry/<int:entry_id>/edit', methods=['GET', 'POST'])
def edit_char_entry(entry_id):
    entry = get_owned_or_404(CharEntry, entry_id)

    if request.method == 'POST':
        fill_char_entry_from_form(entry)
        db.session.commit()
        return redirect(url_for('characterizations'))

    return render_template(
        'edit_char_entry.html',
        entry=entry,
        sample_prep_options=SAMPLE_PREP_OPTIONS,
        outcome_options=OUTCOME_OPTIONS,
    )


@app.route('/char-entry/<int:entry_id>/delete', methods=['POST'])
def delete_char_entry(entry_id):
    entry = get_owned_or_404(CharEntry, entry_id)
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for('characterizations'))


TICK_COLOR_DEFAULT = '#333333'


def dp_get_state():
    """Persistent per-session plotting state for the sidebar interface."""
    state = session.get('dp_state')
    if not state:
        state = {
            'file_ids': [],
            'plot_type': 'scatter_plot',
            'derivative': False,
            'format': {
                'legend': True,
                'legend_loc': 'best',
                'line_width': 1.6,
                'marker_size': 18,
                'tick_width': 1.0,
                'label_size': 11,
                'bold_labels': False,
                'grid': True,
                'colormap': 'default',
                'legend_orientation': 'vertical',
                'legend_scale': 1.0,
                'log_x': False, 'log_y': False,
                'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None,
            },
            'wide_mode': False,
        }
        session['dp_state'] = state
    return state


def dp_save_state(state):
    session['dp_state'] = state


def dp_selected_files(state):
    ids = state.get('file_ids', [])
    if not ids:
        return []
    files = DataFile.query.filter(DataFile.id.in_(ids), DataFile.user_id == session['user_id']).all()
    by_id = {f.id: f for f in files}
    return [by_id[i] for i in ids if i in by_id]


def dp_build_series(state):
    """Auto-detects X/Y per selected file (no manual column picking) and optionally
    converts each series to its numerical derivative dY/dX, with an approximate
    uncertainty band from a rolling standard deviation (not true experimental error,
    since no repeat measurements are available — just a local-noise estimate)."""
    files = dp_selected_files(state)
    series_list = []
    errors = []

    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()
        try:
            df = read_tabular_file(filepath, ext, delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'), header_row_override=f.parse_header_row)
            x_col, y_col = pick_xy_columns(df)
            if not x_col:
                errors.append(f"{f.original_filename}: no usable numeric columns found.")
                continue
            x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
            y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 3:
                errors.append(f"{f.original_filename}: not enough valid data points.")
                continue

            if f.technique_name != 'Cyclic Voltammetry (CV)':
                # Sorting by X assumes Y is a function of X — true for a spectrum/chromatogram,
                # but wrong for a CV scan: the forward and reverse sweeps pass through the same
                # potential twice at different currents, so sorting collapses that loop into a
                # scrambled zig-zag between the two branches (which at normal line width renders
                # as a solid-looking filled blob). Keep the file's own recorded sweep order instead.
                order = np.argsort(x)
                x, y = x[order], y[order]
            label = f.label or f.original_filename
            yerr = None

            if state.get('derivative'):
                # duplicate x-values (e.g. forward/reverse sweeps in cyclic voltammetry) make
                # the derivative undefined at those points (dx=0) — drop consecutive duplicates first
                keep = np.concatenate(([True], np.diff(x) != 0))
                x_d, y_d = x[keep], y[keep]

                if len(x_d) < 3:
                    errors.append(f"{f.original_filename}: not enough distinct X values to compute a derivative.")
                    continue

                dydx = np.gradient(y_d, x_d)
                finite = np.isfinite(dydx)
                x_d, dydx = x_d[finite], dydx[finite]

                # rough local-noise error estimate via rolling std of the derivative (window=5)
                window = 5
                yerr = np.array([
                    np.std(dydx[max(0, i - window // 2): i + window // 2 + 1])
                    for i in range(len(dydx))
                ])
                x, y = x_d, dydx
                label = f"{label} (dY/dX)"

            series_list.append({'file_id': f.id, 'x': x, 'y': y, 'yerr': yerr, 'label': label, 'x_col': x_col, 'y_col': y_col})
        except Exception as e:
            errors.append(f"{f.original_filename}: {e}")

    return series_list, errors


def _looks_like_cv_columns(x_col, y_col):
    xl = (x_col or '').lower()
    yl = (y_col or '').lower()
    x_is_potential = 'potential' in xl or re.search(r'\bv\b', xl) or 'volt' in xl
    y_is_current = 'current' in yl or re.search(r'\b(u?a|ma)\b', yl) or 'amp' in yl
    return bool(x_is_potential and y_is_current)


def compute_cv_metrics(x, y):
    """Standard cyclic-voltammetry peak metrics from a single scan."""
    ipa_idx = int(np.argmax(y))
    ipc_idx = int(np.argmin(y))
    ipa, epa = float(y[ipa_idx]), float(x[ipa_idx])
    ipc, epc = float(y[ipc_idx]), float(x[ipc_idx])
    delta_ep = abs(epa - epc)
    e_half = (epa + epc) / 2
    ratio = abs(ipa / ipc) if ipc != 0 else None
    return {'ipa': ipa, 'epa': epa, 'ipc': ipc, 'epc': epc, 'delta_ep': delta_ep, 'e_half': e_half, 'ratio': ratio}


def build_electrochemical_analysis(series_list):
    """Extracts catalyst-relevant metrics from cyclic voltammetry curves: peak currents/potentials,
    peak separation (a measure of electron-transfer kinetics/reversibility), and how peak current
    scales with the swept parameter (distinguishing diffusion-controlled vs surface-confined behavior)."""
    metrics = []
    for s in series_list:
        m = compute_cv_metrics(s['x'], s['y'])
        m['label'] = s['label']
        m['param'] = extract_numeric_label(s['label'])
        metrics.append(m)

    parts = []
    n = len(metrics)

    ipa_vals = [m['ipa'] for m in metrics]
    ipc_vals = [m['ipc'] for m in metrics]
    delta_eps = [m['delta_ep'] for m in metrics]
    e_halves = [m['e_half'] for m in metrics]
    ratios = [m['ratio'] for m in metrics if m['ratio'] is not None]

    parts.append(f"Anodic peak current (Ipa) ranges from {min(ipa_vals):.4g} to {max(ipa_vals):.4g}; cathodic peak current (Ipc) ranges from {min(ipc_vals):.4g} to {max(ipc_vals):.4g}.")

    mean_delta_ep = float(np.mean(delta_eps))
    if mean_delta_ep < 0.09:
        reversibility = "close to the ~59 mV expected for a reversible one-electron transfer, indicating fast, near-reversible electron-transfer kinetics"
    elif mean_delta_ep < 0.2:
        reversibility = "larger than the ideal reversible value, indicating quasi-reversible electron-transfer kinetics (some sluggishness at the electrode surface)"
    else:
        reversibility = "substantially larger than the reversible limit, indicating slow, irreversible electron-transfer kinetics — often linked to higher overpotential or poorer intrinsic catalytic activity"
    parts.append(f"The average peak-to-peak separation (ΔEp) is {mean_delta_ep*1000:.1f} mV, {reversibility}.")

    if ratios:
        mean_ratio = float(np.mean(ratios))
        if 0.85 <= mean_ratio <= 1.15:
            parts.append(f"The Ipa/Ipc ratio averages {mean_ratio:.2f}, close to 1 — consistent with a simple, uncomplicated redox couple.")
        else:
            parts.append(f"The Ipa/Ipc ratio averages {mean_ratio:.2f}, deviating from 1 — this can indicate a coupled chemical reaction (EC mechanism), adsorption effects, or catalytic turnover consuming/regenerating the redox species.")

    # relate peak current to the swept parameter (e.g. scan rate or concentration) if labels are numeric
    params = [m['param'] for m in metrics if m['param'] is not None]
    if len(params) >= 3:
        ipa_for_params = [m['ipa'] for m in metrics if m['param'] is not None]
        with np.errstate(invalid='ignore'):
            r_linear = np.corrcoef(params, ipa_for_params)[0, 1]
            sqrt_params = np.sqrt(np.abs(params))
            r_sqrt = np.corrcoef(sqrt_params, ipa_for_params)[0, 1]

        if not np.isnan(r_linear) and not np.isnan(r_sqrt):
            if abs(r_sqrt) > abs(r_linear) and abs(r_sqrt) > 0.85:
                parts.append(f"Ipa correlates more strongly with the square root of the labeled parameter (r = {r_sqrt:.2f}) than linearly (r = {r_linear:.2f}) — consistent with a diffusion-controlled process (Randles–Ševčík behavior), suggesting the catalytic reaction is limited by mass transport of the analyte to the electrode.")
            elif abs(r_linear) > 0.85:
                parts.append(f"Ipa scales approximately linearly with the labeled parameter (r = {r_linear:.2f}) — consistent with a surface-confined or adsorption-controlled process, where the catalyst's active-site coverage governs the response rather than diffusion.")
            else:
                parts.append(f"Ipa does not show a strong linear (r = {r_linear:.2f}) or square-root (r = {r_sqrt:.2f}) relationship with the labeled parameter — the underlying mechanism may be mixed or influenced by other factors.")

    if len(e_halves) >= 2:
        e_half_shift = max(e_halves) - min(e_halves)
        if e_half_shift > 0.02:
            parts.append(f"The formal potential (E1/2) shifts by {e_half_shift*1000:.1f} mV across the dataset, which may reflect changes in the catalyst's electronic environment or surface state between samples.")
        else:
            parts.append(f"The formal potential (E1/2) stays consistent across the dataset (shift < {e_half_shift*1000:.1f} mV), suggesting a stable redox process across samples.")

    return " ".join(parts)



def extract_numeric_label(label):
    match = re.search(r'[-+]?\d*\.?\d+', label)
    return float(match.group()) if match else None


def build_overall_analysis(series_list, plot_type):
    """A single interpretation of the WHOLE plotted graph, not a per-file repeat —
    covers combined range, whether curves cluster or diverge, and whether any numeric
    value in the file labels correlates with curve amplitude (useful when files are
    named by a swept parameter, e.g. concentration or scan rate)."""
    parts = []
    n = len(series_list)
    parts.append(f"This graph shows {n} dataset{'s' if n != 1 else ''}" + (f" as a {plot_type} plot." if plot_type else "."))

    all_y = np.concatenate([s['y'] for s in series_list])
    parts.append(f"Across all curves combined, values range from {np.min(all_y):.4g} to {np.max(all_y):.4g}, with an overall mean of {np.mean(all_y):.4g}.")

    if n >= 2:
        means = [np.mean(s['y']) for s in series_list]
        overall_std = np.std(all_y)
        spread_of_means = np.std(means)
        if overall_std > 0:
            if spread_of_means / overall_std < 0.25:
                parts.append("The curves are closely clustered together, suggesting similar overall behavior across the datasets.")
            else:
                parts.append("The curves are noticeably separated from one another, indicating distinct behavior between datasets.")

    # try correlating a numeric value in each label (e.g. concentration, %, cycle number)
    # with each curve's amplitude, to spot a systematic trend across the samples
    if n >= 3:
        numeric_labels, amplitudes = [], []
        for s in series_list:
            num = extract_numeric_label(s['label'])
            if num is not None:
                numeric_labels.append(num)
                amplitudes.append(float(np.max(s['y']) - np.min(s['y'])))

        if len(numeric_labels) >= 3:
            with np.errstate(invalid='ignore'):
                r = np.corrcoef(numeric_labels, amplitudes)[0, 1]
            if not np.isnan(r) and abs(r) > 0.6:
                direction = "increases" if r > 0 else "decreases"
                parts.append(f"The amplitude of each curve {direction} with the numeric value in its file name (correlation r = {r:.2f}), suggesting a systematic, concentration- or parameter-dependent trend across your samples.")
            elif not np.isnan(r):
                parts.append(f"No strong relationship was found between the numeric value in each file name and curve amplitude (r = {r:.2f}) — the differences between curves may be due to other factors.")

    return " ".join(parts)



def dp_render_plot(state):
    numeric_color_values = []
    if state.get('wide_mode'):
        series_list, errors, numeric_color_values = tech_build_wide_series(state)
    else:
        series_list, errors = dp_build_series(state)

    if not series_list:
        return None, [], errors

    fmt = state['format']
    plot_type = state.get('plot_type', 'scatter_plot')
    fig, ax = plt.subplots(figsize=(8, 5.5))

    use_colorbar = False
    if state.get('wide_mode') and len(numeric_color_values) == len(series_list) and len(series_list) > 8:
        cmap = plt.get_cmap('viridis')
        vmin, vmax = min(numeric_color_values), max(numeric_color_values)
        norm = lambda v: (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        colors = [matplotlib.colors.to_hex(cmap(norm(v))) for v in numeric_color_values]
        use_colorbar = True
    else:
        colors = get_series_colors(fmt.get('colormap', 'default'), len(series_list))

    results = []

    def add_result(label, y_values, note=None, shape=None):
        stats = compute_series_stats(y_values)
        analysis = build_stats_analysis(label, stats, plot_type)
        if note:
            analysis = note + " " + analysis
        results.append({'label': label, 'stats': stats, 'analysis': analysis, 'shape': shape})

    if plot_type == 'sankey_alluvial':
        ax.text(0.5, 0.5, "Sankey/Alluvial needs flow-structured data\n(source, target, value columns) —\nnot supported by simple X/Y column selection yet.",
                ha='center', va='center', fontsize=11, color='#888', transform=ax.transAxes, wrap=True)
        ax.axis('off')
        for s in series_list:
            add_result(s['label'], s['y'], note="(Sankey/Alluvial not rendered — see note on plot.)")

    elif plot_type == 'heatmap':
        # stack each series' Y values as a row (resampled to a common length) to compare magnitude across series
        target_len = min(len(s['y']) for s in series_list)
        matrix = np.array([s['y'][:target_len] for s in series_list])
        im = ax.imshow(matrix, aspect='auto', cmap=fmt.get('colormap') if fmt.get('colormap', 'default') != 'default' else 'viridis')
        ax.set_yticks(range(len(series_list)))
        ax.set_yticklabels([s['label'] for s in series_list], fontsize=8)
        ax.set_xlabel('Sample index', fontsize=fmt['label_size'])
        fig.colorbar(im, ax=ax, shrink=0.8)
        for s in series_list:
            add_result(s['label'], s['y'])

    elif plot_type == 'volcano_plot':
        if len(series_list) >= 2:
            fc = series_list[0]['y']
            pval = series_list[1]['y']
            n = min(len(fc), len(pval))
            fc, pval = fc[:n], pval[:n]
            neglogp = -np.log10(np.clip(pval, 1e-300, None))
            significant = (np.abs(fc) > 1) & (pval < 0.05)
            ax.scatter(fc[~significant], neglogp[~significant], s=fmt['marker_size'], color='#999', alpha=0.6, label='Not significant')
            ax.scatter(fc[significant], neglogp[significant], s=fmt['marker_size'], color='#e53e3e', alpha=0.8, label='Significant (|FC|>1, p<0.05)')
            ax.axhline(-np.log10(0.05), color='#888', linestyle='--', linewidth=1)
            ax.axvline(1, color='#888', linestyle='--', linewidth=1)
            ax.axvline(-1, color='#888', linestyle='--', linewidth=1)
            ax.set_xlabel('log2(Fold Change)', fontsize=fmt['label_size'])
            ax.set_ylabel('-log10(p-value)', fontsize=fmt['label_size'])
            add_result('Volcano', fc, note=f"{int(significant.sum())} of {n} points meet |FC|>1 and p<0.05.")
        else:
            ax.text(0.5, 0.5, "Volcano Plot needs 2 Y columns:\nfirst = log2(Fold Change), second = p-value.", ha='center', va='center', color='#888', transform=ax.transAxes)
            ax.axis('off')

    elif plot_type == 'survival_curve':
        for i, s in enumerate(series_list):
            times = np.sort(s['y'])
            n = len(times)
            at_risk = n
            surv_prob = 1.0
            step_x, step_y = [0], [1.0]
            for t in times:
                surv_prob *= (1 - 1 / at_risk)
                step_x.append(t)
                step_y.append(surv_prob)
                at_risk -= 1
            ax.step(step_x, step_y, where='post', color=colors[i], linewidth=fmt['line_width'], label=s['label'])
            add_result(s['label'], s['y'], note="Kaplan-Meier estimate assuming no censoring (all values treated as event times).")
        ax.set_xlabel('Time', fontsize=fmt['label_size'])
        ax.set_ylabel('Survival probability', fontsize=fmt['label_size'])
        ax.set_ylim(0, 1.05)

    elif plot_type == 'forest_plot':
        y_positions = np.arange(len(series_list))
        effects = [np.mean(s['y']) for s in series_list]
        errs = [np.std(s['y']) for s in series_list]
        ax.errorbar(effects, y_positions, xerr=errs, fmt='o', color='#2b6cb0', ecolor='#888', capsize=4)
        ax.axvline(0, color='#888', linestyle='--', linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([s['label'] for s in series_list])
        ax.set_xlabel('Effect size (mean ± std dev shown; not a true 95% CI)', fontsize=fmt['label_size'] * 0.85)
        for s in series_list:
            add_result(s['label'], s['y'], note="Forest plot uses mean ± std dev as a stand-in for effect size ± CI — supply true CI bounds for a rigorous meta-analysis plot.")

    elif plot_type == 'waterfall_chart':
        for s_idx, s in enumerate(series_list):
            values = s['y']
            cumulative = 0
            for i, v in enumerate(values):
                color = '#38a169' if v >= 0 else '#e53e3e'
                ax.bar(i, v, bottom=cumulative, color=color, width=0.6)
                cumulative += v
            add_result(s['label'], values, note=f"Final cumulative value: {cumulative:.4g}.")
        ax.set_xlabel('Step', fontsize=fmt['label_size'])
        ax.set_ylabel('Cumulative value', fontsize=fmt['label_size'])

    elif plot_type == 'bubble_pca':
        if len(series_list) >= 3:
            x, y, size_raw = series_list[0]['y'], series_list[1]['y'], series_list[2]['y']
            n = min(len(x), len(y), len(size_raw))
            x, y, size_raw = x[:n], y[:n], size_raw[:n]
            size_range = np.ptp(size_raw) or 1
            sizes = 30 + 300 * (size_raw - size_raw.min()) / size_range
            ax.scatter(x, y, s=sizes, color=colors[0], alpha=0.6, edgecolor='white')
            ax.set_xlabel(series_list[0]['label'], fontsize=fmt['label_size'])
            ax.set_ylabel(series_list[1]['label'], fontsize=fmt['label_size'])
            add_result('Bubble', x, note=f"Bubble size encodes '{series_list[2]['label']}'.")
        else:
            for i, s in enumerate(series_list):
                ax.scatter(s['x'], s['y'], s=fmt['marker_size'], color=colors[i], alpha=0.8, label=s['label'])
                add_result(s['label'], s['y'], note="Bubble/PCA needs 3 Y columns (X, Y, size) for a true bubble chart — showing a plain scatter instead.")

    elif plot_type == 'box_violin':
        parts = ax.violinplot([s['y'] for s in series_list], showmeans=True, showextrema=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        ax.boxplot([s['y'] for s in series_list], widths=0.15, patch_artist=True,
                   boxprops=dict(facecolor='white', alpha=0.8))
        ax.set_xticks(range(1, len(series_list) + 1))
        ax.set_xticklabels([s['label'] for s in series_list], rotation=15, ha='right')
        for s in series_list:
            add_result(s['label'], s['y'])

    elif plot_type == 'histogram_density':
        for i, s in enumerate(series_list):
            ax.hist(s['y'], bins=20, alpha=0.5, color=colors[i], label=s['label'], density=True, edgecolor='white')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s['y'])
                x_range = np.linspace(min(s['y']), max(s['y']), 200)
                ax.plot(x_range, kde(x_range), color=colors[i], linewidth=fmt['line_width'])
            except Exception:
                pass
            add_result(s['label'], s['y'])
        ax.set_ylabel('Density', fontsize=fmt['label_size'])

    else:
        for i, s in enumerate(series_list):
            color = colors[i]

            if plot_type == 'line_timeseries' or state.get('derivative'):
                ax.plot(s['x'], s['y'], color=color, linewidth=fmt['line_width'], label=s['label'])
                if s.get('yerr') is not None:
                    ax.fill_between(s['x'], s['y'] - s['yerr'], s['y'] + s['yerr'], color=color, alpha=0.15)
            elif plot_type == 'bar_column':
                ax.bar(i, np.mean(s['y']), color=color, label=s['label'])
            else:  # scatter_plot (default)
                ax.scatter(s['x'], s['y'], s=fmt['marker_size'], color=color, alpha=0.8, label=s['label'])

            shape = None
            if plot_type in ('scatter_plot', 'line_timeseries') and not state.get('derivative'):
                shape = suggest_curve_shape(s['x'], s['y'])
            add_result(s['label'], s['y'], shape=shape)

        if plot_type == 'bar_column':
            ax.set_xticks(range(len(series_list)))
            ax.set_xticklabels([s['label'] for s in series_list], rotation=20, ha='right')

    if plot_type not in ('heatmap', 'sankey_alluvial'):
        label_weight = 'bold' if fmt.get('bold_labels') else 'normal'
        if plot_type not in ('survival_curve', 'forest_plot', 'waterfall_chart', 'volcano_plot'):
            ax.set_xlabel('X', fontsize=fmt['label_size'], fontweight=label_weight)
            ax.set_ylabel('dY/dX' if state.get('derivative') else 'Y', fontsize=fmt['label_size'], fontweight=label_weight)
        ax.tick_params(width=fmt['tick_width'])
        if fmt.get('log_x'):
            ax.set_xscale('log')
        if fmt.get('log_y'):
            ax.set_yscale('log')
        x_min, x_max = fmt.get('x_min'), fmt.get('x_max')
        if x_min is not None or x_max is not None:
            cur_min, cur_max = ax.get_xlim()
            ax.set_xlim(x_min if x_min is not None else cur_min, x_max if x_max is not None else cur_max)
        y_min, y_max = fmt.get('y_min'), fmt.get('y_max')
        if y_min is not None or y_max is not None:
            cur_min, cur_max = ax.get_ylim()
            ax.set_ylim(y_min if y_min is not None else cur_min, y_max if y_max is not None else cur_max)
        if fmt.get('grid', True):
            ax.grid(alpha=0.25)
        if use_colorbar:
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Series value')
        elif fmt.get('legend', True) and plot_type not in ('forest_plot', 'waterfall_chart'):
            legend_scale = fmt.get('legend_scale', 1.0)
            base_fontsize = 9 * legend_scale
            if fmt.get('legend_orientation') == 'horizontal':
                ncol = min(len(series_list), 3) if len(series_list) > 3 else len(series_list)
            else:
                ncol = 1
            legend = ax.legend(
                fontsize=base_fontsize, loc=fmt.get('legend_loc', 'best'), ncol=ncol,
                markerscale=legend_scale, framealpha=0.9,
            )
            legend.set_zorder(10)

    fig.tight_layout()
    plot_filename = f"dp_{session.get('_id', 'anon')}_{int(datetime.now().timestamp())}.png"
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)

    is_cv = (
        not state.get('derivative')
        and state.get('plot_type') in ('scatter', 'line')
        and series_list
        and all(_looks_like_cv_columns(s.get('x_col'), s.get('y_col')) for s in series_list)
    )
    if is_cv:
        overall_analysis = build_electrochemical_analysis(series_list)
    else:
        overall_analysis = build_overall_analysis(series_list, state.get('plot_type', 'scatter'))
    return plot_filename, results, errors, overall_analysis


TECHNIQUE_CATEGORIES = [
    ("Spectroscopy", ["FTIR", "UV-Vis", "Fluorescence", "Raman", "NMR (1H, 13C)", "CD (Circular Dichroism)"]),
    ("Mass & Separation", ["MALDI", "LC-MS", "HPLC / GC", "GPC / SEC"]),
    ("Microscopy & Imaging", ["SEM", "AFM", "TEM", "Confocal / Fluorescence", "EDS/EDX", "EBSD"]),
    ("Crystallography & Surface", ["XPS", "XRD", "BET Nitrogen Sorption"]),
    ("Biophysics & Kinetics", ["SPR / BLI", "ITC", "DLS"]),
    ("Cellular & Phenotypic", ["FACS / Flow Cytometry", "scRNA-seq / Omics"]),
    ("Mechanics & Electrochemistry", ["UTM / Nanoindentation", "Rheometer", "Cyclic Voltammetry (CV)"]),
]

# Computational lives on its own top-level page (see computational_home()), not nested under
# Data Interpretation's 7-category sidebar — kept as a separate list for that reason, and merged
# with TECHNIQUE_CATEGORIES below only where a technique needs to be looked up regardless of
# which top-level page it belongs to (slug resolution, tab lookup, parent-category lookup).
COMPUTATIONAL_CATEGORIES = [
    ("Computational", ["Monte Carlo", "DFT (small molecule)"]),
]

ALL_TECHNIQUE_CATEGORIES = TECHNIQUE_CATEGORIES + COMPUTATIONAL_CATEGORIES

# Tabs shown per technique — tailored to what's actually relevant for that measurement type.
# Techniques not listed explicitly fall back to DEFAULT_TECHNIQUE_TABS.
DEFAULT_TECHNIQUE_TABS = ["Select files", "Plot", "Format", "Analysis"]

TECHNIQUE_TABS = {
    "FTIR": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Baseline Correction", "Format", "Analysis"],
    "UV-Vis": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "Fluorescence": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "Raman": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Baseline Correction", "Format", "Analysis"],
    "NMR (1H, 13C)": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Integration", "Format", "Analysis"],
    "CD (Circular Dichroism)": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Format", "Analysis"],

    "MALDI": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "PMF", "Imaging", "Format", "Analysis"],
    "LC-MS": ["Select files", "Simulate", "Compare", "Chromatogram", "Formula ID", "Quantification", "Analysis"],
    "HPLC / GC": ["Select files", "Simulate", "Compare", "Plot Chromatogram", "Peak Picking", "Peak Integration", "Quantification", "Format", "Analysis"],
    "GPC / SEC": ["Select files", "Plot Chromatogram", "Molecular Weight", "Format", "Analysis"],

    "SEM": ["Select images", "Measure Particles", "Porosity", "Roughness", "Analysis"],
    "AFM": ["Select data", "Topography", "Mechanical", "Electrical", "Magnetic", "Chemical / Frictional", "Biological", "Analysis"],
    "TEM": ["Select images", "Measure Particles", "Segment", "Layer Thickness", "Defects", "SAED", "Lattice Fringes", "Strain Mapping", "Analysis"],
    "Confocal / Fluorescence": ["Select images", "Preprocess", "Segment", "Measure Particles", "Analysis"],
    "EDS/EDX": ["Select files", "Simulate", "Compare", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "EBSD": ["Select images", "Measure Particles", "Analysis"],

    "XPS": ["Select files", "Plot Spectrum", "Peak Fitting", "Format", "Analysis"],
    "XRD": ["Select files", "Plot Pattern", "Peak ID", "Crystallite Size", "Format", "Analysis"],
    "BET Nitrogen Sorption": ["Select files", "Plot Isotherm", "Surface Area", "Format", "Analysis"],

    "SPR / BLI": ["Select files", "Plot Sensorgram", "Kinetics Fitting", "Format", "Analysis"],
    "ITC": ["Select files", "Plot Thermogram", "Binding Fit", "Format", "Analysis"],
    "DLS": ["Select files", "Plot Distribution", "Format", "Analysis"],

    "FACS / Flow Cytometry": ["Select files", "Gating", "Plot", "Format", "Analysis"],
    "scRNA-seq / Omics": ["Select files", "Plot", "Format", "Analysis"],

    "UTM / Nanoindentation": ["Select files", "Plot Stress-Strain", "Modulus Fitting", "Format", "Analysis"],
    "Rheometer": ["Select files", "Plot", "Format", "Analysis"],
    "Cyclic Voltammetry (CV)": ["Select files", "Plot", "Format", "Derivative", "Analysis"],

    "Monte Carlo": ["Ising Model", "Monte Carlo Integration", "Random Walk"],
    "DFT (small molecule)": ["Run Calculation"],
}


def slugify_technique(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


TECHNIQUE_SLUGS = {slugify_technique(t): t for cat, techs in ALL_TECHNIQUE_CATEGORIES for t in techs}


def spectrum_plot_tab_name(technique_name):
    """The literal tab name each technique uses for its base spectrum/chromatogram plot —
    shared spectrum-pipeline techniques mostly use 'Plot Spectrum', but the chromatography
    and mechanics/electrochemistry ones use their own more accurate label."""
    if technique_name == 'LC-MS':
        return 'Chromatogram'
    if technique_name == 'HPLC / GC':
        return 'Plot Chromatogram'
    if technique_name == 'UTM / Nanoindentation':
        return 'Plot Stress-Strain'
    if technique_name in ('Rheometer', 'Cyclic Voltammetry (CV)'):
        return 'Plot'
    return 'Plot Spectrum'

# Plot modes allowed per technique, matching the Spectroscopy plot-type/graph-type table.
# Each entry: (internal_plot_type, display_label)
TECHNIQUE_PLOT_MODES = {
    "ftir": [("line_spectrum", "1D Transmittance / Absorbance Spectrum (Line Graph)")],
    "uv-vis": [
        ("line_spectrum", "1D Absorption Spectrum (Line Graph)"),
        ("calibration_scatter", "Linear Calibration Curve (Scatter Plot with Linear Fit)"),
    ],
    "raman": [("line_spectrum", "1D Raman Spectrum (Line Graph)")],
    "nmr-1h-13c": [
        ("line_spectrum", "1D Spectrum: Intensity vs Shift (Line Graph)"),
        ("contour_2d", "2D COSY / HSQC (Contour / Heatmap Plot)"),
    ],
    "fluorescence": [("line_spectrum", "Emission / Excitation Spectrum (Line Graph)")],
    "cd-circular-dichroism": [
        ("line_spectrum", "1D CD Spectrum (Line Graph)"),
        ("line_spectrum", "Thermal Denaturation Curve (Line Graph)"),
    ],
    "eds-edx": [("line_spectrum", "EDS/EDX Spectrum: Counts vs Energy (keV) (Line Graph)")],
    "lc-ms": [("line_spectrum", "Chromatogram: Intensity vs Retention Time (Line Graph)")],
    "hplc-gc": [("line_spectrum", "Chromatogram: Detector Signal vs Retention Time (Line Graph)")],
    "xps": [("line_spectrum", "Binding Energy vs Intensity (Line Graph)")],
    "utm-nanoindentation": [("line_spectrum", "Stress-Strain Curve (Line Graph)")],
    "rheometer": [("line_spectrum", "Rheology Curve (Line Graph)")],
    "cyclic-voltammetry-cv": [("line_spectrum", "Current vs Potential (Line Graph)")],
}


def tech_state_key(slug):
    return f'tech_state_{slug}'


# Grouped by which "box" each belongs to on the Format tab, so each box can be reset to
# its own defaults independently of the others.
DEFAULT_TECH_FORMAT_SECTIONS = {
    'legend': {'legend': True, 'legend_loc': 'best', 'legend_orientation': 'vertical', 'legend_scale': 1.0},
    'colors': {'colormap': 'default'},
    'lines': {'line_width': 1.6, 'marker_size': 18, 'tick_width': 1.0, 'grid': True, 'fill_under': False},
    'axis_labels': {'x_label': 'X', 'y_label': 'Y', 'label_size': 11, 'bold_labels': False},
    'axis': {'log_x': False, 'log_y': False, 'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None},
}


def default_tech_format():
    fmt = {}
    for section in DEFAULT_TECH_FORMAT_SECTIONS.values():
        fmt.update(section)
    return fmt


def tech_get_state(slug):
    key = tech_state_key(slug)
    state = session.get(key)
    if not state:
        modes = TECHNIQUE_PLOT_MODES.get(slug, [("line_spectrum", "Line Graph")])
        state = {
            'file_ids': [],
            'plot_type': modes[0][0],
            'derivative': False,
            'format': default_tech_format(),
            'layout': 'overlay',
            'peaks': {'prominence': 0.1, 'min_height': None},
            'smoothing': {'enabled': False, 'window': 11, 'polyorder': 3},
            'wide_mode': False,
        }
        session[key] = state
    return state


def tech_save_state(slug, state):
    session[tech_state_key(slug)] = state


# Reference peak-assignment tables — (range_min, range_max, description). Ranges are approximate
# and drawn from standard spectroscopy references; real assignment always needs chemist judgement,
# so this is offered as a starting interpretation, not a definitive one.
FTIR_TABLE = [
    (3200, 3550, "O–H / N–H stretch (alcohols, amines, hydrogen bonding)"),
    (2850, 3000, "C–H stretch (alkane, sp3)"),
    (3000, 3100, "C–H stretch (alkene/aromatic, sp2)"),
    (2210, 2260, "C≡N or C≡C stretch (nitrile/alkyne)"),
    (1735, 1750, "C=O stretch (ester)"),
    (1700, 1725, "C=O stretch (ketone/aldehyde)"),
    (1650, 1700, "C=O stretch (amide) or C=C stretch"),
    (1580, 1650, "N–H bend or aromatic C=C"),
    (1400, 1470, "C–H bend (CH2/CH3)"),
    (1000, 1300, "C–O / C–N stretch (esters, ethers, amines)"),
    (650, 900, "C–H out-of-plane bend (aromatic substitution pattern)"),
]

RAMAN_TABLE = [
    (1580, 1620, "G band — graphitic sp2 carbon (ordered)"),
    (1330, 1370, "D band — disordered/defective sp2 carbon"),
    (2650, 2700, "2D band — graphene layer number indicator"),
    (900, 1150, "C–C skeletal stretch"),
    (1600, 1650, "C=C stretch"),
    (1000, 1030, "Aromatic ring breathing mode"),
    (2800, 3000, "C–H stretch"),
]

# 1H NMR chemical shift ranges (ppm) — most common technique; 13C ranges differ substantially
# and aren't included here since the two are rarely distinguished automatically from a plain X/Y file.
NMR_1H_TABLE = [
    (0.0, 1.5, "Aliphatic C–H (alkyl, e.g. CH3/CH2 not adjacent to heteroatoms)"),
    (1.5, 2.5, "Allylic / benzylic C–H, or C–H alpha to C=O"),
    (2.5, 3.5, "C–H adjacent to N or halogen"),
    (3.3, 4.5, "C–H adjacent to O (ethers, alcohols) or O–CH3"),
    (4.5, 6.0, "Vinyl / alkene C–H"),
    (6.0, 8.5, "Aromatic C–H"),
    (9.0, 10.5, "Aldehyde C–H"),
    (10.0, 13.0, "Carboxylic acid O–H (broad, concentration-dependent)"),
]

# CD secondary-structure signature bands (nm) for proteins/peptides
CD_TABLE = [
    (190, 200, "positive band — often α-helix or random coil contribution"),
    (205, 212, "negative band — characteristic of α-helix"),
    (218, 225, "negative band — characteristic of α-helix (second minimum) or β-sheet"),
    (195, 200, "positive band — characteristic of β-sheet"),
    (215, 220, "negative band — characteristic of β-sheet"),
    (200, 210, "negative band — characteristic of random coil"),
]

# Characteristic X-ray line energies (keV, standard Bearden reference values) for common
# elements seen in EDS/EDX spectra — narrow windows around each Kα (or, for the heavier
# elements more commonly seen via their M line at low keV, Mα/Lα) line, same range-table
# shape as the other assign_peaks() tables above.
EDS_TABLE = [
    (0.25, 0.30, "C Kα (Carbon)"), (0.37, 0.42, "N Kα (Nitrogen)"), (0.50, 0.55, "O Kα (Oxygen)"),
    (0.65, 0.70, "F Kα (Fluorine)"), (1.00, 1.08, "Na Kα (Sodium)"), (1.20, 1.30, "Mg Kα (Magnesium)"),
    (1.45, 1.52, "Al Kα (Aluminum)"), (1.70, 1.78, "Si Kα (Silicon)"), (1.97, 2.05, "P Kα (Phosphorus)"),
    (2.02, 2.08, "Pt Mα (Platinum)"), (2.10, 2.15, "Au Mα (Gold)"), (2.28, 2.34, "S Kα (Sulfur)"),
    (2.59, 2.65, "Cl Kα (Chlorine)"), (2.95, 3.01, "Ag Lα (Silver)"), (3.28, 3.34, "K Kα (Potassium)"),
    (3.66, 3.72, "Ca Kα (Calcium)"), (4.48, 4.54, "Ti Kα (Titanium)"), (5.38, 5.44, "Cr Kα (Chromium)"),
    (5.87, 5.93, "Mn Kα (Manganese)"), (6.37, 6.43, "Fe Kα (Iron)"), (6.90, 6.96, "Co Kα (Cobalt)"),
    (7.44, 7.50, "Ni Kα (Nickel)"), (8.02, 8.08, "Cu Kα (Copper)"), (8.60, 8.66, "Zn Kα (Zinc)"),
]

# Common XPS binding energies (eV), grouped loosely by core level — genuinely ambiguous
# without knowing which element's core level was actually scanned (a C 1s survey covers a
# totally different chemistry than an O 1s survey at a similar-looking number), so the
# analysis text built from this always names the specific line, not just a bare eV value.
XPS_TABLE = [
    (98.0, 99.8, "Si 2p — elemental Si (Si–Si)"),
    (102.0, 104.5, "Si 2p — SiO2 / silicate (Si–O)"),
    (160.0, 162.5, "S 2p — sulfide (metal-S / thiolate)"),
    (163.0, 164.5, "S 2p — thiol / disulfide / elemental S"),
    (168.0, 170.0, "S 2p — sulfate / sulfonate (oxidized S)"),
    (282.0, 283.3, "C 1s — metal carbide"),
    (284.4, 285.3, "C 1s — C–C / C–H (aliphatic, incl. adventitious carbon reference)"),
    (285.5, 286.9, "C 1s — C–O / C–N (ether, alcohol, amine)"),
    (287.4, 289.6, "C 1s — C=O / O–C=O (carbonyl, carboxyl, ester)"),
    (290.5, 292.5, "C 1s — π→π* shake-up (aromatic/graphitic ring)"),
    (398.3, 399.6, "N 1s — amine / pyridinic N"),
    (399.7, 400.6, "N 1s — amide / pyrrolic N"),
    (400.8, 402.8, "N 1s — protonated amine / graphitic N+"),
    (529.4, 530.6, "O 1s — metal oxide (lattice O2−)"),
    (530.7, 532.0, "O 1s — hydroxide / oxygen-deficient oxide"),
    (532.1, 533.8, "O 1s — C=O / C–O (organic oxygen or adsorbed water)"),
    (74.0, 75.3, "Al 2p — metallic Al"),
    (75.4, 76.8, "Al 2p — Al2O3 (oxidized Al)"),
    (710.0, 711.2, "Fe 2p3/2 — Fe2+ (e.g. FeO)"),
    (711.3, 712.8, "Fe 2p3/2 — Fe3+ (e.g. Fe2O3/Fe3O4)"),
    (932.0, 933.3, "Cu 2p3/2 — Cu(0) / Cu(I)"),
    (933.4, 935.5, "Cu 2p3/2 — Cu(II)"),
]


def shirley_background(x, y, tol=1e-6, max_iters=50):
    """The standard XPS background — unlike a straight line between the endpoints, it
    tracks the step the inelastic-scattering tail actually produces: background at any
    point is proportional to the peak area still to come on the low-binding-energy side.
    Computed iteratively (Shirley's own method) until it stops changing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    y_left, y_right = float(ys[0]), float(ys[-1])
    b = np.full_like(ys, y_right)
    for _ in range(max_iters):
        signal = ys - b
        total = float(np.trapezoid(signal, xs))
        if abs(total) < 1e-12:
            break
        cum = np.concatenate([[0.0], np.cumsum((signal[:-1] + signal[1:]) / 2 * np.diff(xs))])
        area_from_here_to_end = cum[-1] - cum
        b_new = y_right + (y_left - y_right) * (area_from_here_to_end / total)
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    b_full = np.empty_like(b)
    b_full[order] = b
    return b_full


def fit_multi_gaussian_peaks(x, y, n_peaks, prominence=None):
    """Deconvolutes a background-subtracted spectrum into n_peaks overlapping Gaussian
    components via least-squares — genuine peak fitting (what XPS calls "peak fitting" is
    exactly this: separating chemical states that sit too close together to show up as
    separate maxima), not just relabeling the raw peak list. Seeds the fit from the
    n_peaks tallest local maxima, or evenly-spaced positions if fewer real maxima exist
    than requested. Returns (components, fitted_curve) or (None, None) if the fit fails."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_peaks = max(1, min(6, int(n_peaks)))

    idx, props = find_peaks(y, prominence=prominence or (np.ptp(y) * 0.05 or None))
    if len(idx) >= n_peaks:
        order = np.argsort(y[idx])[::-1][:n_peaks]
        seed_idx = np.sort(idx[order])
    else:
        seed_idx = np.linspace(0, len(x) - 1, n_peaks + 2)[1:-1].astype(int)

    span = float(x.max() - x.min()) or 1.0
    width_guess = span / (n_peaks * 4)
    p0 = []
    bounds_lo, bounds_hi = [], []
    for i in seed_idx:
        p0 += [float(y[i]) or 1.0, float(x[i]), width_guess]
        bounds_lo += [0, float(x.min()), width_guess / 10]
        bounds_hi += [np.inf, float(x.max()), span]

    def multi_gauss(xv, *params):
        result = np.zeros_like(xv)
        for i in range(0, len(params), 3):
            amp, cen, wid = params[i:i + 3]
            result = result + amp * np.exp(-((xv - cen) ** 2) / (2 * wid ** 2))
        return result

    try:
        popt, _ = curve_fit(multi_gauss, x, y, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000)
    except Exception:
        return None, None

    components = []
    for i in range(0, len(popt), 3):
        amp, cen, wid = popt[i:i + 3]
        area = float(amp * abs(wid) * np.sqrt(2 * np.pi))
        components.append({'amplitude': float(amp), 'center': float(cen), 'sigma': float(wid),
                            'fwhm': float(2.3548 * abs(wid)), 'area': area})
    components.sort(key=lambda c: c['center'])
    total_area = sum(c['area'] for c in components) or 1.0
    for c in components:
        c['pct_area'] = round(100 * c['area'] / total_area, 1)

    fitted_curve = multi_gauss(x, *popt)
    return components, fitted_curve

# Common LC-MS ESI adducts: the mass added to a neutral molecule's monoisotopic mass to get
# the observed m/z (accounts for the lost/gained electron on the charged species, not just
# the adduct atom's neutral mass). {key: (display label, mass added in Da)}.
LC_MS_ADDUCTS = {
    'M': ('Neutral / already-deconvoluted mass', 0.0),
    'M+H': ('[M+H]+  (positive ESI)', 1.007276),
    'M-H': ('[M-H]-  (negative ESI)', -1.007276),
    'M+Na': ('[M+Na]+  (positive ESI)', 22.989221),
    'M+K': ('[M+K]+  (positive ESI)', 38.963158),
    'M+NH4': ('[M+NH4]+  (positive ESI)', 18.033825),
    'M+Cl': ('[M+Cl]-  (negative ESI)', 34.969402),
}

# Monoisotopic masses (Da) of the elements searched for molecular formula prediction from
# accurate mass — scoped to organic small molecules/metabolites/peptide fragments, not
# inorganics or intact proteins (formula search doesn't scale to protein-sized masses).
LC_MS_ELEMENT_MASSES = {'C': 12.000000, 'H': 1.007825, 'N': 14.003074, 'O': 15.994915, 'P': 30.973762, 'S': 31.972071}


def lc_ms_format_formula(c, h, n, o, p, s):
    parts = []
    for sym, count in (('C', c), ('H', h), ('N', n), ('O', o), ('P', p), ('S', s)):
        if count > 0:
            parts.append(sym if count == 1 else f"{sym}{count}")
    return "".join(parts) or "—"


def generate_formula_candidates(target_mass, tolerance_ppm=10.0, max_results=5):
    """Heuristic molecular formula search from an accurate (neutral) mass: brute-force over
    C/N/O/P/S counts in a small-molecule/peptide-scale range, solving for H analytically at
    each combination, then filtering by degree-of-unsaturation (DBE) and H/C ratio sanity —
    a simplified version of the standard 'seven golden rules' formula-validity heuristics
    used by real formula-generation tools. This is a mass-only plausibility check, not a
    spectral-library match, so multiple candidates near the same mass are expected; they are
    ranked only by mass error (ppm), not by chemical likelihood."""
    if target_mass is None or target_mass <= 0 or target_mass > 1500:
        return []

    m = LC_MS_ELEMENT_MASSES
    tol_da = target_mass * tolerance_ppm / 1e6
    c_max = min(80, int(target_mass / m['C']) + 1)
    n_max = min(10, int(target_mass / m['N']) + 1)
    o_max = min(25, int(target_mass / m['O']) + 1)
    p_max = min(4, int(target_mass / m['P']) + 1)
    s_max = min(4, int(target_mass / m['S']) + 1)

    candidates = []
    for c in range(0, c_max + 1):
        base_c = c * m['C']
        if base_c > target_mass + tol_da:
            break
        for n in range(0, n_max + 1):
            base_cn = base_c + n * m['N']
            if base_cn > target_mass + tol_da:
                break
            for o in range(0, o_max + 1):
                base_cno = base_cn + o * m['O']
                if base_cno > target_mass + tol_da:
                    break
                for p in range(0, p_max + 1):
                    base_cnop = base_cno + p * m['P']
                    if base_cnop > target_mass + tol_da:
                        break
                    for s in range(0, s_max + 1):
                        base = base_cnop + s * m['S']
                        if base > target_mass + tol_da:
                            break
                        h = round((target_mass - base) / m['H'])
                        if h < 0 or h > 150 or (c == 0 and h == 0):
                            continue
                        mass = base + h * m['H']
                        error_da = mass - target_mass
                        if abs(error_da) > tol_da:
                            continue
                        dbe = c - h / 2 + n / 2 + 1
                        if dbe < 0 or dbe > 40 or (dbe * 2) % 1 != 0:
                            continue
                        if c > 0 and not (0.1 <= h / c <= 3.2):
                            continue
                        candidates.append({
                            'formula': lc_ms_format_formula(c, h, n, o, p, s),
                            'mass': round(mass, 4),
                            'error_ppm': round(1e6 * error_da / target_mass, 2),
                            'dbe': dbe,
                        })

    candidates.sort(key=lambda cand: abs(cand['error_ppm']))
    return candidates[:max_results]


def lc_ms_extract_peak_table(files, max_peaks=20):
    """Looks for an m/z (or 'mass') column plus an intensity/area column in each selected
    tabular file — the feature/peak-list export format LC-MS software typically produces,
    distinct from the plain retention-time-vs-intensity chromatogram the Chromatogram tab
    plots. Returns the max_peaks most intense rows across all selected files, plus any
    per-file errors (e.g. no recognizable m/z column)."""
    rows = []
    errors = []
    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()
        try:
            df = read_tabular_file(
                filepath, ext,
                delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'),
                header_row_override=f.parse_header_row,
            )
        except Exception as e:
            errors.append(f"{f.original_filename}: {e}")
            continue

        mz_col = next((c for c in df.columns if re.search(r'm.?/?.?z|^mass$|monoisotopic', str(c), re.I)), None)
        if not mz_col:
            errors.append(f"{f.original_filename}: no m/z column detected (looked for a header containing 'm/z' or 'mass').")
            continue
        inten_col = next((c for c in df.columns if re.search(r'intensity|area|height|abundance', str(c), re.I)), None)
        rt_col = next((c for c in df.columns if re.search(r'^rt$|retention', str(c), re.I)), None)

        mz_vals = pd.to_numeric(df[mz_col], errors='coerce')
        inten_vals = pd.to_numeric(df[inten_col], errors='coerce') if inten_col is not None else pd.Series([None] * len(df))
        rt_vals = pd.to_numeric(df[rt_col], errors='coerce') if rt_col is not None else pd.Series([None] * len(df))

        for i in range(len(df)):
            mz = mz_vals.iloc[i]
            if pd.isna(mz) or mz <= 0:
                continue
            rows.append({
                'file_label': f.label or f.original_filename,
                'mz': float(mz),
                'intensity': float(inten_vals.iloc[i]) if pd.notna(inten_vals.iloc[i]) else None,
                'rt': float(rt_vals.iloc[i]) if pd.notna(rt_vals.iloc[i]) else None,
            })

    rows.sort(key=lambda r: r['intensity'] if r['intensity'] is not None else -1, reverse=True)
    return rows[:max_peaks], errors


def lc_ms_parse_csv_lines(text, second_col_numeric_only=True):
    """Parses 'a,b' per line (one pair per line, comma-separated) into (a, b) tuples,
    skipping blank or malformed lines — used for the Quantification tab's plain-text
    standards/unknowns entry, the same 'one item per line' convention Protocol steps use
    elsewhere in this app."""
    pairs = []
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            second = float(parts[1])
        except ValueError:
            continue
        first = parts[0]
        if second_col_numeric_only:
            try:
                first = float(first)
            except ValueError:
                continue
        pairs.append((first, second))
    return pairs


# Standard monoisotopic residue masses (Da) — amino acid mass minus the water lost when it
# joins a peptide chain. Used for in-silico tryptic digestion / peptide mass fingerprinting
# (PMF) against a user-supplied candidate sequence. Free cysteine (no alkylation) and no
# post-translational modifications — a plain baseline, same spirit as this app's other
# 'starting interpretation, not definitive' reference tables.
AA_MONO_MASS = {
    'G': 57.02146, 'A': 71.03711, 'S': 87.03203, 'P': 97.05276, 'V': 99.06841,
    'T': 101.04768, 'C': 103.00919, 'L': 113.08406, 'I': 113.08406, 'N': 114.04293,
    'D': 115.02694, 'Q': 128.05858, 'K': 128.09496, 'E': 129.04259, 'M': 131.04049,
    'H': 137.05891, 'F': 147.06841, 'R': 156.10111, 'Y': 163.06333, 'W': 186.07931,
}
WATER_MASS = 18.010565
PROTON_MASS = 1.007276


def tryptic_digest(sequence, missed_cleavages=1, min_length=4, max_length=50):
    """In-silico trypsin digest: cleaves after K or R, except when followed by P (the
    standard trypsin specificity exception). Generates every fragment allowed by the given
    number of missed cleavages, same convention real PMF search tools use."""
    seq = re.sub(r'[^A-Za-z]', '', sequence or '').upper()
    if not seq:
        return [], seq

    sites = [0]
    for i, aa in enumerate(seq):
        if aa in ('K', 'R') and (i + 1 >= len(seq) or seq[i + 1] != 'P'):
            sites.append(i + 1)
    if sites[-1] != len(seq):
        sites.append(len(seq))
    sites = sorted(set(sites))

    peptides = []
    for i in range(len(sites) - 1):
        for j in range(i + 1, min(i + 2 + missed_cleavages, len(sites))):
            start, end = sites[i], sites[j]
            pep_seq = seq[start:end]
            if min_length <= len(pep_seq) <= max_length:
                peptides.append({'sequence': pep_seq, 'start': start + 1, 'end': end})
    return peptides, seq


def peptide_mono_mass(pep_seq):
    try:
        return sum(AA_MONO_MASS[aa] for aa in pep_seq) + WATER_MASS
    except KeyError:
        return None


def match_pmf(peptides, observed_mz, tolerance_da=0.3):
    """Matches each theoretical tryptic peptide's [M+H]+ mass against the observed MALDI
    peak list within a Da tolerance (MALDI PMF conventionally uses an absolute Da/mDa
    tolerance, not ppm, since instrument resolution is roughly constant across the mass
    range used here). Returns matches plus the set of covered sequence positions."""
    matches = []
    covered = set()
    for pep in peptides:
        mass = peptide_mono_mass(pep['sequence'])
        if mass is None:
            continue
        theo_mz = mass + PROTON_MASS
        best = None
        for obs in observed_mz:
            err = obs - theo_mz
            if abs(err) <= tolerance_da and (best is None or abs(err) < abs(best[1])):
                best = (obs, err)
        if best:
            matches.append({
                'sequence': pep['sequence'], 'start': pep['start'], 'end': pep['end'],
                'theo_mass': round(mass, 4), 'theo_mz': round(theo_mz, 4),
                'observed_mz': best[0], 'error_da': round(best[1], 4),
            })
            covered.update(range(pep['start'], pep['end'] + 1))
    matches.sort(key=lambda m: m['start'])
    return matches, covered


def maldi_extract_pixel_table(files):
    """Looks for x, y, and intensity-like columns in each selected tabular file — the
    per-pixel ion-intensity export a MALDI imaging workflow would produce for one extracted
    ion, distinct from the plain spectrum the Plot Spectrum tab plots. Returns one entry per
    file (not merged, since each file is one ion's image) plus any per-file errors."""
    images = []
    errors = []
    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()
        try:
            df = read_tabular_file(
                filepath, ext,
                delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'),
                header_row_override=f.parse_header_row,
            )
        except Exception as e:
            errors.append(f"{f.original_filename}: {e}")
            continue

        x_col = next((c for c in df.columns if re.fullmatch(r'x|x.?pos(ition)?|x.?coord(inate)?', str(c).strip(), re.I)), None)
        y_col = next((c for c in df.columns if re.fullmatch(r'y|y.?pos(ition)?|y.?coord(inate)?', str(c).strip(), re.I)), None)
        inten_col = next((c for c in df.columns if re.search(r'intensity|abundance|count', str(c), re.I)), None)

        if not (x_col is not None and y_col is not None and inten_col is not None):
            errors.append(f"{f.original_filename}: needs x, y, and intensity columns for an ion image (found: {', '.join(str(c) for c in df.columns)}).")
            continue

        x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
        y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
        inten = pd.to_numeric(df[inten_col], errors='coerce').to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(inten))
        x, y, inten = x[mask], y[mask], inten[mask]
        if len(x) < 4:
            errors.append(f"{f.original_filename}: fewer than 4 valid (x, y, intensity) rows — not enough for an image.")
            continue

        images.append({'file_label': f.label or f.original_filename, 'x': x, 'y': y, 'intensity': inten})
    return images, errors


def integrate_chromatogram_peaks(x, y, prominence=None, min_height=None):
    """Detects peaks, then integrates each one valley-to-valley — area bounded by the local
    minimum on either side (or the data edge for the first/last peak), the standard baseline
    convention real chromatography integrators use, rather than an arbitrary height cutoff.
    Also reports the standard column-performance metrics computed straight from each peak's
    shape: theoretical plates N (from the half-height width) and USP tailing factor (from
    the 5%-height front/back asymmetry), both via scipy's peak_widths."""
    idx, _ = find_peaks(y, prominence=prominence, height=min_height)
    if len(idx) == 0:
        return []

    idx = sorted(idx)
    n = len(y)
    xi = np.arange(n)

    try:
        widths_half, _, left_half, right_half = peak_widths(y, idx, rel_height=0.5)
        _, _, left_5pct, right_5pct = peak_widths(y, idx, rel_height=0.95)
    except Exception:
        left_half = right_half = left_5pct = right_5pct = [None] * len(idx)

    peaks = []
    total_area = 0.0
    for i, apex in enumerate(idx):
        left_bound = 0 if i == 0 else idx[i - 1] + int(np.argmin(y[idx[i - 1]:apex + 1]))
        right_bound = n - 1 if i == len(idx) - 1 else apex + int(np.argmin(y[apex:idx[i + 1] + 1]))
        seg_x, seg_y = x[left_bound:right_bound + 1], y[left_bound:right_bound + 1]
        # Drop-line baseline: a straight line between the two valley levels, each taken as the
        # median of a few neighbouring points. Valleys are picked as the *lowest* point between
        # peaks, which under noise is biased low — using that single point (and clipping
        # negatives) inflated areas on noisy data.
        k = max(1, min(5, n // 200))
        level_left = float(np.median(y[max(0, left_bound - k):left_bound + k + 1]))
        level_right = float(np.median(y[max(0, right_bound - k):right_bound + k + 1]))
        span = seg_x[-1] - seg_x[0]
        drop_line = level_left + (level_right - level_left) * ((seg_x - seg_x[0]) / span if span else 0)
        baseline = float(np.interp(x[apex], seg_x, drop_line))
        area = max(float(np.trapezoid(seg_y - drop_line, seg_x)), 0.0)

        w_half = n_plates = tailing = None
        if left_half[i] is not None:
            left_x = float(np.interp(left_half[i], xi, x))
            right_x = float(np.interp(right_half[i], xi, x))
            w_half = right_x - left_x
            if w_half > 0:
                n_plates = 5.54 * (x[apex] / w_half) ** 2
        if left_5pct[i] is not None:
            front = x[apex] - float(np.interp(left_5pct[i], xi, x))
            back = float(np.interp(right_5pct[i], xi, x)) - x[apex]
            if front > 0:
                tailing = (front + back) / (2 * front)

        peaks.append({
            'apex_idx': apex, 'rt': float(x[apex]), 'height': float(y[apex] - baseline),
            'area': area, 'w_half': w_half,
            'n_plates': round(n_plates) if n_plates else None,
            'tailing': round(tailing, 2) if tailing else None,
        })
        total_area += area

    for p in peaks:
        p['pct_area'] = round(100 * p['area'] / total_area, 2) if total_area > 0 else None
    peaks[0]['resolution'] = None
    for i in range(1, len(peaks)):
        w1, w2 = peaks[i - 1]['w_half'], peaks[i]['w_half']
        peaks[i]['resolution'] = round(2 * (peaks[i]['rt'] - peaks[i - 1]['rt']) / (w1 + w2), 2) if (w1 and w2) else None

    return peaks


def parse_compound_library(text):
    """Parses the Peak Integration compound library: one line per standard injection, as
    'name,RT' for identification only, or 'name,RT,concentration,area' to also contribute one
    calibration point for that compound — repeat the same name across several lines (one per
    standard level) to build its own calibration curve, mirroring the classic HPLC workflow of
    running a blank, then a multi-level standard series, before the unknown sample: identify
    each sample peak by matching its retention time to a standard, then quantify it using that
    specific compound's own calibration curve (not a single curve shared across every peak)."""
    entries = []
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            rt = float(parts[1])
        except ValueError:
            continue
        conc = area = None
        if len(parts) >= 4:
            try:
                conc = float(parts[2])
                area = float(parts[3])
            except ValueError:
                pass
        entries.append({'name': name, 'rt': rt, 'conc': conc, 'area': area})
    return entries


def build_compound_calibrations(entries):
    """Fits a per-compound linear calibration (area vs concentration) from whichever library
    entries include calibration points — needs at least 2 same-named points to fit a line."""
    by_name = {}
    for e in entries:
        if e['conc'] is not None and e['area'] is not None:
            by_name.setdefault(e['name'], []).append((e['conc'], e['area']))
    calibrations = {}
    for name, pts in by_name.items():
        if len(pts) >= 2:
            conc = np.array([p[0] for p in pts])
            area = np.array([p[1] for p in pts])
            if np.ptp(conc) > 0:
                slope, intercept = np.polyfit(conc, area, 1)
                calibrations[name] = {'slope': float(slope), 'intercept': float(intercept), 'n': len(pts)}
    return calibrations


def match_retention_library(peaks, entries, tolerance_min=0.2, calibrations=None):
    """Labels each integrated peak with the nearest compound name from a user-supplied
    retention-time library if within tolerance — this app has no real compound database, so
    identification here is only ever as good as the reference RTs the user brings in
    themselves, run on their own column under their own conditions. If that compound has its
    own fitted calibration, also back-calculates a concentration from the peak's area."""
    name_rts = {}
    for e in entries:
        name_rts.setdefault(e['name'], []).append(e['rt'])
    name_rt = {name: float(np.mean(rts)) for name, rts in name_rts.items()}
    calibrations = calibrations or {}

    for p in peaks:
        best = None
        for name, rt in name_rt.items():
            diff = abs(p['rt'] - rt)
            if diff <= tolerance_min and (best is None or diff < best[1]):
                best = (name, diff)
        p['name'] = best[0] if best else None
        cal = calibrations.get(p['name']) if p['name'] else None
        if cal and cal['slope']:
            p['concentration'] = round((p['area'] - cal['intercept']) / cal['slope'], 4)
            p['cal_n'] = cal['n']
        else:
            p['concentration'] = None
            p['cal_n'] = None
    return peaks


def assign_peaks(peaks, table):
    """Matches each (x, y) peak against a reference range table, returns list of (x, y, description)."""
    assigned = []
    for x, y in peaks:
        match = next((desc for lo, hi, desc in table if lo <= x <= hi), None)
        assigned.append((x, y, match))
    return assigned


def synthesize_ftir_interpretation(assigned):
    """Goes past 'this peak = this bond' to what the peaks mean *together* — the actual
    reasoning a chemist does: is a carbonyl an ester or an acid (depends on whether an O-H
    is also present), is a substituted-benzene pattern mono- or para- (depends on exactly
    where its out-of-plane C-H band falls), etc. Returns None if nothing present is
    distinctive enough to reason about, rather than forcing a comment."""
    bands = {'oh_nh': [], 'ester_co': [], 'ketone_co': [], 'amide_co': [], 'c_o_c_n': [], 'aromatic_oop': []}
    for x, y, d in assigned:
        if 3200 <= x <= 3550:
            bands['oh_nh'].append(x)
        if 1735 <= x <= 1750:
            bands['ester_co'].append(x)
        if 1700 <= x <= 1725:
            bands['ketone_co'].append(x)
        if 1650 <= x <= 1700:
            bands['amide_co'].append(x)
        if 1000 <= x <= 1300:
            bands['c_o_c_n'].append(x)
        if 650 <= x <= 900:
            bands['aromatic_oop'].append(x)

    has_oh = bool(bands['oh_nh'])
    has_carbonyl = bool(bands['ester_co'] or bands['ketone_co'] or bands['amide_co'])
    parts = []

    if bands['ester_co'] and not has_oh:
        parts.append(f"The carbonyl near {bands['ester_co'][0]:.0f} cm⁻¹ sits in the ester range, and there's no broad O–H stretch above 3200 cm⁻¹ to go with it — that combination points to an ester rather than a carboxylic acid (a free acid would show both bands).")
    elif bands['ketone_co'] and not has_oh:
        parts.append(f"The carbonyl near {bands['ketone_co'][0]:.0f} cm⁻¹ falls in the ketone/aldehyde range without an accompanying O–H stretch, which rules out a carboxylic acid at this position.")
    elif has_carbonyl and has_oh:
        parts.append("Both a carbonyl and a broad O–H/N–H stretch are present — this combination is what a carboxylic acid looks like, though it's equally consistent with an ester/ketone/amide sitting alongside a separate, unrelated hydroxyl group (residual solvent, moisture, or an -OH elsewhere on the molecule). The C-O stretch pattern and exact O-H shape (sharp vs. very broad) are what actually distinguish these.")
    elif has_oh and not has_carbonyl:
        parts.append("A broad O–H/N–H stretch with no carbonyl anywhere in the spectrum reads as an alcohol, amine, or simply adsorbed moisture — not a carbonyl-containing functional group.")

    if bands['amide_co']:
        parts.append(f"The band near {bands['amide_co'][0]:.0f} cm⁻¹ sits in the amide I region — if there's also an N–H stretch around 3300 cm⁻¹ and an amide II band (N–H bend/C–N stretch) near 1550 cm⁻¹, that trio is the signature of an amide bond or peptide backbone rather than a plain alkene, which only overlaps this one band.")

    if bands['aromatic_oop']:
        oop = bands['aromatic_oop'][0]
        if 690 <= oop <= 730:
            pattern = "monosubstituted (a companion band near 730–770 cm⁻¹ would confirm it) or meta-disubstituted"
        elif 730 < oop <= 770:
            pattern = "ortho-disubstituted, or monosubstituted if there's a companion band near 690–710 cm⁻¹"
        elif 800 <= oop <= 860:
            pattern = "para-disubstituted"
        else:
            pattern = "substituted"
        parts.append(f"The aromatic C–H out-of-plane bend at {oop:.0f} cm⁻¹ is consistent with a {pattern} benzene ring — this region (675–900 cm⁻¹) is where substitution pattern actually gets read, not just 'aromatic present.'")

    return " ".join(parts) if parts else None


def synthesize_nmr_interpretation(assigned):
    """The chemical-shift table already names each peak's likely proton environment — this
    looks at which broad regions are occupied *together* and reasons about the molecule as a
    whole, the way you'd actually read a 1H spectrum rather than looking up each shift alone."""
    regions = {'aromatic': (6.5, 8.5), 'vinyl': (4.5, 6.5), 'carbinol_alpha_co': (3.3, 4.5),
               'heteroatom_ch': (2.0, 3.3), 'aliphatic': (0.5, 2.0)}
    present = {name: [] for name in regions}
    for x, y, d in assigned:
        for name, (lo, hi) in regions.items():
            if lo <= x <= hi:
                present[name].append(x)

    parts = []
    if present['aromatic'] and present['aliphatic'] and not present['vinyl']:
        parts.append("Signals in both the aromatic region (6.5–8.5 ppm) and the aliphatic region (0.5–2.0 ppm), with nothing in the vinyl range, fit an aromatic ring attached to a saturated alkyl chain rather than an extended conjugated/olefinic system.")
    elif present['vinyl'] and not present['aromatic']:
        parts.append("Signals in the vinyl region (4.5–6.5 ppm) without any aromatic signals point to an isolated alkene rather than an aromatic ring.")
    elif present['aromatic'] and present['vinyl']:
        parts.append("Both aromatic (6.5–8.5 ppm) and vinyl (4.5–6.5 ppm) signals are present — consistent with a conjugated system (e.g. a styrene-type vinyl group) linking the two, rather than two unrelated environments.")

    if present['heteroatom_ch'] and not present['aromatic']:
        parts.append("Signals in the 2.0–3.3 ppm range with no aromatic peaks suggest CH adjacent to a carbonyl, halogen, or other electronegative group on an otherwise non-aromatic backbone.")
    if present['carbinol_alpha_co']:
        parts.append(f"The peak(s) near {present['carbinol_alpha_co'][0]:.2f} ppm sit where CH next to oxygen (e.g. O-CH2, OCH3) or a carbonyl typically appears — the exact shift and any coupling pattern (best read directly off the spectrum, not from position alone) distinguish which.")

    if len(present['aliphatic']) >= 3 and not present['aromatic'] and not present['vinyl']:
        parts.append(f"All {len(present['aliphatic'])} detected peaks fall in the aliphatic region (0.5–2.0 ppm) — consistent with a purely saturated, non-aromatic, non-olefinic structure.")

    return " ".join(parts) if parts else None


def _integrated_analysis(file_notes, overall=None, group_label=None):
    """One integrated write-up covering every selected file, instead of a separate card
    per file plus a summary bolted on at the end — comparison/trend context belongs in the
    same narrative as the individual readings, not tacked on afterward. A single selected
    file still gets its own plain note; only 2+ files get the 'across N files' framing."""
    if not file_notes:
        return [{'label': overall and 'Interpretation' or 'No data', 'analysis': overall or "Nothing to analyze."}] if overall else []
    if len(file_notes) == 1:
        text = file_notes[0][1]
        if overall:
            text += " " + overall
        return [{'label': file_notes[0][0], 'analysis': text}]
    sentences = []
    for label, note in file_notes:
        note = note.strip()
        if note and note[-1] not in '.!?':
            note += '.'
        sentences.append(f"{label} — {note}")
    text = f"Across the {len(file_notes)} selected files: " + " ".join(sentences)
    if overall:
        text += " " + overall
    return [{'label': group_label or f"All {len(file_notes)} files", 'analysis': text}]


def generate_spectroscopy_analysis(technique_name, series_list):
    """Produces a genuine technique-specific interpretation: peak assignments for
    vibrational/NMR/CD techniques, or computed quantities (λmax, band gap, Stokes shift)
    for UV-Vis/Fluorescence — combined into one integrated narrative across whatever files
    are selected, rather than a separate block per file. Falls back to basic stats if the
    technique isn't specially handled."""
    if not series_list:
        return []

    file_notes = []
    overall = None

    if technique_name == 'FTIR':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.1 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, FTIR_TABLE)
            lines = [f"{x:.0f} cm⁻¹ → {desc}" if desc else f"{x:.0f} cm⁻¹ → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} peak(s) detected. " + ("; ".join(lines) + "." if lines else "No significant peaks found.")
            interpretation = synthesize_ftir_interpretation(assigned)
            if interpretation:
                text += " " + interpretation
            file_notes.append((s['label'], text))

    elif technique_name == 'Raman':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.1 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, RAMAN_TABLE)
            lines = [f"{x:.0f} cm⁻¹ → {desc}" if desc else f"{x:.0f} cm⁻¹ → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} peak(s) detected. " + ("; ".join(lines) + "." if lines else "No significant peaks found.")
            d_peak = next((x for x, y, d in assigned if d and 'D band' in d), None)
            g_peak = next((x for x, y, d in assigned if d and 'G band' in d), None)
            if d_peak and g_peak:
                d_intensity = next(y for x, y, d in assigned if x == d_peak)
                g_intensity = next(y for x, y, d in assigned if x == g_peak)
                id_ig = d_intensity / g_intensity if g_intensity else None
                if id_ig is not None:
                    text += f" ID/IG ratio ≈ {id_ig:.2f} — {'higher disorder' if id_ig > 1 else 'more graphitic/ordered'} carbon structure."
            file_notes.append((s['label'], text))

    elif technique_name == 'NMR (1H, 13C)':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.1 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, NMR_1H_TABLE)
            lines = [f"{x:.2f} ppm → {desc}" if desc else f"{x:.2f} ppm → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} peak(s) detected (assuming 1H shifts). " + ("; ".join(lines) + "." if lines else "No significant peaks found.")
            interpretation = synthesize_nmr_interpretation(assigned)
            if interpretation:
                text += " " + interpretation
            file_notes.append((s['label'], text))

    elif technique_name == 'CD (Circular Dichroism)':
        for s in series_list:
            idx, _ = find_peaks(np.abs(s['y']), prominence=(max(np.abs(s['y'])) - min(np.abs(s['y']))) * 0.15 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, CD_TABLE)
            lines = [f"{x:.0f} nm ({'+' if y >= 0 else '−'}) → {desc}" if desc else f"{x:.0f} nm ({'+' if y >= 0 else '−'}) → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} band(s) detected. " + ("; ".join(lines) + "." if lines else "No significant bands found.")
            has_208 = any(203 <= x <= 212 and y < 0 for x, y, d in assigned)
            has_222 = any(218 <= x <= 225 and y < 0 for x, y, d in assigned)
            if has_208 and has_222:
                text += " The double minima near 208 nm and 222 nm are a classic α-helix signature."
            file_notes.append((s['label'], text))

    elif technique_name == 'UV-Vis':
        for s in series_list:
            peak_idx = int(np.argmax(s['y']))
            lam_max = float(s['x'][peak_idx])
            peak_intensity = float(s['y'][peak_idx])
            text = f"λmax ≈ {lam_max:.1f} nm (absorbance {peak_intensity:.3g})."
            if 190 <= lam_max <= 1100:
                gap_ev = 1240 / lam_max
                text += f" Approximate optical transition energy ≈ {gap_ev:.2f} eV (E = 1240/λmax — a rough estimate, not a substitute for Tauc analysis)."
            file_notes.append((s['label'], text))

    elif technique_name == 'Fluorescence':
        peak_positions = []
        for s in series_list:
            peak_idx = int(np.argmax(s['y']))
            lam_em = float(s['x'][peak_idx])
            peak_positions.append((s['label'], lam_em, float(s['y'][peak_idx])))
            file_notes.append((s['label'], f"emission maximum ≈ {lam_em:.1f} nm (intensity {s['y'][peak_idx]:.3g})"))

        if len(peak_positions) >= 2:
            lam_values = [p[1] for p in peak_positions]
            shift = max(lam_values) - min(lam_values)
            if shift > 2:
                direction = "red-shifted" if peak_positions[-1][1] > peak_positions[0][1] else "blue-shifted"
                overall = f"Emission maxima span {shift:.1f} nm across samples — later samples appear {direction} relative to the first, which can indicate changes in the local environment, conjugation, or aggregation state."

    elif technique_name == 'EDS/EDX':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.05 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, EDS_TABLE)
            matched = [(x, y, d) for x, y, d in assigned if d]
            if matched:
                total = sum(y for _, y, _ in matched)
                comp_lines = sorted(matched, key=lambda t: t[1], reverse=True)
                comp_text = "; ".join(
                    f"{d} at {x:.2f} keV (~{100 * y / total:.1f}% of identified signal)" for x, y, d in comp_lines
                )
                text = f"{len(peaks)} peak(s) detected, {len(matched)} matched to an element. {comp_text}."
            else:
                text = f"{len(peaks)} peak(s) detected but none matched a known characteristic X-ray line within tolerance."
            file_notes.append((s['label'], text))
        overall = ("Peak identification only, from standard reference line energies — not ZAF-corrected "
                   "quantification. Treat the % as a rough relative-abundance guide, not certified composition.")

    elif technique_name == 'MALDI':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.05 or None)
            masses = s['x'][idx]
            intensities = s['y'][idx]
            if len(masses) >= 3:
                # Mn/Mw/PDI from the peak list treated as an intensity-weighted mass distribution —
                # the standard molecular-weight-average formulas, applied directly to MALDI peaks
                # rather than a chromatographic elution profile (how GPC/SEC gets the same numbers).
                mn = float(np.sum(intensities * masses) / np.sum(intensities))
                mw = float(np.sum(intensities * masses ** 2) / np.sum(intensities * masses))
                pdi = mw / mn if mn else None
                spread = "narrow (near-monodisperse)" if pdi and pdi < 1.05 else (
                    "moderately broad" if pdi and pdi < 1.2 else "broad")
                text = (f"{len(masses)} peak(s) detected across {masses.min():.1f}–{masses.max():.1f} Da. "
                        f"Mn ≈ {mn:.1f} Da, Mw ≈ {mw:.1f} Da, PDI (Mw/Mn) ≈ {pdi:.3f} — a {spread} mass distribution.")
            else:
                text = f"{len(masses)} peak(s) detected — need at least 3 resolved peaks to estimate Mn/Mw/PDI."
            file_notes.append((s['label'], text))
        overall = "Treats MALDI peak intensities as relative population counts — a common approximation, though MALDI ionization efficiency isn't perfectly uniform across mass, so this skews toward better-ionizing species."

    elif technique_name == 'HPLC / GC':
        for s in series_list:
            peaks = integrate_chromatogram_peaks(s['x'], s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.05 or None)
            if peaks:
                total_area = sum(p['area'] for p in peaks)
                top = sorted(peaks, key=lambda p: p['area'], reverse=True)[:3]
                top_text = "; ".join(f"{p['rt']:.2f} min ({p['pct_area']}% area)" for p in top)
                n_vals = [p['n_plates'] for p in peaks if p['n_plates']]
                plates_text = f" Median theoretical plates ≈ {int(np.median(n_vals))}." if n_vals else ""
                text = f"{len(peaks)} peak(s) integrated, total area {total_area:.4g}. Largest: {top_text}.{plates_text}"
            else:
                text = "No peaks detected at the current sensitivity — check Peak Picking settings."
            file_notes.append((s['label'], text))
        overall = "See Peak Integration for the full per-peak table, including resolution and tailing factor."

    elif technique_name == 'LC-MS':
        for s in series_list:
            peaks = integrate_chromatogram_peaks(s['x'], s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.05 or None)
            if peaks:
                total_area = sum(p['area'] for p in peaks)
                top = sorted(peaks, key=lambda p: p['area'], reverse=True)[:3]
                top_text = "; ".join(f"{p['rt']:.2f} min ({p['pct_area']}% area)" for p in top)
                text = f"{len(peaks)} peak(s) integrated, total area {total_area:.4g}. Largest: {top_text}."
            else:
                text = "No peaks detected at the current sensitivity — check Peak Picking settings."
            file_notes.append((s['label'], text))
        overall = "Retention-time peaks only — for mass-based compound identification from these peaks, use Formula ID."

    elif technique_name == 'Cyclic Voltammetry (CV)':
        for s in series_list:
            m = compute_cv_metrics(s['x'], s['y'])
            text = (f"Ipa ≈ {m['ipa']:.4g} at Epa ≈ {m['epa']:.4g}; Ipc ≈ {m['ipc']:.4g} at Epc ≈ {m['epc']:.4g}. "
                    f"ΔEp ≈ {m['delta_ep']*1000:.1f} mV, E1/2 ≈ {m['e_half']:.4g}.")
            if m['ratio'] is not None:
                text += f" Ipa/Ipc ≈ {m['ratio']:.2f}."
            file_notes.append((s['label'], text))
        if len(series_list) >= 2:
            overall = build_electrochemical_analysis(series_list)

    elif technique_name == 'UTM / Nanoindentation':
        for s in series_list:
            x, y = s['x'], s['y']
            n = len(x)
            n_lin = max(5, int(n * 0.15))
            parts = []
            try:
                slope, _ = np.polyfit(x[:n_lin], y[:n_lin], 1)
                parts.append(f"elastic modulus (slope of the initial ~{100 * n_lin / n:.0f}% of the curve, assumed linear-elastic) ≈ {slope:.4g}")
            except Exception:
                pass
            uts_idx = int(np.argmax(y))
            uts, uts_x = float(y[uts_idx]), float(x[uts_idx])
            parts.append(f"peak stress (UTS) ≈ {uts:.4g} at strain ≈ {uts_x:.4g}")
            if uts_idx < n - 1:
                drop = uts - float(y[-1])
                parts.append(f"stress falls by {drop:.4g} after the peak (to {float(y[-1]):.4g} at the final recorded point), consistent with necking or fracture beyond the UTS")
            else:
                parts.append("no post-peak softening visible in the recorded range — may be cut off before fracture")
            try:
                toughness = float(np.trapezoid(y, x))
                parts.append(f"toughness (area under the curve) ≈ {toughness:.4g}")
            except Exception:
                pass
            file_notes.append((s['label'], "; ".join(parts) + "."))

    elif technique_name == 'Rheometer':
        for s in series_list:
            x, y = s['x'], s['y']
            mask = (x > 0) & (y > 0)
            if mask.sum() >= 3:
                log_x, log_y = np.log10(x[mask]), np.log10(y[mask])
                slope, intercept = np.polyfit(log_x, log_y, 1)
                n_flow = slope + 1
                k = 10 ** intercept
                if n_flow < 0.9:
                    behavior = f"shear-thinning (pseudoplastic, n ≈ {n_flow:.2f})"
                elif n_flow > 1.1:
                    behavior = f"shear-thickening (dilatant, n ≈ {n_flow:.2f})"
                else:
                    behavior = f"approximately Newtonian (n ≈ {n_flow:.2f})"
                text = f"power-law fit: K ≈ {k:.4g}, n ≈ {n_flow:.2f} — {behavior}."
            else:
                text = "not enough positive-valued points to fit a power-law flow curve."
            file_notes.append((s['label'], text))
        overall = "Assumes X = shear rate, Y = viscosity, the most common single flow-curve setup — if this is actually a frequency sweep of storage/loss modulus, this power-law fit isn't meaningful."

    elif technique_name == 'XPS':
        for s in series_list:
            x, y = s['x'], s['y']
            bg = shirley_background(x, y)
            y_sub = np.clip(y - bg, 0, None)
            components, _fitted = fit_multi_gaussian_peaks(x, y_sub, n_peaks=3)
            if not components:
                file_notes.append((s['label'], "peak fit did not converge — try Peak Fitting directly with fewer components."))
                continue
            lines = []
            for c in components:
                assigned = next((desc for lo, hi, desc in XPS_TABLE if lo <= c['center'] <= hi), None)
                line = f"{c['center']:.1f} eV ({c['pct_area']}% area, FWHM {c['fwhm']:.2f} eV)"
                line += f" → {assigned}" if assigned else " → unassigned"
                lines.append(line)
            file_notes.append((s['label'], f"Shirley-background peak fit (3 components): " + "; ".join(lines) + "."))
        overall = ("Chemical-state assignments come from a fixed binding-energy reference table and are ambiguous without "
                   "knowing which element's core level was actually scanned — treat them as a starting hypothesis, not "
                   "a confirmed identification. Adjust the component count directly in Peak Fitting if 3 doesn't match what's really there.")

    return _integrated_analysis(file_notes, overall)



def tech_detect_peaks(series_list, prominence, min_height):
    """Runs scipy's peak finder on each series, returns {label: [(x, y), ...]}."""
    peak_results = {}
    for s in series_list:
        y = s['y']
        x = s['x']
        try:
            idx, _ = find_peaks(y, prominence=prominence if prominence else None, height=min_height if min_height else None)
            peak_results[s['label']] = [(float(x[i]), float(y[i])) for i in idx]
        except Exception:
            peak_results[s['label']] = []
    return peak_results


def tech_build_wide_series(state):
    """For 'wide format' matrix files (e.g. fluorescence EEM: one X column + many Y columns,
    one per emission wavelength) — treats the first usable column as X and EVERY other numeric
    column as its own series, all from the same file. Uses a continuous colormap keyed by the
    column's own numeric value (e.g. wavelength) when column headers are numeric, since a
    standard legend becomes unreadable with dozens of series."""
    files = dp_selected_files(state)
    series_list = []
    errors = []
    numeric_column_values = []   # for colormap scaling, if headers are numeric

    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()
        try:
            df = read_tabular_file(filepath, ext, delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'), header_row_override=f.parse_header_row)
        except Exception as e:
            errors.append(f"{f.original_filename}: {e}")
            continue

        columns = list(df.columns)
        if len(columns) < 2:
            errors.append(f"{f.original_filename}: needs at least 2 columns for wide-format plotting.")
            continue

        x_col = columns[0]
        x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()

        for y_col in columns[1:]:
            y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                continue
            xv, yv = x[mask], y[mask]
            order = np.argsort(xv)
            xv, yv = xv[order], yv[order]

            label = f"{y_col}" if len(files) == 1 else f"{f.original_filename}: {y_col}"
            col_numeric = extract_numeric_label(str(y_col))
            series_list.append({'file_id': f.id, 'x': xv, 'y': yv, 'yerr': None, 'label': label,
                                 'x_col': x_col, 'y_col': y_col, 'color_value': col_numeric})
            if col_numeric is not None:
                numeric_column_values.append(col_numeric)

    return series_list, errors, numeric_column_values



def save_tech_plot_files(fig, prefix='tech'):
    """Saves a rendered technique plot in every export format the Plot tab offers a
    download link for, and returns the PNG's filename (the others are derived from it
    by extension in the template)."""
    base = f"{prefix}_{int(datetime.now().timestamp() * 1000)}"
    plot_filename = f"{base}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.svg"))
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.pdf"))
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.tiff"), dpi=300)
    return plot_filename


# Plot types offered for every technique in addition to whatever technique-specific modes
# TECHNIQUE_PLOT_MODES lists (line_spectrum/calibration_scatter/contour_2d etc.) — generic
# ways to look at the same selected series that aren't tied to any one measurement type.
GENERIC_PLOT_MODES = [
    ('scatter', 'Scatter Plot'),
    ('histogram', 'Histogram (distribution of Y values)'),
    ('bar', 'Bar Chart (mean ± std per file)'),
    ('heatmap', 'Heatmap (all files stacked, color = Y)'),
]


def tech_render_spectrum_plot(state, mark_peaks=False):
    """Renders a technique's plot. Most plot_type values are per-series (line/scatter/
    histogram, drawn either overlaid or as one subplot per file); calibration_scatter adds
    a linear fit; bar and heatmap summarize all files on a single shared axes instead (a
    per-file panel wouldn't make sense for either); contour_2d is an honest placeholder for
    2D data (not supported by our flat X/Y column model)."""
    plot_type = state.get('plot_type', 'line_spectrum')

    if plot_type == 'contour_2d':
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.text(0.5, 0.5, "2D contour/heatmap data (e.g. COSY/HSQC) needs a full 2D intensity\nmatrix, not flat X/Y columns — not supported by this tool yet.",
                ha='center', va='center', fontsize=11, color='#888', transform=ax.transAxes, wrap=True)
        ax.axis('off')
        fig.tight_layout()
        plot_filename = save_tech_plot_files(fig)
        plt.close(fig)
        return plot_filename, [], [], {}

    numeric_color_values = []
    if state.get('wide_mode'):
        series_list, errors, numeric_color_values = tech_build_wide_series(state)
    else:
        series_list, errors = dp_build_series(state)

    if not series_list:
        return None, [], errors, {}

    fmt = state['format']

    if plot_type == 'heatmap':
        results = [{'label': s['label'], 'stats': compute_series_stats(s['y']),
                    'analysis': build_stats_analysis(s['label'], compute_series_stats(s['y']), plot_type)} for s in series_list]
        if len(series_list) < 2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Heatmap needs 2 or more selected files.", ha='center', va='center', color='#888', transform=ax.transAxes)
            ax.axis('off')
        else:
            x_lo = min(float(s['x'].min()) for s in series_list)
            x_hi = max(float(s['x'].max()) for s in series_list)
            grid = np.linspace(x_lo, x_hi, 300)
            matrix = np.array([np.interp(grid, s['x'], s['y']) for s in series_list])
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(series_list) + 2)))
            cmap_name = fmt.get('colormap', 'default')
            im = ax.imshow(matrix, aspect='auto', cmap=cmap_name if cmap_name != 'default' else 'viridis',
                            extent=[grid[0], grid[-1], len(series_list), 0])
            ax.set_yticks(np.arange(len(series_list)) + 0.5)
            ax.set_yticklabels([s['label'] for s in series_list], fontsize=8)
            ax.set_xlabel(fmt.get('x_label') or 'X', fontsize=fmt['label_size'])
            fig.colorbar(im, ax=ax, label=fmt.get('y_label') or 'Y')
        fig.tight_layout()
        plot_filename = save_tech_plot_files(fig)
        plt.close(fig)
        return plot_filename, results, errors, {}

    if plot_type == 'bar':
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(series_list)), 5.5))
        colors = get_series_colors(fmt.get('colormap', 'default'), len(series_list))
        means = [float(np.mean(s['y'])) for s in series_list]
        stds = [float(np.std(s['y'])) for s in series_list]
        ax.bar(range(len(series_list)), means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(range(len(series_list)))
        ax.set_xticklabels([s['label'] for s in series_list], rotation=30, ha='right', fontsize=8)
        label_weight = 'bold' if fmt.get('bold_labels') else 'normal'
        ax.set_xlabel(fmt.get('x_label') or 'X', fontsize=fmt['label_size'], fontweight=label_weight)
        ax.set_ylabel(fmt.get('y_label') or 'Y', fontsize=fmt['label_size'], fontweight=label_weight)
        ax.tick_params(width=fmt['tick_width'])
        if fmt.get('grid', True):
            ax.grid(alpha=0.25, axis='y')
        results = [{'label': s['label'], 'stats': compute_series_stats(s['y']),
                    'analysis': build_stats_analysis(s['label'], compute_series_stats(s['y']), plot_type)} for s in series_list]
        fig.tight_layout()
        plot_filename = save_tech_plot_files(fig)
        plt.close(fig)
        return plot_filename, results, errors, {}

    if state.get('wide_mode') and len(numeric_color_values) == len(series_list) and len(series_list) > 8:
        # many series with numeric labels (e.g. emission wavelengths) — a standard legend would be
        # unreadable, so color by the series' own value using a continuous colormap with a colorbar instead
        cmap = plt.get_cmap('viridis')
        vmin, vmax = min(numeric_color_values), max(numeric_color_values)
        norm = lambda v: (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        colors = [matplotlib.colors.to_hex(cmap(norm(v))) for v in numeric_color_values]
        use_colorbar = True
    else:
        colors = get_series_colors(fmt.get('colormap', 'default'), len(series_list))
        use_colorbar = False

    n = len(series_list)
    # Small multiples instead of one overlay axes — one panel per file, titled with its
    # label instead of a shared legend. Not offered for the colorbar case (already a
    # many-series continuous-color view, not a good fit for per-file panels).
    use_subplots = state.get('layout') == 'subplots' and n > 1 and not use_colorbar
    if use_subplots:
        ncols = min(3, n)
        nrows = -(-n // ncols)  # ceil division
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
        axes_flat = list(axes.flatten())
        for extra_ax in axes_flat[n:]:
            extra_ax.axis('off')  # unused grid cells when n doesn't fill the grid evenly
        target_axes = axes_flat[:n]
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        target_axes = [ax] * n

    results = []
    peaks_by_label = {}

    if plot_type == 'calibration_scatter':
        for i, s in enumerate(series_list):
            tax = target_axes[i]
            tax.scatter(s['x'], s['y'], s=fmt['marker_size'], color=colors[i], alpha=0.8, label=s['label'])
            try:
                popt, _ = curve_fit(linear_fn, s['x'], s['y'])
                x_smooth = np.linspace(min(s['x']), max(s['x']), 200)
                tax.plot(x_smooth, linear_fn(x_smooth, *popt), color=colors[i], linestyle='--', linewidth=fmt['line_width'])
                y_pred = linear_fn(s['x'], *popt)
                ss_res = np.sum((s['y'] - y_pred) ** 2)
                ss_tot = np.sum((s['y'] - np.mean(s['y'])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
                analysis = f"Linear fit: y = {popt[0]:.4g}x + {popt[1]:.4g}, R² = {r_squared:.4f}." if r_squared is not None else "Linear fit could not be scored."
            except Exception:
                analysis = "Linear fit did not converge for this series."
            if use_subplots:
                tax.set_title(s['label'], fontsize=9)
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'analysis': analysis})
    elif plot_type == 'scatter':
        for i, s in enumerate(series_list):
            tax = target_axes[i]
            tax.scatter(s['x'], s['y'], s=fmt['marker_size'], color=colors[i], alpha=0.8, label=s['label'])
            if use_subplots:
                tax.set_title(s['label'], fontsize=9)
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'analysis': build_stats_analysis(s['label'], stats, plot_type)})
    elif plot_type == 'histogram':
        for i, s in enumerate(series_list):
            tax = target_axes[i]
            tax.hist(s['y'], bins=min(30, max(5, len(s['y']) // 3)), color=colors[i], alpha=0.65 if not use_subplots else 0.9, edgecolor='white', label=s['label'])
            if use_subplots:
                tax.set_title(s['label'], fontsize=9)
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'analysis': build_stats_analysis(s['label'], stats, plot_type)})
    else:  # line_spectrum (default)
        smoothing = state.get('smoothing', {})
        if smoothing.get('enabled'):
            for s in series_list:
                window = min(smoothing.get('window', 11), len(s['y']) - (1 - len(s['y']) % 2))
                if window >= 5 and window % 2 == 1 and window <= len(s['y']):
                    polyorder = min(smoothing.get('polyorder', 3), window - 1)
                    try:
                        s['y'] = savgol_filter(s['y'], window_length=window, polyorder=polyorder)
                    except Exception:
                        pass  # fall back to raw data if smoothing fails on this series

        peaks_by_label = tech_detect_peaks(series_list, state['peaks'].get('prominence'), state['peaks'].get('min_height')) if mark_peaks else {}
        for i, s in enumerate(series_list):
            tax = target_axes[i]
            tax.plot(s['x'], s['y'], color=colors[i], linewidth=fmt['line_width'], label=s['label'])
            if fmt.get('fill_under'):
                tax.fill_between(s['x'], 0, s['y'], color=colors[i], alpha=0.25, zorder=0)
            if mark_peaks and peaks_by_label.get(s['label']):
                px = [p[0] for p in peaks_by_label[s['label']]]
                py = [p[1] for p in peaks_by_label[s['label']]]
                tax.scatter(px, py, color=colors[i], marker='v', s=60, edgecolor='black', zorder=5)
            if use_subplots:
                tax.set_title(s['label'], fontsize=9)
            stats = compute_series_stats(s['y'])
            n_peaks = len(peaks_by_label.get(s['label'], []))
            note = f"{n_peaks} peak(s) detected." if mark_peaks else None
            analysis = build_stats_analysis(s['label'], stats, plot_type)
            if note:
                analysis = note + " " + analysis
            results.append({'label': s['label'], 'stats': stats, 'analysis': analysis})

    label_weight = 'bold' if fmt.get('bold_labels') else 'normal'
    unique_axes = list(dict.fromkeys(target_axes))  # de-dupe while preserving order (overlay reuses one ax)
    # A histogram's axes mean something different from every other plot type here: its X is
    # the data's Y-values (binned), and its Y is a count — so the custom X/Y label text swaps
    # sides, with a hardcoded 'Count' rather than the user's Y label.
    x_axis_label = (fmt.get('y_label') or 'Y') if plot_type == 'histogram' else (fmt.get('x_label') or 'X')
    y_axis_label = 'Count' if plot_type == 'histogram' else (fmt.get('y_label') or 'Y')
    for a in unique_axes:
        a.set_xlabel(x_axis_label, fontsize=fmt['label_size'], fontweight=label_weight)
        a.set_ylabel(y_axis_label, fontsize=fmt['label_size'], fontweight=label_weight)
        a.tick_params(width=fmt['tick_width'])
        if fmt.get('log_x'):
            a.set_xscale('log')
        if fmt.get('log_y'):
            a.set_yscale('log')

        # explicit axis range (zoom), applied after scale type is set
        x_min, x_max = fmt.get('x_min'), fmt.get('x_max')
        if x_min is not None or x_max is not None:
            cur_min, cur_max = a.get_xlim()
            a.set_xlim(x_min if x_min is not None else cur_min, x_max if x_max is not None else cur_max)

        y_min, y_max = fmt.get('y_min'), fmt.get('y_max')
        if y_min is not None or y_max is not None:
            cur_min, cur_max = a.get_ylim()
            a.set_ylim(y_min if y_min is not None else cur_min, y_max if y_max is not None else cur_max)

        if fmt.get('grid', True):
            a.grid(alpha=0.25)

    if use_colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=unique_axes[0])
        cbar.set_label('Emission wavelength' if state.get('wide_mode') else 'Series value')
    elif fmt.get('legend', True) and not use_subplots:
        ncol = min(len(series_list), 3) if fmt.get('legend_orientation') == 'horizontal' and len(series_list) > 3 else (len(series_list) if fmt.get('legend_orientation') == 'horizontal' else 1)
        unique_axes[0].legend(fontsize=9 * fmt.get('legend_scale', 1.0), loc=fmt.get('legend_loc', 'best'), ncol=ncol, framealpha=0.9)

    fig.tight_layout()
    plot_filename = save_tech_plot_files(fig)
    plt.close(fig)

    return plot_filename, results, errors, peaks_by_label


def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005) — a genuinely computed
    smooth curve fit under the data, not a peak-detection heuristic. `lam` controls
    smoothness (higher = stiffer baseline), `p` controls asymmetry (smaller p pulls the
    baseline down to sit under peaks rather than through their middle). Standard method
    for FTIR/Raman baseline removal; needs no new dependency beyond scipy.sparse."""
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 5:
        return y.copy()
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * (D @ D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(max(1, int(niter))):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def render_baseline_comparison(series_list, baselines, correcteds, out_dir):
    """One panel per file (raw / baseline / corrected overlaid) — always small multiples,
    since overlaying several files' raw+baseline+corrected triples on one axes would be
    unreadable clutter regardless of how many files are selected."""
    n = len(series_list)
    if n > 1:
        ncols = min(3, n)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
        axes_flat = list(axes.flatten())
        for extra_ax in axes_flat[n:]:
            extra_ax.axis('off')
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        axes_flat = [ax]

    for i, s in enumerate(series_list):
        a = axes_flat[i]
        a.plot(s['x'], s['y'], color='#999', linewidth=1.1, label='Raw')
        a.plot(s['x'], baselines[i], color='#e53e3e', linestyle='--', linewidth=1.3, label='Baseline')
        a.plot(s['x'], correcteds[i], color='#2b6cb0', linewidth=1.4, label='Corrected')
        a.set_title(s['label'], fontsize=9)
        a.set_xlabel('X')
        a.set_ylabel('Y')
        a.grid(alpha=0.25)
        a.legend(fontsize=8)

    fig.tight_layout()
    filename = f"baseline_{int(datetime.now().timestamp() * 1000)}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], filename), dpi=130)
    plt.close(fig)
    return filename


def xps_peak_fit_params(args, saved=None):
    """Reads n_peaks/background/file_id from the request, falling back to whatever was
    last used in Peak Fitting (persisted in session state) rather than hardcoded defaults
    — so visiting Format and seeing the fitted preview shows the fit actually in progress,
    not a generic 2-peak default that doesn't match what's on the Peak Fitting tab."""
    saved = saved or {}

    def _i(name, default):
        try:
            return int(float(args.get(name, default)))
        except (TypeError, ValueError):
            return default

    n_peaks = max(1, min(6, _i('n_peaks', saved.get('n_peaks', 2))))
    background = args.get('background', saved.get('background', 'shirley'))
    if background not in ('shirley', 'linear', 'none'):
        background = 'shirley'
    file_id = args.get('file_id', type=int) or saved.get('file_id')
    return {'n_peaks': n_peaks, 'background': background, 'file_id': file_id}


def render_xps_peak_fit(x, y, bg, components, fitted_curve, out_dir, tag='', fmt=None):
    """This is the figure that actually goes in a paper, so it respects the same Format
    tab (line width, colors, legend, axis labels/fonts, grid, log/range) as every other
    plot in the app, instead of being stuck on hardcoded defaults — and exports to the
    same PNG/SVG/PDF/TIFF set."""
    fmt = fmt or {}
    line_width = fmt.get('line_width', 1.6)
    label_size = fmt.get('label_size', 11)
    label_weight = 'bold' if fmt.get('bold_labels') else 'normal'
    x_label = fmt.get('x_label') or 'Binding Energy (eV)'
    y_label = fmt.get('y_label') or 'Intensity'
    colormap = fmt.get('colormap', 'default')
    n_series_for_color = max(len(components) if components else 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x, y, color='#333', linewidth=line_width, label='Raw spectrum')
    if bg is not None:
        ax.plot(x, bg, color='#999', linestyle='--', linewidth=max(line_width * 0.7, 0.5), label='Background')
    if components and fitted_curve is not None:
        colors = get_series_colors(colormap, n_series_for_color)
        base = bg if bg is not None else np.zeros_like(x)
        for i, c in enumerate(components):
            comp_curve = c['amplitude'] * np.exp(-((x - c['center']) ** 2) / (2 * c['sigma'] ** 2)) + base
            ax.plot(x, comp_curve, color=colors[i], linewidth=line_width, label=f"Component @ {c['center']:.1f} eV")
            ax.fill_between(x, base, comp_curve, color=colors[i], alpha=0.15)
        ax.plot(x, fitted_curve + base, color='#e53e3e', linestyle=':', linewidth=line_width * 1.15, label='Fit sum')

    ax.set_xlabel(x_label, fontsize=label_size, fontweight=label_weight)
    ax.set_ylabel(y_label, fontsize=label_size, fontweight=label_weight)
    ax.tick_params(width=fmt.get('tick_width', 1.0))
    if fmt.get('log_y'):
        ax.set_yscale('log')
    # log_x is skipped on purpose: XPS binding energy is never plotted log-scale, and doing
    # so would fight with the axis-invert below.

    x_min, x_max = fmt.get('x_min'), fmt.get('x_max')
    y_min, y_max = fmt.get('y_min'), fmt.get('y_max')
    if x_min is not None or x_max is not None:
        cur_min, cur_max = min(x[0], x[-1]), max(x[0], x[-1])
        ax.set_xlim(x_max if x_max is not None else cur_max, x_min if x_min is not None else cur_min)
    elif x[0] < x[-1]:
        ax.invert_xaxis()  # XPS convention: binding energy decreasing left to right
    if y_min is not None or y_max is not None:
        cur_min, cur_max = ax.get_ylim()
        ax.set_ylim(y_min if y_min is not None else cur_min, y_max if y_max is not None else cur_max)

    if fmt.get('grid', True):
        ax.grid(alpha=0.25)
    if fmt.get('legend', True):
        ncol = min(3, len(ax.get_legend_handles_labels()[0])) if fmt.get('legend_orientation') == 'horizontal' else 1
        ax.legend(fontsize=9 * fmt.get('legend_scale', 1.0), loc=fmt.get('legend_loc', 'best'), ncol=ncol, framealpha=0.9)

    fig.tight_layout()
    plot_filename = save_tech_plot_files(fig, prefix=f"xps_fit_{tag}")
    plt.close(fig)
    return plot_filename


@app.route('/characterizations/data')
def data_interpretation():
    category_names = [c[0] for c in TECHNIQUE_CATEGORIES]
    active_category = request.args.get('category', category_names[0])
    if active_category not in category_names:
        active_category = category_names[0]

    techniques = dict(TECHNIQUE_CATEGORIES)[active_category]
    active_technique = request.args.get('technique', techniques[0])
    if active_technique not in techniques:
        active_technique = techniques[0]

    technique_slugs = {t: slugify_technique(t) for t in techniques}

    return render_template(
        'data_interpretation.html',
        page_title='Data Interpretation',
        categories=TECHNIQUE_CATEGORIES,
        active_category=active_category,
        techniques=techniques,
        active_technique=active_technique,
        technique_slugs=technique_slugs,
        banner_image='images/characterizations-banner.png',
    )


@app.route('/computational')
def computational_home():
    techniques = dict(COMPUTATIONAL_CATEGORIES)['Computational']
    technique_slugs = {t: slugify_technique(t) for t in techniques}
    return render_template(
        'computational.html',
        page_title='Computational',
        techniques=techniques,
        technique_slugs=technique_slugs,
        banner_image='images/characterizations-banner.png',
    )


def confocal_load_gray(data_file):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    return np.array(Image.open(filepath).convert('L'), dtype=float)


def confocal_preprocess_params(args):
    steps = [s for s in args.getlist('steps') if s in confocal.PREPROCESS_STEPS]

    def _f(name, default):
        try:
            return float(args.get(name, default))
        except (TypeError, ValueError):
            return default

    def _i(name, default):
        try:
            return int(float(args.get(name, default)))
        except (TypeError, ValueError):
            return default

    denoise_method = args.get('denoise_method', 'gaussian')
    if denoise_method not in confocal.DENOISE_METHODS:
        denoise_method = 'gaussian'

    params = {
        'bg_radius': _f('bg_radius', 25),
        'flatfield_sigma': _f('flatfield_sigma', 50),
        'denoise_method': denoise_method,
        'denoise_amount': _f('denoise_amount', 1.0),
        'psf_sigma': _f('psf_sigma', 2.0),
        'deconv_iterations': _i('deconv_iterations', 15),
    }
    return steps, params


def confocal_segment_params(args):
    def _f(name, default):
        try:
            return float(args.get(name, default))
        except (TypeError, ValueError):
            return default

    def _i(name, default):
        try:
            return int(float(args.get(name, default)))
        except (TypeError, ValueError):
            return default

    return {
        'min_distance': _i('min_distance', 10),
        'threshold_offset': _f('threshold_offset', 0.0),
    }


def tech_baseline_params(args):
    def _f(name, default):
        try:
            return float(args.get(name, default))
        except (TypeError, ValueError):
            return default

    def _i(name, default):
        try:
            return int(float(args.get(name, default)))
        except (TypeError, ValueError):
            return default

    return {
        'lam': _f('lam', 100000.0),
        'p': _f('p', 0.01),
        'niter': _i('niter', 10),
    }


@app.route('/characterizations/data/technique/<slug>')
def technique_workspace(slug):
    technique_name = TECHNIQUE_SLUGS.get(slug)
    if not technique_name:
        return redirect(url_for('data_interpretation'))

    tabs = TECHNIQUE_TABS.get(technique_name, DEFAULT_TECHNIQUE_TABS)
    active_tab = request.args.get('tab', tabs[0])
    if active_tab not in tabs:
        active_tab = tabs[0]

    parent_category = next((cat for cat, techs in ALL_TECHNIQUE_CATEGORIES if technique_name in techs), None)

    if parent_category == 'Computational':
        job_running = _comp_collect_job(slug)
        comp_key = f'comp_state_{slug}'
        comp_state = session.get(comp_key, {})
        return render_template(
            'technique_workspace_computational.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            mc_integrals=computational.MC_INTEGRALS,
            dft_basis_options=computational.DFT_BASIS_OPTIONS,
            dft_functional_options=computational.DFT_FUNCTIONAL_OPTIONS,
            result=comp_state.get(active_tab),
            comp_error=session.pop('comp_error', None),
            job_running=job_running,
            banner_image='images/characterizations-banner.png',
        )

    # Spectroscopy techniques get the fully wired workflow; others still show the placeholder for now.
    # EDS/EDX is grouped under Microscopy & Imaging (it's acquired alongside SEM/TEM imaging) but is
    # itself a spectrum (counts vs energy), and LC-MS/MALDI/HPLC-GC are grouped under Mass & Separation
    # but their base spectrum/chromatogram view is also just a spectrum-shaped X/Y plot — all of them
    # reuse this same pipeline, then layer their own extra tabs on top.
    if parent_category in ('Spectroscopy', 'Mechanics & Electrochemistry') or technique_name in ('EDS/EDX', 'LC-MS', 'MALDI', 'HPLC / GC', 'XPS'):
        state = tech_get_state(slug)
        all_files = DataFile.query.filter_by(file_type='tabular', technique_name=technique_name, user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected_files = [f for f in all_files if f.id in state['file_ids']]
        plot_modes = TECHNIQUE_PLOT_MODES.get(slug, [('line_spectrum', 'Line Graph')]) + GENERIC_PLOT_MODES
        plot_tab_name = spectrum_plot_tab_name(technique_name)

        # XPS is the one technique with two genuinely different plots (the raw spectrum vs
        # the fitted-components figure) — each gets its own independent Format settings,
        # picked via ?target=. Every other technique only ever has the one plot.
        format_target = request.args.get('target', 'plot') if technique_name == 'XPS' else 'plot'
        if format_target == 'xps_fit':
            display_format = state.get('xps_fit_format') or default_tech_format()
        else:
            format_target = 'plot'
            display_format = state['format']

        plot_filename, results, plot_errors, peaks_by_label = (None, [], [], {})
        if active_tab in (plot_tab_name, 'Peak Picking', 'PMF', 'Peak Integration', 'Baseline Correction', 'Derivative', 'Peak Fitting', 'Format', 'Analysis') and state['file_ids']:
            mark_peaks = (active_tab in ('Peak Picking', 'PMF', 'Peak Integration', 'Analysis'))
            plot_filename, results, plot_errors, peaks_by_label = tech_render_spectrum_plot(state, mark_peaks=mark_peaks)

        spectroscopy_analysis = []
        if active_tab == 'Analysis' and state['file_ids'] and state.get('plot_type') != 'contour_2d':
            if state.get('wide_mode'):
                analysis_series, _, _ = tech_build_wide_series(state)
            else:
                analysis_series, _ = dp_build_series(state)
            spectroscopy_analysis = generate_spectroscopy_analysis(technique_name, analysis_series)

        baseline_result = None
        if active_tab == 'Baseline Correction' and state['file_ids']:
            baseline_params = tech_baseline_params(request.args)
            baseline_series, baseline_errors = dp_build_series(state)
            if baseline_series:
                baselines = [baseline_als(s['y'], **baseline_params) for s in baseline_series]
                correcteds = [s['y'] - b for s, b in zip(baseline_series, baselines)]
                plot_filename_baseline = render_baseline_comparison(baseline_series, baselines, correcteds, app.config['UPLOAD_FOLDER'])
                baseline_result = {
                    'params': baseline_params, 'plot_filename': plot_filename_baseline,
                    'errors': baseline_errors, 'n_files': len(baseline_series),
                }
            else:
                baseline_result = {'params': baseline_params, 'plot_filename': None, 'errors': baseline_errors, 'n_files': 0}

        peak_fit_result = None
        if technique_name == 'XPS' and active_tab in ('Peak Fitting', 'Format') and state['file_ids']:
            pf_params = xps_peak_fit_params(request.args, saved=state.get('xps_peak_fit'))
            pf_series, pf_errors = dp_build_series(state)
            if pf_series:
                target = next((s for s in pf_series if s['file_id'] == pf_params['file_id']), pf_series[0])
                pf_params['file_id'] = target['file_id']
                x, y = target['x'], target['y']
                if pf_params['background'] == 'shirley':
                    bg = shirley_background(x, y)
                elif pf_params['background'] == 'linear':
                    bg = np.linspace(y[0], y[-1], len(y))
                else:
                    bg = np.zeros_like(y)
                y_sub = np.clip(y - bg, 0, None)
                components, fitted_curve = fit_multi_gaussian_peaks(x, y_sub, pf_params['n_peaks'])
                xps_fit_fmt = state.get('xps_fit_format') or default_tech_format()
                plot_filename_pf = render_xps_peak_fit(x, y, bg, components, fitted_curve, app.config['UPLOAD_FOLDER'], tag=str(target['file_id']), fmt=xps_fit_fmt)
                peak_fit_result = {
                    'params': pf_params, 'target_file_id': target['file_id'], 'target_label': target['label'],
                    'components': components, 'plot_filename': plot_filename_pf,
                    'series_options': [(s['file_id'], s['label']) for s in pf_series], 'errors': pf_errors,
                }
                # Peak Fitting is the tab with the controls — only it should overwrite what
                # Format then just reads back, so tweaking Format never silently changes the fit.
                if active_tab == 'Peak Fitting':
                    state['xps_peak_fit'] = pf_params
                    tech_save_state(slug, state)
            else:
                peak_fit_result = {'params': pf_params, 'components': None, 'plot_filename': None, 'series_options': [], 'errors': pf_errors}

        formula_id = None
        if technique_name == 'LC-MS' and active_tab == 'Formula ID':
            adduct = request.args.get('adduct', 'M+H')
            if adduct not in LC_MS_ADDUCTS:
                adduct = 'M+H'
            try:
                tolerance_ppm = float(request.args.get('tolerance_ppm', 10))
            except ValueError:
                tolerance_ppm = 10.0
            peaks, peak_errors = lc_ms_extract_peak_table(selected_files) if selected_files else ([], [])
            adduct_shift = LC_MS_ADDUCTS[adduct][1]
            for peak in peaks:
                neutral_mass = peak['mz'] - adduct_shift
                peak['neutral_mass'] = round(neutral_mass, 4)
                peak['candidates'] = generate_formula_candidates(neutral_mass, tolerance_ppm)
            formula_id = {
                'adduct': adduct, 'tolerance_ppm': tolerance_ppm,
                'peaks': peaks, 'errors': peak_errors, 'adducts': LC_MS_ADDUCTS,
            }

        quant_result = state.get('quant_result') if technique_name in ('LC-MS', 'HPLC / GC') else None

        quant_file_options = []
        if technique_name in ('LC-MS', 'HPLC / GC') and active_tab == 'Quantification' and state['file_ids']:
            quant_file_options = build_quant_file_options(state)

        pmf_result = None
        if technique_name == 'MALDI' and active_tab == 'PMF':
            pmf_settings = state.get('pmf', {'sequence': '', 'missed_cleavages': 1, 'tolerance_da': 0.3})
            # observed masses are each peak's X position (m/z); peaks_by_label is {label: [(x, y), ...]}
            observed_mz = sorted({round(px, 4) for peaks in peaks_by_label.values() for px, py in peaks})
            peptides, clean_seq = tryptic_digest(pmf_settings['sequence'], pmf_settings['missed_cleavages'])
            matches, covered = match_pmf(peptides, observed_mz, pmf_settings['tolerance_da'])
            coverage_pct = round(100 * len(covered) / len(clean_seq), 1) if clean_seq else 0.0
            pmf_result = {
                'settings': pmf_settings, 'sequence_length': len(clean_seq),
                'n_theoretical': len(peptides), 'n_observed': len(observed_mz),
                'matches': matches, 'coverage_pct': coverage_pct,
            }

        imaging_result = None
        if technique_name == 'MALDI' and active_tab == 'Imaging':
            images, imaging_errors = maldi_extract_pixel_table(selected_files) if selected_files else ([], [])
            plot_paths = []
            for img in images:
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(img['x'], img['y'], c=img['intensity'], cmap='inferno', s=40, marker='s')
                ax.set_xlabel('X'); ax.set_ylabel('Y')
                ax.set_aspect('equal', adjustable='box')
                ax.invert_yaxis()
                fig.colorbar(sc, ax=ax, label='Intensity')
                fig.tight_layout()
                img_filename = f"maldi_ion_image_{int(datetime.now().timestamp())}_{len(plot_paths)}.png"
                fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], img_filename), dpi=130)
                plt.close(fig)
                plot_paths.append({'file_label': img['file_label'], 'plot_filename': img_filename, 'n_pixels': len(img['x'])})
            imaging_result = {'images': plot_paths, 'errors': imaging_errors}

        compare_ctx = None
        if active_tab == 'Compare' and technique_name in simulate.SIM_SPECS:
            compare_ctx = build_compare_context(all_files, state.get('compare'))
            if compare_ctx['result']:
                plot_filename = compare_ctx['result']['plot_filename']

        sim_spec = None
        simulated_files = []
        if active_tab == 'Simulate' and technique_name in simulate.SIM_SPECS:
            sim_spec = sim_spec_for(technique_name)
            simulated_files = [{'file': f, 'key': json.loads(f.simulation_key or '[]')} for f in all_files if f.is_simulated]

        integration_result = None
        if technique_name == 'HPLC / GC' and active_tab == 'Peak Integration':
            rt_library_text = state.get('rt_library_text', '')
            library_entries = parse_compound_library(rt_library_text)
            calibrations = build_compound_calibrations(library_entries)
            all_peaks = []
            if state.get('wide_mode'):
                series_list, _, _ = tech_build_wide_series(state)
            else:
                series_list, _ = dp_build_series(state)
            for s in series_list:
                peaks = integrate_chromatogram_peaks(
                    s['x'], s['y'],
                    prominence=state['peaks'].get('prominence'),
                    min_height=state['peaks'].get('min_height'),
                )
                if library_entries:
                    peaks = match_retention_library(peaks, library_entries, calibrations=calibrations)
                all_peaks.append({'label': s['label'], 'peaks': peaks})
            integration_result = {'series': all_peaks, 'rt_library_text': rt_library_text, 'calibrations': calibrations}

        return render_template(
            'technique_workspace_spectroscopy.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            state=state,
            all_files=all_files,
            selected_files=selected_files,
            plot_modes=plot_modes,
            plot_tab_name=plot_tab_name,
            plot_filename=plot_filename,
            results=results,
            plot_errors=plot_errors,
            peaks_by_label=peaks_by_label,
            spectroscopy_analysis=spectroscopy_analysis,
            baseline_result=baseline_result,
            peak_fit_result=peak_fit_result,
            format_target=format_target,
            display_format=display_format,
            formula_id=formula_id,
            quant_result=quant_result,
            quant_file_options=quant_file_options,
            pmf_result=pmf_result,
            imaging_result=imaging_result,
            integration_result=integration_result,
            compare_ctx=compare_ctx,
            sim_spec=sim_spec,
            sim_slugs=[slugify_technique(t) for t in simulate.SIM_SPECS],
            simulated_files=simulated_files,
            sim_error=session.pop('sim_error', None),
            colormap_options=COLORMAP_OPTIONS,
            banner_image='images/characterizations-banner.png',
        )

    if technique_name == 'AFM':
        state = tech_get_state(slug)
        all_files = DataFile.query.filter_by(technique_name='AFM', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected = [f for f in all_files if f.id in state['file_ids']]

        topography_results = []
        if active_tab == 'Topography':
            for f in [f for f in selected if f.file_type == 'image']:
                if f.parse_status == 'parsed_native' and f.height_data_filename:
                    arr = np.load(os.path.join(app.config['UPLOAD_FOLDER'], f.height_data_filename))
                    units = f.data_units or 'nm'
                    stats = {
                        'ra': float(np.mean(np.abs(arr - np.mean(arr)))),
                        'rq': float(np.std(arr)),
                        'range': float(np.max(arr) - np.min(arr)),
                        'units': units,
                        'real': True,
                    }
                else:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
                    try:
                        gray = np.array(Image.open(filepath).convert('L'))
                        crop = f.image_crop_bottom or gray.shape[0]
                        proxy = compute_intensity_roughness(gray, crop)
                        stats = {'ra': proxy['ra_proxy'], 'rq': proxy['rq_proxy'], 'range': None,
                                 'units': 'a.u. (intensity)', 'real': False}
                    except Exception as e:
                        topography_results.append({'file': f, 'error': str(e)})
                        continue
                topography_results.append({'file': f, 'stats': stats})

        force_curve_results = []
        if active_tab in ('Mechanical', 'Biological'):
            for f in [f for f in selected if f.channel_type == 'force_curve']:
                entry = {'file': f}
                if f.fit_params:
                    entry['results'] = json.loads(f.fit_params)
                force_curve_results.append(entry)

        channel_tab_types = {
            'Electrical': ('kpfm_potential', 'cafm_current', 'scm_capacitance', 'pfm_amplitude', 'pfm_phase'),
            'Magnetic': ('mfm_phase',),
            'Chemical / Frictional': ('lfm_friction',),
        }
        channel_results = []
        if active_tab in channel_tab_types:
            wanted = channel_tab_types[active_tab]
            for f in [f for f in selected if f.file_type == 'image' and f.channel_type in wanted]:
                try:
                    overlay = generate_channel_overlay(f)
                    channel_results.append({'file': f, **overlay})
                except Exception as e:
                    channel_results.append({'file': f, 'error': str(e)})

        afm_analyses = []
        if active_tab == 'Analysis' and selected:
            file_ids = [f.id for f in selected]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(file_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected}
            for a in analyses:
                afm_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id)})

        return render_template(
            'technique_workspace_afm.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            state=state,
            all_files=all_files,
            selected=selected,
            topography_results=topography_results,
            force_curve_results=force_curve_results,
            channel_results=channel_results,
            channel_type_info=CHANNEL_TYPE_INFO,
            afm_analyses=afm_analyses,
            banner_image='images/characterizations-banner.png',
        )

    if technique_name == 'TEM':
        state = tech_get_state(slug)
        all_images = DataFile.query.filter_by(file_type='image', technique_name='TEM', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected_images = [f for f in all_images if f.id in state['file_ids']]
        images_by_id = {f.id: f for f in selected_images}

        segment_result = None
        if active_tab == 'Segment' and selected_images:
            image_id = request.args.get('image_id', type=int)
            target = images_by_id.get(image_id) or selected_images[0]
            seg_params = confocal_segment_params(request.args)
            gray = confocal_load_gray(target)
            labels_image, mask, objects = confocal.segment_watershed(gray, **seg_params)
            plot_filename = confocal.render_segmentation_overlay(gray, labels_image, app.config['UPLOAD_FOLDER'], tag=str(target.id))
            segment_result = {'target': target, 'params': seg_params, 'objects': objects, 'plot_filename': plot_filename}

        defect_results = []
        if active_tab == 'Defects' and selected_images:
            image_ids = [f.id for f in selected_images]
            annotations = DefectAnnotation.query.filter(DefectAnnotation.file_id.in_(image_ids)).order_by(DefectAnnotation.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in annotations:
                defect_results.append({'annotation': a, 'file': files_by_id.get(a.file_id)})

        tem_analyses = []
        combined_particle_analyses = []
        if active_tab == 'Analysis' and selected_images:
            image_ids = [f.id for f in selected_images]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(image_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in analyses:
                extra_images = json.loads(a.extra_images_json) if a.extra_images_json else []
                tem_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id), 'extra_images': extra_images})
            combined_particle_analyses = combine_particle_analyses(tem_analyses)

        return render_template(
            'technique_workspace_tem.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            state=state,
            all_images=all_images,
            selected_images=selected_images,
            segment_result=segment_result,
            defect_types=DEFECT_TYPES,
            defect_results=defect_results,
            tem_analyses=tem_analyses,
            combined_particle_analyses=combined_particle_analyses,
            tem_error=session.pop('tem_error', None),
            banner_image='images/characterizations-banner.png',
        )

    if technique_name == 'Confocal / Fluorescence':
        state = tech_get_state(slug)
        all_images = DataFile.query.filter_by(file_type='image', technique_name=technique_name, user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected_images = [f for f in all_images if f.id in state['file_ids']]
        images_by_id = {f.id: f for f in selected_images}

        preprocess_result = None
        if active_tab == 'Preprocess' and selected_images:
            image_id = request.args.get('image_id', type=int)
            target = images_by_id.get(image_id) or selected_images[0]
            steps, params = confocal_preprocess_params(request.args)
            gray = confocal_load_gray(target)
            if steps:
                processed, log = confocal.run_preprocess(gray, steps, params)
            else:
                processed, log = gray, []
            plot_filename = confocal.render_preprocess_comparison(gray, processed, app.config['UPLOAD_FOLDER'], tag=str(target.id))
            preprocess_result = {'target': target, 'steps': steps, 'params': params, 'log': log, 'plot_filename': plot_filename}

        align_result = None
        align_error = None
        if active_tab == 'Preprocess' and len(selected_images) >= 2 and request.args.get('do_align'):
            reference = images_by_id.get(request.args.get('reference_id', type=int)) or selected_images[0]
            moving = images_by_id.get(request.args.get('moving_id', type=int)) or selected_images[1]
            if reference.id == moving.id:
                align_error = "Pick two different images to align one against the other."
            else:
                ref_gray = confocal_load_gray(reference)
                mov_gray = confocal_load_gray(moving)
                aligned, shift_yx, err = confocal.align_channels(ref_gray, mov_gray)
                plot_filename = confocal.render_alignment_comparison(
                    ref_gray, mov_gray, aligned, shift_yx, app.config['UPLOAD_FOLDER'], tag=f'{reference.id}-{moving.id}')
                align_result = {
                    'reference': reference, 'moving': moving, 'shift': shift_yx, 'error': err,
                    'plot_filename': plot_filename, 'aligned_array': aligned,
                }

        segment_result = None
        if active_tab == 'Segment' and selected_images:
            image_id = request.args.get('image_id', type=int)
            target = images_by_id.get(image_id) or selected_images[0]
            seg_params = confocal_segment_params(request.args)
            gray = confocal_load_gray(target)
            labels_image, mask, objects = confocal.segment_watershed(gray, **seg_params)
            plot_filename = confocal.render_segmentation_overlay(gray, labels_image, app.config['UPLOAD_FOLDER'], tag=str(target.id))
            segment_result = {'target': target, 'params': seg_params, 'objects': objects, 'plot_filename': plot_filename}

        particle_analyses = []
        combined_particle_analyses = []
        if active_tab == 'Analysis' and selected_images:
            image_ids = [f.id for f in selected_images]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(image_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in analyses:
                particle_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id)})
            combined_particle_analyses = combine_particle_analyses(particle_analyses)

        return render_template(
            'technique_workspace_confocal.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            state=state,
            all_images=all_images,
            selected_images=selected_images,
            preprocess_result=preprocess_result,
            align_result=align_result,
            align_error=align_error,
            segment_result=segment_result,
            particle_analyses=particle_analyses,
            combined_particle_analyses=combined_particle_analyses,
            preprocess_steps=confocal.PREPROCESS_STEPS,
            denoise_methods=confocal.DENOISE_METHODS,
            banner_image='images/characterizations-banner.png',
        )

    if parent_category == 'Microscopy & Imaging':
        state = tech_get_state(slug)
        all_images = DataFile.query.filter_by(file_type='image', technique_name=technique_name, user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected_images = [f for f in all_images if f.id in state['file_ids']]

        roughness_results = []
        if active_tab == 'Roughness' and technique_name in ('AFM', 'SEM') and selected_images:
            for f in selected_images:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
                try:
                    gray = np.array(Image.open(filepath).convert('L'))
                    crop = f.image_crop_bottom or gray.shape[0]
                    stats = compute_intensity_roughness(gray, crop)
                    roughness_results.append({'file': f, 'stats': stats})
                except Exception as e:
                    roughness_results.append({'file': f, 'error': str(e)})

        porosity_results = []
        if active_tab == 'Porosity' and selected_images:
            for f in selected_images:
                try:
                    result = generate_porosity_overlay(f)
                    porosity_results.append({'file': f, **result})
                except Exception as e:
                    porosity_results.append({'file': f, 'error': str(e)})

        # for the Analysis tab: pull in any particle measurements already done on the selected images
        particle_analyses = []
        combined_particle_analyses = []
        if active_tab == 'Analysis' and selected_images:
            image_ids = [f.id for f in selected_images]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(image_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in analyses:
                particle_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id)})
            combined_particle_analyses = combine_particle_analyses(particle_analyses)

        return render_template(
            'technique_workspace_microscopy.html',
            page_title=technique_name,
            technique_name=technique_name,
            slug=slug,
            tabs=tabs,
            active_tab=active_tab,
            parent_category=parent_category,
            state=state,
            all_images=all_images,
            selected_images=selected_images,
            roughness_results=roughness_results,
            porosity_results=porosity_results,
            particle_analyses=particle_analyses,
            combined_particle_analyses=combined_particle_analyses,
            channel_type_info=CHANNEL_TYPE_INFO,
            banner_image='images/characterizations-banner.png',
        )

    return render_template(
        'technique_workspace.html',
        page_title=technique_name,
        technique_name=technique_name,
        slug=slug,
        tabs=tabs,
        active_tab=active_tab,
        parent_category=parent_category,
        banner_image='images/characterizations-banner.png',
    )


def save_confocal_derived_image(source_file, array, suffix_label, notes):
    """Persists a processed grayscale array (Preprocess/Align output) as a new
    DataFile, so it flows back through Select images like any uploaded file."""
    arr = np.clip(array, 0, 255).astype(np.uint8)
    base_name = os.path.splitext(source_file.original_filename)[0]
    stored_filename = secure_filename(f"{datetime.now().timestamp()}_{suffix_label}_{base_name}.png")
    Image.fromarray(arr).save(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename))

    new_file = DataFile(
        user_id=session['user_id'],
        original_filename=f"{suffix_label}_{source_file.original_filename}",
        stored_filename=stored_filename,
        file_type='image',
        label=f"{source_file.label or source_file.original_filename} ({suffix_label})",
        technique_name=source_file.technique_name,
        pixel_size_nm=source_file.pixel_size_nm,
        imaging_notes=notes,
    )
    db.session.add(new_file)
    db.session.commit()
    return new_file


@app.route('/characterizations/data/confocal/<int:file_id>/preprocess/save', methods=['POST'])
def confocal_save_preprocess(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    steps, params = confocal_preprocess_params(request.form)

    if not steps:
        return redirect(url_for('technique_workspace', slug='confocal-fluorescence', tab='Preprocess', image_id=file_id))

    gray = confocal_load_gray(data_file)
    processed, log = confocal.run_preprocess(gray, steps, params)
    save_confocal_derived_image(data_file, processed, 'processed', '; '.join(log))

    return redirect(url_for('technique_workspace', slug='confocal-fluorescence', tab='Select images'))


@app.route('/characterizations/data/confocal/align/save', methods=['POST'])
def confocal_save_alignment():
    reference = get_owned_or_404(DataFile, request.form.get('reference_id', type=int))
    moving = get_owned_or_404(DataFile, request.form.get('moving_id', type=int))

    ref_gray = confocal_load_gray(reference)
    mov_gray = confocal_load_gray(moving)
    aligned, shift_yx, err = confocal.align_channels(ref_gray, mov_gray)
    notes = f"Aligned to '{reference.label or reference.original_filename}' via phase cross-correlation: shift=({shift_yx[0]:.2f}, {shift_yx[1]:.2f})px, registration error={err:.4f}"
    save_confocal_derived_image(moving, aligned, 'aligned', notes)

    return redirect(url_for('technique_workspace', slug='confocal-fluorescence', tab='Select images'))


@app.route('/characterizations/data/confocal/segment/save', methods=['POST'])
def confocal_save_segmentation():
    slug = request.form.get('slug', 'confocal-fluorescence')
    data_file = get_owned_or_404(DataFile, request.form.get('image_id', type=int))
    seg_params = confocal_segment_params(request.form)

    gray = confocal_load_gray(data_file)
    labels_image, mask, objects = confocal.segment_watershed(gray, **seg_params)

    if not objects:
        return redirect(url_for('technique_workspace', slug=slug, tab='Segment', image_id=data_file.id))

    scale = data_file.pixel_size_nm
    if scale:
        sizes = [o['equiv_diameter_px'] * scale for o in objects]
        unit = 'nm'
    else:
        sizes = [o['equiv_diameter_px'] for o in objects]
        unit = 'px'

    quantity_label = 'Particle diameter'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=min(20, max(5, len(sizes) // 2)), color='#2b6cb0', edgecolor='white')
    ax.axvline(np.mean(sizes), color='#e53e3e', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(sizes):.1f} {unit}')
    ax.set_xlabel(f'{quantity_label} ({unit})')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    hist_filename = f"particle_hist_{data_file.id}_{int(datetime.now().timestamp())}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], hist_filename), dpi=130)
    plt.close(fig)

    analysis_text = build_particle_size_analysis(sizes, unit, method_label='Automatic watershed detection')
    analysis_text += (
        f" Detected via Otsu thresholding + distance-transform watershed"
        f" (min separation={seg_params['min_distance']}px) — mean object"
        f" intensity across detections: {np.mean([o['mean_intensity'] for o in objects]):.1f}."
    )

    analysis = ImageAnalysis(
        file_id=data_file.id,
        pixel_size_nm=scale,
        sizes_nm_json=json.dumps(sizes),
        unit=unit,
        histogram_filename=hist_filename,
        analysis_text=analysis_text,
        measurement_type='particle_size',
        config_json=json.dumps({'method': 'watershed', 'objects': objects, 'params': seg_params}),
    )
    db.session.add(analysis)
    db.session.commit()

    return redirect(url_for('view_particle_analysis', analysis_id=analysis.id))


@app.route('/characterizations/data/technique/<slug>/select-files', methods=['POST'])
def tech_select_files(slug):
    state = tech_get_state(slug)
    selected = request.form.getlist('file_ids')
    new_file_ids = [int(i) for i in selected if i.isdigit()]

    if new_file_ids != state['file_ids']:
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            state['format'][key] = None
        # A previously fitted calibration's "unknowns" table is tied to the files that were
        # selected when it ran — keeping it around after the selection changes is exactly
        # what was overriding the Unknown-samples auto-fill with stale data every time.
        state.pop('quant_result', None)

    state['file_ids'] = new_file_ids
    tech_save_state(slug, state)
    plot_tab_name = spectrum_plot_tab_name(TECHNIQUE_SLUGS.get(slug))
    return redirect(url_for('technique_workspace', slug=slug, tab=plot_tab_name))


@app.route('/characterizations/data/technique/<slug>/select-images', methods=['POST'])
def tech_select_images(slug):
    state = tech_get_state(slug)
    selected = request.form.getlist('file_ids')
    state['file_ids'] = [int(i) for i in selected if i.isdigit()]
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Measure Particles'))


@app.route('/characterizations/data/technique/<slug>/select-afm-data', methods=['POST'])
def tech_select_afm_data(slug):
    state = tech_get_state(slug)
    selected = request.form.getlist('file_ids')
    state['file_ids'] = [int(i) for i in selected if i.isdigit()]
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Topography'))


@app.route('/characterizations/data/<int:file_id>/set-imaging-notes', methods=['POST'])
def set_imaging_notes(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    data_file.imaging_notes = request.form.get('imaging_notes', '').strip() or None
    db.session.commit()
    redirect_slug = request.form.get('slug', '').strip()
    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab='Biological'))
    return redirect(url_for('data_interpretation_workspace'))


@app.route('/characterizations/data/technique/<slug>/set-plot', methods=['POST'])
def tech_set_plot(slug):
    state = tech_get_state(slug)
    state['plot_type'] = request.form.get('plot_type', state['plot_type'])
    layout = request.form.get('layout', state.get('layout', 'overlay'))
    state['layout'] = layout if layout in ('overlay', 'subplots') else 'overlay'
    state['smoothing']['enabled'] = request.form.get('smoothing_enabled') == 'on'
    window_str = request.form.get('smoothing_window', '').strip()
    polyorder_str = request.form.get('smoothing_polyorder', '').strip()
    if window_str:
        w = int(window_str)
        state['smoothing']['window'] = w if w % 2 == 1 else w + 1  # savgol needs an odd window
    if polyorder_str:
        state['smoothing']['polyorder'] = int(polyorder_str)
    state['wide_mode'] = request.form.get('wide_mode') == 'on'
    tech_save_state(slug, state)
    plot_tab_name = spectrum_plot_tab_name(TECHNIQUE_SLUGS.get(slug))
    return redirect(url_for('technique_workspace', slug=slug, tab=plot_tab_name))


@app.route('/characterizations/data/technique/<slug>/baseline/save', methods=['POST'])
def tech_save_baseline(slug):
    """Applies the same ALS baseline correction shown in the preview and writes each
    corrected series back out as its own new tabular file (X unchanged, Y = raw - baseline),
    so the corrected spectra flow back into Select files like any upload."""
    state = tech_get_state(slug)
    params = tech_baseline_params(request.form)
    series_list, _errors = dp_build_series(state)

    file_ids = [s['file_id'] for s in series_list]
    source_files = {f.id: f for f in DataFile.query.filter(DataFile.id.in_(file_ids)).all()} if file_ids else {}

    for s in series_list:
        source = source_files.get(s['file_id'])
        if not source:
            continue
        baseline = baseline_als(s['y'], **params)
        corrected = s['y'] - baseline

        stored_filename = secure_filename(f"{datetime.now().timestamp()}_baseline_{source.original_filename}")
        stored_filename = os.path.splitext(stored_filename)[0] + '.csv'
        df_out = pd.DataFrame({s['x_col'] or 'X': s['x'], s['y_col'] or 'Y': corrected})
        df_out.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename), index=False)

        new_file = DataFile(
            user_id=session['user_id'],
            original_filename=f"baseline_{source.original_filename}",
            stored_filename=stored_filename,
            file_type='tabular',
            label=f"{source.label or source.original_filename} (baseline corrected)",
            technique_name=source.technique_name,
        )
        db.session.add(new_file)

    db.session.commit()
    return redirect(url_for('technique_workspace', slug=slug, tab='Select files'))


@app.route('/characterizations/data/technique/<slug>/set-derivative', methods=['POST'])
def tech_set_derivative(slug):
    state = tech_get_state(slug)
    state['derivative'] = request.form.get('derivative') == 'on'
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Derivative'))


@app.route('/characterizations/data/technique/<slug>/set-peaks', methods=['POST'])
def tech_set_peaks(slug):
    state = tech_get_state(slug)
    prominence_str = request.form.get('prominence', '').strip()
    height_str = request.form.get('min_height', '').strip()
    state['peaks']['prominence'] = float(prominence_str) if prominence_str else None
    state['peaks']['min_height'] = float(height_str) if height_str else None
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Peak Picking'))


@app.route('/characterizations/data/technique/<slug>/set-format', methods=['POST'])
def tech_set_format(slug):
    """Each Format-tab box (Legend / Colors / Lines-ticks-labels / Axis scaling) posts here
    on its own, identified by `section` — only that section's fields are touched, so applying
    or resetting one box never clobbers the others. `reset=1` restores that section's defaults
    instead of reading its fields from the form. `target` picks which plot's own format this
    edits — 'plot' (the raw spectrum, default) or 'xps_fit' (XPS's separately-styled fitted
    figure) — since the two show completely different content (a single trace vs several
    fitted components) and forcing one style onto both doesn't make sense."""
    state = tech_get_state(slug)
    target = request.form.get('target', 'plot')
    if target == 'xps_fit':
        fmt = state.get('xps_fit_format') or default_tech_format()
    else:
        target = 'plot'
        fmt = state['format']
    section = request.form.get('section', 'legend')
    reset = request.form.get('reset') == '1'

    if section not in DEFAULT_TECH_FORMAT_SECTIONS:
        section = 'legend'

    if reset:
        fmt.update(DEFAULT_TECH_FORMAT_SECTIONS[section])
    elif section == 'legend':
        fmt['legend'] = request.form.get('legend') == 'on'
        fmt['legend_loc'] = request.form.get('legend_loc', 'best')
        fmt['legend_orientation'] = request.form.get('legend_orientation', 'vertical')
        fmt['legend_scale'] = float(request.form.get('legend_scale', 1.0) or 1.0)
    elif section == 'colors':
        fmt['colormap'] = request.form.get('colormap', 'default')
    elif section == 'lines':
        fmt['line_width'] = float(request.form.get('line_width', 1.6) or 1.6)
        fmt['marker_size'] = float(request.form.get('marker_size', 18) or 18)
        fmt['tick_width'] = float(request.form.get('tick_width', 1.0) or 1.0)
        fmt['grid'] = request.form.get('grid') == 'on'
        fmt['fill_under'] = request.form.get('fill_under') == 'on'
    elif section == 'axis_labels':
        fmt['x_label'] = request.form.get('x_label', 'X').strip() or 'X'
        fmt['y_label'] = request.form.get('y_label', 'Y').strip() or 'Y'
        fmt['label_size'] = float(request.form.get('label_size', 11) or 11)
        fmt['bold_labels'] = request.form.get('bold_labels') == 'on'
    elif section == 'axis':
        fmt['log_x'] = request.form.get('log_x') == 'on'
        fmt['log_y'] = request.form.get('log_y') == 'on'
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            val = request.form.get(key, '').strip()
            fmt[key] = float(val) if val else None

    if target == 'xps_fit':
        state['xps_fit_format'] = fmt
    else:
        state['format'] = fmt
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Format', target=target))


SIM_REFS = {
    'EDS_TABLE': EDS_TABLE, 'LC_MS_ADDUCTS': LC_MS_ADDUCTS, 'AA_MONO_MASS': AA_MONO_MASS,
    'PROTON_MASS': PROTON_MASS, 'tryptic_digest': tryptic_digest, 'peptide_mono_mass': peptide_mono_mass,
}


def sim_spec_for(technique_name):
    """The technique's Simulate-tab form spec, with the adduct dropdown filled in from the
    app's own adduct table (the simulator module doesn't import app.py)."""
    spec = json.loads(json.dumps(simulate.SIM_SPECS[technique_name], default=list))
    for mode in spec['modes']:
        for field in mode['fields']:
            if field.get('options') == 'ADDUCTS':
                field['options'] = [(k, v[0].strip()) for k, v in LC_MS_ADDUCTS.items()]
    return spec


def save_simulated_file(technique_name, result):
    stored = secure_filename(f"{int(time.time() * 1000)}_simulated.csv")
    with open(os.path.join(app.config['UPLOAD_FOLDER'], stored), 'w', encoding='utf-8', newline='') as fh:
        fh.write(result['csv'])
    data_file = DataFile(
        user_id=session['user_id'],
        original_filename=re.sub(r'[^a-z0-9]+', '_', result['label'].lower()).strip('_') + '.csv',
        stored_filename=stored, file_type='tabular',
        label=f"{result['label']} ({datetime.now().strftime('%H:%M:%S')})",
        technique_name=technique_name, is_simulated=True, simulation_key=json.dumps(result['key']),
    )
    db.session.add(data_file)
    db.session.commit()
    return data_file


def _comp_save_result(slug, tab_name, result):
    comp_key = f'comp_state_{slug}'
    state = session.get(comp_key, {})
    state[tab_name] = result
    session[comp_key] = state


@app.route('/characterizations/computational/<slug>/run-ising', methods=['POST'])
@limiter.limit(MC_RATE_LIMIT)
def comp_run_ising(slug):
    if TECHNIQUE_SLUGS.get(slug) != 'Monte Carlo':
        abort(404)
    try:
        L = int(request.form.get('L', 32))
        temperature = float(request.form.get('temperature', 2.269))
        n_sweeps = int(request.form.get('n_sweeps', 400))
        seed_raw = request.form.get('seed', '').strip()
        seed = int(seed_raw) if seed_raw else None
        result = computational.run_ising_2d(
            L=L, temperature=temperature, n_sweeps=n_sweeps, seed=seed,
            out_dir=app.config['UPLOAD_FOLDER'], tag='u{}_'.format(session.get('user_id', 0)),
        )
    except Exception as e:
        session['comp_error'] = f'Ising simulation failed: {e}'
        return redirect(url_for('technique_workspace', slug=slug, tab='Ising Model'))
    _comp_save_result(slug, 'Ising Model', result)
    return redirect(url_for('technique_workspace', slug=slug, tab='Ising Model'))


@app.route('/characterizations/computational/<slug>/run-mc-integration', methods=['POST'])
@limiter.limit(MC_RATE_LIMIT)
def comp_run_mc_integration(slug):
    if TECHNIQUE_SLUGS.get(slug) != 'Monte Carlo':
        abort(404)
    try:
        target = request.form.get('target', 'circle')
        n_samples = int(request.form.get('n_samples', 100000))
        seed_raw = request.form.get('seed', '').strip()
        seed = int(seed_raw) if seed_raw else None
        result = computational.run_mc_integration(
            target=target, n_samples=n_samples, seed=seed,
            out_dir=app.config['UPLOAD_FOLDER'], tag='u{}_'.format(session.get('user_id', 0)),
        )
    except Exception as e:
        session['comp_error'] = f'Monte Carlo integration failed: {e}'
        return redirect(url_for('technique_workspace', slug=slug, tab='Monte Carlo Integration'))
    _comp_save_result(slug, 'Monte Carlo Integration', result)
    return redirect(url_for('technique_workspace', slug=slug, tab='Monte Carlo Integration'))


@app.route('/characterizations/computational/<slug>/run-random-walk', methods=['POST'])
@limiter.limit(MC_RATE_LIMIT)
def comp_run_random_walk(slug):
    if TECHNIQUE_SLUGS.get(slug) != 'Monte Carlo':
        abort(404)
    try:
        n_steps = int(request.form.get('n_steps', 2000))
        n_walkers = int(request.form.get('n_walkers', 200))
        dim = int(request.form.get('dim', 2))
        step_size = float(request.form.get('step_size', 1.0))
        seed_raw = request.form.get('seed', '').strip()
        seed = int(seed_raw) if seed_raw else None
        result = computational.run_random_walk(
            n_steps=n_steps, n_walkers=n_walkers, dim=dim, step_size=step_size, seed=seed,
            out_dir=app.config['UPLOAD_FOLDER'], tag='u{}_'.format(session.get('user_id', 0)),
        )
    except Exception as e:
        session['comp_error'] = f'Random walk simulation failed: {e}'
        return redirect(url_for('technique_workspace', slug=slug, tab='Random Walk'))
    _comp_save_result(slug, 'Random Walk', result)
    return redirect(url_for('technique_workspace', slug=slug, tab='Random Walk'))


@app.route('/characterizations/computational/<slug>/run-dft', methods=['POST'])
@limiter.limit(DFT_RATE_LIMIT)
def comp_run_dft(slug):
    if TECHNIQUE_SLUGS.get(slug) != 'DFT (small molecule)':
        abort(404)
    back = redirect(url_for('technique_workspace', slug=slug, tab='Run Calculation'))
    input_type = request.form.get('input_type', 'smiles')
    # The form has two inputs named "structure" — the SMILES box, then the XYZ textarea —
    # and both are submitted whichever is visible, so pick by input type.
    structure_fields = request.form.getlist('structure') + ['', '']
    structure_input = (structure_fields[1] if input_type == 'xyz' else structure_fields[0]).strip()
    basis = request.form.get('basis', '6-31g')
    functional = request.form.get('functional', 'b3lyp')
    label = (request.form.get('label') or '').strip()
    if not structure_input:
        session['comp_error'] = 'Give a SMILES string or XYZ coordinates first.'
        return back
    try:
        charge = int(request.form.get('charge', 0) or 0)
        spin = int(request.form.get('spin', 0) or 0)
    except ValueError:
        session['comp_error'] = 'Charge and spin must be whole numbers.'
        return back
    job_kwargs = dict(structure_input=structure_input, input_type=input_type, basis=basis,
                      functional=functional, charge=charge, spin=spin, label=label)

    if compute_queue is not None:
        job_key = f'comp_job_{slug}'
        if _comp_job_status(session.get(job_key)) in ('queued', 'started', 'deferred', 'scheduled'):
            session['comp_error'] = 'A DFT calculation is already running for you — wait for it to finish first.'
            return back
        job = compute_queue.enqueue(
            computational.run_dft_job, kwargs=job_kwargs,
            job_timeout=DFT_JOB_TIMEOUT, result_ttl=COMP_JOB_KEEP_SECONDS,
            failure_ttl=COMP_JOB_KEEP_SECONDS, meta={'user_id': session['user_id']},
        )
        session[job_key] = {'id': job.id, 'tab': 'Run Calculation'}
        return back

    outcome = computational.run_dft_job(**job_kwargs)
    if outcome['ok']:
        _comp_save_result(slug, 'Run Calculation', outcome['result'])
    else:
        session['comp_error'] = outcome['error']
    return back


def _comp_job_status(job_ref):
    if not job_ref or compute_queue is None:
        return None
    try:
        return Job.fetch(job_ref['id'], connection=redis_conn).get_status(refresh=False)
    except NoSuchJobError:
        return None


def _comp_collect_job(slug):
    """If this user has a queued compute job for this page, fold a finished result into
    the page state (or surface its error). Returns True while it's still running."""
    job_key = f'comp_job_{slug}'
    job_ref = session.get(job_key)
    if not job_ref or compute_queue is None:
        return False
    try:
        job = Job.fetch(job_ref['id'], connection=redis_conn)
    except NoSuchJobError:
        session.pop(job_key, None)
        session['comp_error'] = 'That calculation expired before its result was collected — please run it again.'
        return False
    if job.meta.get('user_id') != session.get('user_id'):
        session.pop(job_key, None)
        return False
    status = job.get_status()
    if status == 'finished':
        outcome = job.return_value() or {'ok': False, 'error': 'The calculation returned no result.'}
        if outcome.get('ok'):
            _comp_save_result(slug, job_ref['tab'], outcome['result'])
        else:
            session['comp_error'] = outcome.get('error')
        session.pop(job_key, None)
        job.delete()
        return False
    if status in ('failed', 'stopped', 'canceled'):
        # run_dft_job catches its own errors, so a failed job means the worker itself gave
        # up on it: the timeout, or the process dying (e.g. out of memory).
        latest = job.latest_result()
        if latest is not None and 'JobTimeoutException' in (latest.exc_string or ''):
            session['comp_error'] = 'The calculation took too long and was stopped — try a smaller molecule or basis set.'
        else:
            app.logger.error(f'Compute job {job.id} ended as {status}: {latest.exc_string if latest else "no result"}')
            session['comp_error'] = 'The calculation stopped unexpectedly on the server — please try again.'
        session.pop(job_key, None)
        job.delete()
        return False
    return True


@app.route('/characterizations/data/technique/<slug>/simulate', methods=['POST'])
def tech_simulate(slug):
    technique_name = TECHNIQUE_SLUGS.get(slug)
    if technique_name not in simulate.SIM_SPECS:
        abort(404)
    result = simulate.run_simulation(technique_name, request.form, SIM_REFS)
    if result.get('error'):
        session['sim_error'] = result['error']
        return redirect(url_for('technique_workspace', slug=slug, tab='Simulate'))

    data_file = save_simulated_file(technique_name, result)
    state = tech_get_state(slug)
    state['file_ids'] = [data_file.id]
    state['wide_mode'] = False
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        state['format'][key] = None
    if result.get('prominence'):
        state['peaks']['prominence'] = float(f"{result['prominence']:.3g}")
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab=result.get('next_tab') or spectrum_plot_tab_name(technique_name)))


def is_xy_curve_file(data_file):
    """False for simulated files that are tables rather than X/Y curves (m/z feature lists, ion
    images), which can't be drawn on the same axes as a real spectrum."""
    label = (data_file.label or '').lower()
    return not (data_file.is_simulated and ('feature table' in label or 'ion image' in label))


def compare_real_vs_simulated(real_id, sim_ids, normalize=True, show_residual=True):
    """Overlays one real file with one or more simulated ones on shared axes, with an optional
    residual panel (real minus simulated, on the real file's x grid) and per-pair numbers:
    RMSE and correlation over the overlapping x range, plus how far each real peak sits from
    the nearest simulated peak. Normalising scales every curve to its own maximum, since a
    simulation rarely shares the real instrument's units."""
    real_series, real_err = dp_build_series({'file_ids': [real_id]})
    sim_series, sim_err = dp_build_series({'file_ids': list(sim_ids)})
    errors = list(real_err) + list(sim_err)
    if not real_series or not sim_series:
        return {'plot_filename': None, 'rows': [], 'errors': errors or ['Nothing to compare.'], 'normalized': normalize}

    def scale(y):
        return y / (np.max(np.abs(y)) or 1.0) if normalize else y

    def peak_positions(x, y):
        idx, _ = find_peaks(y, prominence=0.05 * (np.ptp(y) or 1.0))
        return x[idx]

    real = real_series[0]
    rx, ry = real['x'], scale(real['y'])
    if show_residual:
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax_res = None
    ax.plot(rx, ry, color='#1f4e79', linewidth=1.7, label=f"Real: {real['label']}")

    palette = ['#e67e22', '#c0392b', '#8e44ad', '#16a085', '#7f8c8d']
    real_peaks = peak_positions(rx, ry)
    rows = []
    for i, s in enumerate(sim_series):
        sx, sy = s['x'], scale(s['y'])
        color = palette[i % len(palette)]
        ax.plot(sx, sy, '--', color=color, linewidth=1.5, label=f"Simulated: {s['label']}")
        row = {'label': s['label'], 'color': color, 'note': None, 'rmse': None, 'rmse_pct': None, 'r': None,
               'overlap': None, 'peak_matches': [], 'mean_abs_dx': None,
               'n_real_peaks': len(real_peaks), 'n_sim_peaks': 0}
        lo, hi = max(rx.min(), sx.min()), min(rx.max(), sx.max())
        mask = (rx >= lo) & (rx <= hi)
        if mask.sum() >= 5:
            sim_on_real = np.interp(rx[mask], sx, sy)
            resid = ry[mask] - sim_on_real
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            span = float(np.ptp(ry[mask])) or 1.0
            row.update(rmse=rmse, rmse_pct=100 * rmse / span, overlap=(float(lo), float(hi)))
            if np.std(ry[mask]) > 0 and np.std(sim_on_real) > 0:
                row['r'] = float(np.corrcoef(ry[mask], sim_on_real)[0, 1])
            if ax_res is not None:
                ax_res.plot(rx[mask], resid, color=color, linewidth=1.1)
        else:
            row['note'] = "The x ranges barely overlap, so there's nothing to compare."
        sim_peaks = peak_positions(sx, sy)
        row['n_sim_peaks'] = len(sim_peaks)
        if len(real_peaks) and len(sim_peaks):
            for p in real_peaks[:10]:
                q = sim_peaks[int(np.argmin(np.abs(sim_peaks - p)))]
                row['peak_matches'].append((float(p), float(q), float(q - p)))
            row['mean_abs_dx'] = float(np.mean([abs(m[2]) for m in row['peak_matches']]))
        rows.append(row)

    ax.set_ylabel('Normalized intensity' if normalize else 'Signal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    if ax_res is not None:
        ax_res.axhline(0, color='#999', linewidth=0.8)
        ax_res.set_ylabel('Real \u2212 sim')
        ax_res.set_xlabel('X')
        ax_res.grid(alpha=0.25)
    else:
        ax.set_xlabel('X')
    fig.tight_layout()
    base = f"compare_{int(time.time() * 1000)}"
    plot_filename = f"{base}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.svg"))
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{base}.pdf"))
    plt.close(fig)
    return {'plot_filename': plot_filename, 'rows': rows, 'errors': errors, 'normalized': normalize}


def build_compare_context(candidates, saved):
    """Works out what the Compare tab should show: which real file, which simulated files,
    and the resulting plot. Falls back to the first real file and the newest simulation so the
    tab is useful the moment it opens."""
    real_files = [f for f in candidates if not f.is_simulated]
    sim_files = [f for f in candidates if f.is_simulated and is_xy_curve_file(f)]
    saved = saved or {}
    real_id = saved.get('real_id') if saved.get('real_id') in {f.id for f in real_files} else (real_files[0].id if real_files else None)
    sim_ids = [i for i in saved.get('sim_ids', []) if i in {f.id for f in sim_files}] or ([sim_files[0].id] if sim_files else [])
    normalize = saved.get('normalize', True)
    residual = saved.get('residual', True)
    result = compare_real_vs_simulated(real_id, sim_ids, normalize, residual) if (real_id and sim_ids) else None
    return {'real_files': real_files, 'sim_files': sim_files, 'real_id': real_id, 'sim_ids': sim_ids,
            'normalize': normalize, 'residual': residual, 'result': result}


@app.route('/characterizations/data/technique/<slug>/compare-settings', methods=['POST'])
def tech_compare_settings(slug):
    if TECHNIQUE_SLUGS.get(slug) not in simulate.SIM_SPECS:
        abort(404)
    state = tech_get_state(slug)
    state['compare'] = {
        'real_id': request.form.get('real_id', type=int),
        'sim_ids': [int(i) for i in request.form.getlist('sim_ids') if i.isdigit()],
        'normalize': request.form.get('normalize') == 'on',
        'residual': request.form.get('residual') == 'on',
    }
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Compare'))


@app.route('/characterizations/data/compare-settings', methods=['POST'])
def dp_compare_settings():
    state = dp_get_state()
    state['compare'] = {
        'real_id': request.form.get('real_id', type=int),
        'sim_ids': [int(i) for i in request.form.getlist('sim_ids') if i.isdigit()],
        'normalize': request.form.get('normalize') == 'on',
        'residual': request.form.get('residual') == 'on',
    }
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='compare'))


def fit_calibration_and_quantify(standards, unknowns, plot_prefix):
    """Fits a linear calibration curve (peak area vs concentration) through pasted standards,
    then back-calculates concentration for each pasted unknown from its peak area — the
    external-standard quantification method, shared by LC-MS and HPLC/GC's Quantification
    tabs since the math is identical regardless of separation technique."""
    result = {'standards': standards, 'unknowns_raw': unknowns, 'error': None, 'plot_filename': None,
              'slope': None, 'intercept': None, 'r2': None, 'unknowns': []}

    if len(standards) < 2:
        result['error'] = "Need at least 2 standards (concentration, peak area) to fit a calibration curve."
        return result

    conc = np.array([p[0] for p in standards], dtype=float)
    area = np.array([p[1] for p in standards], dtype=float)
    if np.ptp(conc) == 0:
        result['error'] = "All standard concentrations are identical — can't fit a curve through a single point."
        return result

    slope, intercept = np.polyfit(conc, area, 1)
    pred = slope * conc + intercept
    ss_res = np.sum((area - pred) ** 2)
    ss_tot = np.sum((area - np.mean(area)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else None
    result['slope'] = float(slope)
    result['intercept'] = float(intercept)
    result['r2'] = round(float(r2), 4) if r2 is not None else None

    for label, u_area in unknowns:
        calc_conc = (u_area - intercept) / slope if slope != 0 else None
        result['unknowns'].append({'label': label, 'area': u_area, 'concentration': calc_conc})

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(conc, area, s=50, color='#2a6f2a', zorder=3, label='Standards')
    fit_x = np.linspace(min(conc.min(), 0), conc.max() * 1.05, 100)
    ax.plot(fit_x, slope * fit_x + intercept, '--', color='#888', linewidth=1.3, zorder=1, label='Calibration fit')
    unk_conc = [u['concentration'] for u in result['unknowns'] if u['concentration'] is not None]
    unk_area = [u['area'] for u in result['unknowns'] if u['concentration'] is not None]
    if unk_conc:
        ax.scatter(unk_conc, unk_area, s=60, color='#c0392b', marker='D', zorder=4, label='Unknowns (calculated)')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Peak area')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plot_filename = f"{plot_prefix}_quant_{int(datetime.now().timestamp())}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
    plt.close(fig)
    result['plot_filename'] = plot_filename
    return result


def render_chromatogram_with_peaks(x, y, peaks, out_dir, tag=''):
    """The same chromatogram Plot Chromatogram/Peak Picking would show, with each detected
    peak numbered at its apex — so choosing 'which peak' in Quantification means pointing
    at a number you can see on the actual curve, not a blind area value in a text box."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(x, y, color='#2b6cb0', linewidth=1.3)
    for i, p in enumerate(peaks, start=1):
        apex_y = float(y[p['apex_idx']])
        ax.scatter([p['rt']], [apex_y], color='#e53e3e', zorder=5, s=28)
        ax.annotate(str(i), (p['rt'], apex_y), textcoords='offset points', xytext=(0, 8),
                    fontsize=9, ha='center', color='#e53e3e', fontweight='bold')
    ax.set_xlabel('Retention time')
    ax.set_ylabel('Signal')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    filename = f"quant_chrom_{tag}_{int(datetime.now().timestamp() * 1000)}.png"
    fig.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close(fig)
    return filename


def build_quant_file_options(state):
    """Per selected file: its chromatogram (peaks numbered on the plot) and the detected
    peaks to choose from — replaces guessing 'the largest peak' with a real choice the user
    can see and override, plus a checkbox to leave a file out of this fit entirely."""
    if state.get('wide_mode'):
        series_list, _, _ = tech_build_wide_series(state)
    else:
        series_list, _ = dp_build_series(state)

    options = []
    for s in series_list:
        peaks = integrate_chromatogram_peaks(s['x'], s['y'], prominence=state['peaks'].get('prominence'), min_height=state['peaks'].get('min_height'))
        plot_filename = render_chromatogram_with_peaks(s['x'], s['y'], peaks, app.config['UPLOAD_FOLDER'], tag=str(s['file_id']))
        default_idx = max(range(len(peaks)), key=lambda i: peaks[i]['area']) if peaks else None
        options.append({
            'file_id': s['file_id'], 'label': s['label'], 'peaks': peaks,
            'plot_filename': plot_filename, 'default_idx': default_idx,
        })
    return options


def unknowns_from_quant_form(state, form):
    """Rebuilds each included file's chosen peak (by index, same numbering shown on its
    plotted chromatogram) fresh from the submitted form — never trusts a stale area value,
    always re-integrates from the current data and current peak-picking settings."""
    options = build_quant_file_options(state)
    unknowns = []
    for opt in options:
        fid = opt['file_id']
        if form.get(f'include_{fid}') != 'on' or not opt['peaks']:
            continue
        idx_str = form.get(f'peak_{fid}', '')
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if idx >= len(opt['peaks']):
            continue
        unknowns.append((opt['label'], round(opt['peaks'][idx]['area'], 4)))
    return unknowns


@app.route('/characterizations/data/technique/<slug>/lcms-quantify', methods=['POST'])
def lc_ms_quantify(slug):
    state = tech_get_state(slug)
    standards = lc_ms_parse_csv_lines(request.form.get('standards', ''), second_col_numeric_only=True)
    unknowns = unknowns_from_quant_form(state, request.form)
    state['quant_result'] = fit_calibration_and_quantify(standards, unknowns, 'lcms')
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Quantification'))


@app.route('/characterizations/data/technique/<slug>/hplc-gc-quantify', methods=['POST'])
def hplc_gc_quantify(slug):
    state = tech_get_state(slug)
    standards = lc_ms_parse_csv_lines(request.form.get('standards', ''), second_col_numeric_only=True)
    unknowns = unknowns_from_quant_form(state, request.form)
    state['quant_result'] = fit_calibration_and_quantify(standards, unknowns, 'hplcgc')
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Quantification'))


@app.route('/characterizations/data/technique/<slug>/rt-library', methods=['POST'])
def hplc_gc_set_rt_library(slug):
    state = tech_get_state(slug)
    state['rt_library_text'] = request.form.get('rt_library_text', '').strip()
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Peak Integration'))


@app.route('/characterizations/data/technique/<slug>/pmf-settings', methods=['POST'])
def maldi_set_pmf(slug):
    state = tech_get_state(slug)
    try:
        missed_cleavages = max(0, min(3, int(request.form.get('missed_cleavages', 1))))
    except ValueError:
        missed_cleavages = 1
    try:
        tolerance_da = float(request.form.get('tolerance_da', 0.3))
    except ValueError:
        tolerance_da = 0.3
    state['pmf'] = {
        'sequence': request.form.get('sequence', '').strip(),
        'missed_cleavages': missed_cleavages,
        'tolerance_da': tolerance_da,
    }
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='PMF'))


@app.route('/characterizations/data/workspace')
def data_interpretation_workspace():
    tab = request.args.get('tab', 'select')
    state = dp_get_state()

    all_files = DataFile.query.filter_by(file_type='tabular', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
    image_files = DataFile.query.filter_by(file_type='image', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
    selected_files = dp_selected_files(state)

    plot_filename, results, plot_errors, overall_analysis = (None, [], [], None)
    if tab in ('plot', 'format', 'derivative', 'analysis') and state['file_ids']:
        plot_filename, results, plot_errors, overall_analysis = dp_render_plot(state)

    compare_ctx = None
    if tab == 'compare':
        # same scope as this workspace's Simulate tab: generic curves, not technique-specific simulations
        compare_ctx = build_compare_context([f for f in all_files if not (f.is_simulated and f.technique_name)], state.get('compare'))
        if compare_ctx['result']:
            plot_filename = compare_ctx['result']['plot_filename']

    curve_specs, simulated_files = None, []
    if tab == 'simulate':
        curve_specs = simulate.CURVE_SPECS
        simulated_files = [{'file': f, 'key': json.loads(f.simulation_key or '[]')} for f in all_files if f.is_simulated and not f.technique_name]

    snapshots = []
    if tab == 'snapshots':
        snapshots = AnalysisSnapshot.query.filter_by(user_id=session['user_id']).order_by(AnalysisSnapshot.created_at.desc()).all()

    return render_template(
        'data_interpretation_workspace.html',
        page_title='Data Interpretation',
        tab=tab,
        state=state,
        all_files=all_files,
        image_files=image_files,
        selected_files=selected_files,
        snapshots=snapshots,
        curve_specs=curve_specs,
        compare_ctx=compare_ctx,
        simulated_files=simulated_files,
        sim_error=session.pop('sim_error', None),
        plot_filename=plot_filename,
        results=results,
        plot_errors=plot_errors,
        overall_analysis=overall_analysis,
        colormap_options=COLORMAP_OPTIONS,
        error=session.pop('di_error', None),
        banner_image='images/characterizations-banner.png',
    )


@app.route('/characterizations/data/select-files', methods=['POST'])
def dp_select_files():
    state = dp_get_state()
    selected = request.form.getlist('file_ids')
    new_file_ids = [int(i) for i in selected if i.isdigit()]

    if new_file_ids != state['file_ids']:
        # different data selected — a fixed axis range from before almost certainly won't fit,
        # so clear it rather than silently cropping/misrepresenting the new plot
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            state['format'][key] = None

    state['file_ids'] = new_file_ids
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='plot'))


@app.route('/characterizations/data/set-plot-type', methods=['POST'])
def dp_set_plot_type():
    state = dp_get_state()
    state['plot_type'] = request.form.get('plot_type', 'scatter')
    state['wide_mode'] = request.form.get('wide_mode') == 'on'
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='plot'))


@app.route('/characterizations/data/set-format', methods=['POST'])
def dp_set_format():
    state = dp_get_state()
    fmt = state['format']
    fmt['legend'] = request.form.get('legend') == 'on'
    fmt['legend_loc'] = request.form.get('legend_loc', 'best')
    fmt['legend_orientation'] = request.form.get('legend_orientation', 'vertical')
    fmt['legend_scale'] = float(request.form.get('legend_scale', 1.0) or 1.0)
    fmt['colormap'] = request.form.get('colormap', 'default')
    fmt['line_width'] = float(request.form.get('line_width', 1.6) or 1.6)
    fmt['marker_size'] = float(request.form.get('marker_size', 18) or 18)
    fmt['tick_width'] = float(request.form.get('tick_width', 1.0) or 1.0)
    fmt['label_size'] = float(request.form.get('label_size', 11) or 11)
    fmt['bold_labels'] = request.form.get('bold_labels') == 'on'
    fmt['grid'] = request.form.get('grid') == 'on'
    fmt['log_x'] = request.form.get('log_x') == 'on'
    fmt['log_y'] = request.form.get('log_y') == 'on'
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        val = request.form.get(key, '').strip()
        fmt[key] = float(val) if val else None

    state['format'] = fmt
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='format'))


@app.route('/characterizations/data/set-derivative', methods=['POST'])
def dp_set_derivative():
    state = dp_get_state()
    state['derivative'] = request.form.get('derivative') == 'on'
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='derivative'))


@app.route('/characterizations/data/reset', methods=['POST'])
def dp_reset():
    session.pop('dp_state', None)
    return redirect(url_for('data_interpretation_workspace', tab='select'))


SIM_CURVES = {
    'linear': linear_fn, 'quadratic': quadratic_fn, 'exponential': exponential_fn, 'power': power_fn,
    'logarithmic': logarithmic_fn, 'langmuir': langmuir_fn, 'sigmoid': sigmoid_fn, 'gaussian': gaussian_fn,
}


@app.route('/characterizations/data/simulate', methods=['POST'])
def dp_simulate():
    result = simulate.run_curve_simulation(request.form, SIM_CURVES)
    if result.get('error'):
        session['sim_error'] = result['error']
        return redirect(url_for('data_interpretation_workspace', tab='simulate'))
    data_file = save_simulated_file(None, result)
    state = dp_get_state()
    state['file_ids'] = [data_file.id]
    state['wide_mode'] = False
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        state['format'][key] = None
    dp_save_state(state)
    return redirect(url_for('data_interpretation_workspace', tab='analysis'))


@app.route('/characterizations/data/snapshots/save', methods=['POST'])
def dp_save_snapshot():
    state = dp_get_state()
    if not state.get('file_ids'):
        return redirect(url_for('data_interpretation_workspace', tab='snapshots'))

    plot_filename, results, plot_errors, overall_analysis = dp_render_plot(state)
    if not plot_filename:
        session['di_error'] = "Could not generate a plot to snapshot — check your file selection."
        return redirect(url_for('data_interpretation_workspace', tab='snapshots'))

    title = request.form.get('title', '').strip() or f"Snapshot {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    note = request.form.get('note', '').strip() or None

    snapshot_filename = f"snapshot_{session['user_id']}_{int(datetime.now().timestamp())}.png"
    shutil.copy(
        os.path.join(app.config['UPLOAD_FOLDER'], plot_filename),
        os.path.join(app.config['UPLOAD_FOLDER'], snapshot_filename),
    )

    snapshot = AnalysisSnapshot(
        user_id=session['user_id'],
        title=title,
        note=note,
        plot_type=state.get('plot_type', 'scatter_plot'),
        state_json=json.dumps(state),
        results_json=json.dumps(results, default=str),
        overall_analysis=overall_analysis,
        plot_filename=snapshot_filename,
    )
    db.session.add(snapshot)
    db.session.commit()
    return redirect(url_for('dp_view_snapshot', snapshot_id=snapshot.id))


@app.route('/characterizations/data/snapshots/<int:snapshot_id>')
def dp_view_snapshot(snapshot_id):
    snapshot = get_owned_or_404(AnalysisSnapshot, snapshot_id)
    return render_template(
        'dp_snapshot_detail.html',
        snapshot=snapshot,
        results=json.loads(snapshot.results_json),
        state=json.loads(snapshot.state_json),
    )


@app.route('/characterizations/data/snapshots/<int:snapshot_id>/restore', methods=['POST'])
def dp_restore_snapshot(snapshot_id):
    snapshot = get_owned_or_404(AnalysisSnapshot, snapshot_id)
    dp_save_state(json.loads(snapshot.state_json))
    return redirect(url_for('data_interpretation_workspace', tab='plot'))


@app.route('/characterizations/data/snapshots/<int:snapshot_id>/delete', methods=['POST'])
def dp_delete_snapshot(snapshot_id):
    snapshot = get_owned_or_404(AnalysisSnapshot, snapshot_id)
    if snapshot.plot_filename:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], snapshot.plot_filename))
        except OSError:
            pass
    db.session.delete(snapshot)
    db.session.commit()
    return redirect(url_for('data_interpretation_workspace', tab='snapshots'))


@app.route('/characterizations/data/upload', methods=['POST'])
def upload_data_file():
    uploaded_files = request.files.getlist('data_file')
    label = request.form.get('label', '').strip()
    technique_name = request.form.get('technique', '').strip() or None
    redirect_slug = request.form.get('slug', '').strip() or None
    channel_type = request.form.get('channel_type', '').strip() or None

    uploaded_files = [f for f in uploaded_files if f and f.filename]
    if not uploaded_files:
        if redirect_slug:
            return redirect(url_for('technique_workspace', slug=redirect_slug))
        return redirect(url_for('data_interpretation_workspace'))

    created_ids = []
    skipped = []

    for uploaded_file in uploaded_files:
        original_filename = uploaded_file.filename
        ext = os.path.splitext(original_filename)[1].lower()

        is_afm_native = technique_name == 'AFM' and ext in AFM_NATIVE_EXTENSIONS
        is_afm_unparsed = technique_name == 'AFM' and ext in AFM_UNPARSED_EXTENSIONS

        if is_afm_native or is_afm_unparsed:
            file_type = 'image'  # provisional; native parse failure below demotes to 'unparsed'
        elif ext in IMAGE_EXTENSIONS:
            file_type = 'image'
        elif ext in KNOWN_NON_DATA_EXTENSIONS:
            skipped.append(original_filename)
            continue
        else:
            # accept any other extension (.csv, .txt, .dat, .mpt, or anything unrecognized)
            # as a data file to attempt — the parser auto-detects delimiter/header on its own
            file_type = 'tabular'

        stored_filename = secure_filename(f"{datetime.now().timestamp()}_{original_filename}")
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename))
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)

        # if multiple files share one label, use it as a prefix; otherwise each file's own name
        entry_label = f"{label} — {original_filename}" if (label and len(uploaded_files) > 1) else (label or None)

        new_file = DataFile(
            user_id=session['user_id'],
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_type=file_type,
            label=entry_label,
            technique_name=technique_name,
        )

        if is_afm_native:
            parsed = try_parse_afm_native(full_path, ext)
            if parsed is not None:
                new_file.channel_type = channel_type or 'topography'
                new_file.data_units = parsed['units']
                new_file.pixel_size_nm = parsed['pixel_size_nm']
                new_file.parse_status = 'parsed_native'
                height_filename = f"height_{os.path.splitext(stored_filename)[0]}.npy"
                np.save(os.path.join(app.config['UPLOAD_FOLDER'], height_filename), parsed['data'])
                new_file.height_data_filename = height_filename
            else:
                new_file.file_type = 'unparsed'
                new_file.parse_status = 'unparsed_raw'
        elif is_afm_unparsed:
            new_file.file_type = 'unparsed'
            new_file.parse_status = 'unparsed_raw'
        elif file_type == 'image':
            if technique_name == 'AFM':
                new_file.channel_type = channel_type or 'topography'
                new_file.parse_status = 'image_only'
            elif technique_name == 'SEM' and channel_type in ('se', 'bse'):
                new_file.channel_type = channel_type
            try:
                new_file.pixel_size_nm = extract_pixel_size_from_tiff(full_path)
                gray = np.array(Image.open(full_path).convert('L'))
                new_file.image_crop_bottom = detect_image_info_bar(gray)
            except Exception:
                pass

            # browsers can't render TIFF (and some other formats) via <img> — generate a PNG
            # preview for display purposes, while keeping the original file for metadata/analysis
            if ext in ('.tif', '.tiff', '.bmp'):
                try:
                    preview_filename = f"preview_{os.path.splitext(stored_filename)[0]}.png"
                    with Image.open(full_path) as im:
                        im.convert('RGB').save(os.path.join(app.config['UPLOAD_FOLDER'], preview_filename))
                    new_file.preview_filename = preview_filename
                except Exception:
                    pass
        elif file_type == 'tabular' and technique_name == 'AFM':
            new_file.channel_type = 'force_curve'

        db.session.add(new_file)
        db.session.commit()

        # for a successfully-parsed native AFM file, render a real colored preview now that
        # the row has an id (generate_channel_overlay needs one for the output filename)
        if is_afm_native and new_file.parse_status == 'parsed_native':
            try:
                overlay = generate_channel_overlay(new_file)
                if overlay['overlay_filename']:
                    new_file.preview_filename = overlay['overlay_filename']
                    db.session.commit()
            except Exception:
                pass

        created_ids.append((new_file.id, file_type))

    if skipped:
        session['di_error'] = f"Skipped unsupported file(s): {', '.join(skipped)}. Use CSV, TXT, XLSX for data, or PNG/JPG/TIFF for TEM/SEM images."

    # advance the wizard to the "employ files" step, if a worksheet is active
    if session.get('di_stage') in ('import', 'employ'):
        session['di_stage'] = 'employ'

    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        if technique_name == 'AFM':
            select_tab = 'Select data'
        else:
            select_tab = 'Select images' if file_type == 'image' else 'Select files'
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab=select_tab))

    return redirect(url_for('data_interpretation_workspace'))


@app.route('/characterizations/data/<int:file_id>/configure', methods=['GET', 'POST'])
def configure_data_file(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    ext = os.path.splitext(data_file.stored_filename)[1].lower()

    # parsing options — from the request if the user is adjusting them, else fall back
    # to whatever was saved for this file before, else 'auto'
    delim_choice = request.values.get('delimiter', data_file.parse_delimiter or 'auto')
    header_row_str = request.values.get('header_row', '')
    if header_row_str == '' and data_file.parse_header_row is not None:
        header_row_str = str(data_file.parse_header_row)
    header_row_override = int(header_row_str) if header_row_str.strip().isdigit() else None
    delimiter_override = DELIMITER_MAP.get(delim_choice)

    try:
        df = read_tabular_file(filepath, ext, delimiter_override=delimiter_override, header_row_override=header_row_override)
    except Exception as e:
        return render_template(
            'configure_data_file.html', data_file=data_file, columns=[], preview=None,
            delim_choice=delim_choice, header_row_str=header_row_str,
            error=f"Couldn't read this file with these settings: {e}",
        )

    columns = list(df.columns)
    preview = df.head(5).to_dict('records')

    if request.method == 'POST' and request.form.get('x_column'):
        x_col = request.form.get('x_column')
        y_col = request.form.get('y_column')
        fit_type = request.form.get('fit_type', 'none')

        try:
            x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
            y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            if len(x) < 2:
                raise ValueError("Not enough numeric data points in the chosen columns — try adjusting the delimiter or header row above.")

            plot_filename = f"plot_{data_file.id}_{int(datetime.now().timestamp())}.png"
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
            fit_params, r_squared = generate_plot_and_fit(x, y, fit_type, plot_path)

            data_file.x_column = x_col
            data_file.y_column = y_col
            data_file.fit_type = fit_type
            data_file.fit_params = json.dumps(fit_params) if fit_params else None
            data_file.r_squared = r_squared
            data_file.plot_filename = plot_filename
            data_file.parse_delimiter = None if delim_choice == 'auto' else delim_choice
            data_file.parse_header_row = header_row_override
            db.session.commit()

            return redirect(url_for('view_data_file', file_id=data_file.id))
        except Exception as e:
            return render_template(
                'configure_data_file.html', data_file=data_file, columns=columns, preview=preview,
                delim_choice=delim_choice, header_row_str=header_row_str,
                error=f"Couldn't plot this: {e}",
            )

    return render_template(
        'configure_data_file.html', data_file=data_file, columns=columns, preview=preview,
        delim_choice=delim_choice, header_row_str=header_row_str,
    )


@app.route('/characterizations/data/<int:file_id>')
def view_data_file(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    fit_params = json.loads(data_file.fit_params) if data_file.fit_params else None
    analysis = None
    if data_file.file_type == 'tabular' and data_file.plot_filename:
        analysis = generate_fit_analysis(
            data_file.fit_type, fit_params, data_file.r_squared,
            x_col=data_file.x_column or 'X', y_col=data_file.y_column or 'Y',
        )
    return render_template('view_data_file.html', data_file=data_file, fit_params=fit_params, analysis=analysis)


@app.route('/characterizations/data/<int:file_id>/configure-force-curve', methods=['GET', 'POST'])
def configure_force_curve(file_id):
    """Column-picker + fit configuration for an uploaded force-distance curve CSV/TXT —
    used by both the Mechanical tab (Hertz/DMT) and Biological tab (WLC/SMFS), which share
    this exact same route and DataFile columns; only the offered fit_type options differ
    in the calling template."""
    data_file = get_owned_or_404(DataFile, file_id)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    ext = os.path.splitext(data_file.stored_filename)[1].lower()

    try:
        df = read_tabular_file(filepath, ext)
    except Exception as e:
        return render_template(
            'configure_force_curve.html', data_file=data_file, columns=[], preview=None,
            error=f"Couldn't read this file: {e}",
        )

    columns = list(df.columns)
    preview = df.head(5).to_dict('records')

    if request.method == 'POST' and request.form.get('distance_column'):
        dist_col = request.form.get('distance_column')
        approach_col = request.form.get('approach_column')
        retract_col = request.form.get('retract_column') or None
        fit_type = request.form.get('fit_type', 'none')
        tip_radius_str = request.form.get('tip_radius_nm', '').strip()
        tip_radius_nm = float(tip_radius_str) if tip_radius_str else None
        spring_const_str = request.form.get('spring_constant', '').strip()
        spring_constant = float(spring_const_str) if spring_const_str else None
        find_unfolding = request.form.get('find_unfolding') == 'on'

        try:
            x = pd.to_numeric(df[dist_col], errors='coerce').to_numpy()
            y_app = pd.to_numeric(df[approach_col], errors='coerce').to_numpy()
            y_ret = pd.to_numeric(df[retract_col], errors='coerce').to_numpy() if retract_col else None

            mask = ~(np.isnan(x) | np.isnan(y_app))
            if y_ret is not None:
                mask &= ~np.isnan(y_ret)
            x_clean, y_app_clean = x[mask], y_app[mask]
            y_ret_clean = y_ret[mask] if y_ret is not None else None

            # if the force column is raw deflection rather than pre-converted force,
            # convert using the cantilever spring constant: F = deflection * k
            if spring_constant:
                y_app_clean = y_app_clean * spring_constant
                if y_ret_clean is not None:
                    y_ret_clean = y_ret_clean * spring_constant

            if len(x_clean) < 3:
                raise ValueError("Not enough numeric data points in the chosen columns.")

            plot_filename = f"forcecurve_{data_file.id}_{int(datetime.now().timestamp())}.png"
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
            results = generate_force_curve_plot(
                x_clean, y_app_clean, y_ret_clean, fit_type, tip_radius_nm, plot_path,
                find_unfolding=find_unfolding,
            )

            data_file.x_column = dist_col
            data_file.y_column = approach_col
            data_file.y_column_retract = retract_col
            data_file.fit_type = fit_type
            data_file.fit_params = json.dumps(results) if results else None
            data_file.r_squared = results.get('r_squared')
            data_file.adhesion_force = results.get('adhesion_force')
            data_file.spring_constant = spring_constant
            data_file.plot_filename = plot_filename
            db.session.commit()

            return redirect(url_for('view_force_curve', file_id=data_file.id))
        except Exception as e:
            return render_template(
                'configure_force_curve.html', data_file=data_file, columns=columns, preview=preview,
                error=f"Couldn't fit this: {e}",
            )

    return render_template('configure_force_curve.html', data_file=data_file, columns=columns, preview=preview)


@app.route('/characterizations/data/<int:file_id>/force-curve')
def view_force_curve(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    results = json.loads(data_file.fit_params) if data_file.fit_params else {}
    return render_template('view_force_curve.html', data_file=data_file, results=results)


def build_particle_size_analysis(sizes, unit, method_label='Manual sizing'):
    n = len(sizes)
    sizes_arr = np.asarray(sizes, dtype=float)
    mean_size = float(np.mean(sizes_arr))
    std_size = float(np.std(sizes_arr))
    cv = std_size / mean_size if mean_size else 0
    parts = [f"{method_label} of {n} particle{'s' if n != 1 else ''}: mean diameter = {mean_size:.1f} {unit} (± {std_size:.1f} {unit} std dev), range {min(sizes):.1f}–{max(sizes):.1f} {unit}."]

    if n >= 10:
        # A real check, not just eyeballing the CV: does the histogram itself show more than
        # one mode? Two nucleation events, aggregation, or a mixed population all show up this
        # way — something a single mean+std can't reveal.
        bins = min(20, max(6, n // 3))
        counts, edges = np.histogram(sizes_arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        peak_idx, _ = find_peaks(counts, prominence=max(1, counts.max() * 0.15))
        if len(peak_idx) >= 2:
            modes = ", ".join(f"{centers[i]:.1f} {unit}" for i in peak_idx)
            parts.append(f"The size histogram itself looks multimodal, with distinct peaks near {modes} — worth checking whether this is two genuinely different populations (separate nucleation events, aggregation, or a mixed sample) rather than one synthesis batch, since a single mean/std can hide this.")
        else:
            skewness = float(scipy_skew(sizes_arr)) if n >= 5 else 0.0
            if skewness > 0.5:
                parts.append(f"The distribution is right-skewed (skewness ≈ {skewness:.2f}) — a tail of larger particles or aggregates is pulling the mean above the typical (median) size.")
            elif skewness < -0.5:
                parts.append(f"The distribution is left-skewed (skewness ≈ {skewness:.2f}) — a tail of smaller particles/fragments sits alongside a dominant larger population.")

    if cv < 0.15:
        parts.append("The narrow size distribution (CV < 15%) indicates a fairly monodisperse particle population.")
    elif cv < 0.35:
        parts.append("The moderate spread (CV ≈ {:.0f}%) suggests some polydispersity typical of many synthesis routes.".format(cv * 100))
    else:
        parts.append("The broad size distribution (CV ≈ {:.0f}%) indicates significant polydispersity — worth checking for multiple particle populations or agglomeration effects.".format(cv * 100))
    if n < 20:
        parts.append(f"Note: only {n} particles were measured — for a statistically robust size distribution, papers typically report measurements from 50–100+ particles.")
    return " ".join(parts)


def build_step_height_analysis(heights, unit, quantity_label='step-height'):
    n = len(heights)
    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    parts = [f"{n} {quantity_label} measurement{'s' if n != 1 else ''}: mean = {mean_h:.2f} {unit} (± {std_h:.2f} {unit} std dev), range {min(heights):.2f}–{max(heights):.2f} {unit}."]
    if n < 5:
        parts.append(f"Note: only {n} measurement{'s were' if n != 1 else ' was'} taken — measure the same feature at several points for a more reliable {quantity_label} value.")
    return " ".join(parts)


def build_saed_analysis(d_spacings, k, spot_angles):
    n = len(d_spacings)
    parts = [f"{n} spot{'s' if n != 1 else ''} measured, calibration constant k = {k:.4f} nm·px."]
    parts.append("d-spacings: " + ", ".join(f"{d:.4f} nm" for d in d_spacings) + ".")
    if spot_angles:
        parts.append("Angles between spots may indicate orientation relationships between grains/phases — compare against known crystal system geometry for phase identification; this tool reports the measured geometry only, not an automatic phase match.")
    return " ".join(parts)


def build_lattice_fringe_analysis(spacings, unit):
    n = len(spacings)
    mean_d = float(np.mean(spacings))
    parts = [f"{n} lattice fringe spacing{'s' if n != 1 else ''} measured from the FFT: mean = {mean_d:.4f} {unit} (± {float(np.std(spacings)):.4f} {unit})."]
    parts.append("Compare against known d-spacings for candidate phases to help identify the crystal structure/orientation — this is a geometric measurement, not an automatic phase match.")
    return " ".join(parts)


@app.route('/characterizations/data/<int:file_id>/measure')
def measure_particles(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    if data_file.file_type != 'image':
        return redirect(url_for('data_interpretation_workspace'))

    mode = request.args.get('mode', 'particle_size')
    if mode not in ('particle_size', 'layer_thickness'):
        mode = 'particle_size'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    with Image.open(filepath) as img:
        img_w, img_h = img.size
    crop_bottom = data_file.image_crop_bottom or img_h

    return render_template(
        'measure_particles.html',
        data_file=data_file,
        img_w=img_w, img_h=crop_bottom,
        pixel_size_nm=data_file.pixel_size_nm,
        mode=mode,
    )


@app.route('/characterizations/data/<int:file_id>/measure/save', methods=['POST'])
def save_particle_measurements(file_id):
    data_file = get_owned_or_404(DataFile, file_id)

    mode = request.form.get('measurement_type', 'particle_size')
    if mode not in ('particle_size', 'layer_thickness'):
        mode = 'particle_size'

    try:
        pixel_distances = json.loads(request.form.get('pixel_distances', '[]'))
    except (ValueError, TypeError):
        pixel_distances = []

    scale_str = request.form.get('pixel_size_nm', '').strip()
    scale = float(scale_str) if scale_str else None

    if not pixel_distances:
        return redirect(url_for('measure_particles', file_id=file_id, mode=mode))

    if scale:
        sizes = [d * scale for d in pixel_distances]
        unit = 'nm'
    else:
        sizes = pixel_distances
        unit = 'px'

    quantity_label = 'Layer thickness' if mode == 'layer_thickness' else 'Particle diameter'

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=min(20, max(5, len(sizes) // 2)), color='#2b6cb0', edgecolor='white')
    ax.axvline(np.mean(sizes), color='#e53e3e', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(sizes):.1f} {unit}')
    ax.set_xlabel(f'{quantity_label} ({unit})')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    hist_filename = f"particle_hist_{file_id}_{int(datetime.now().timestamp())}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], hist_filename), dpi=130)
    plt.close(fig)

    if mode == 'layer_thickness':
        analysis_text = build_step_height_analysis(sizes, unit, quantity_label='layer-thickness')
    else:
        analysis_text = build_particle_size_analysis(sizes, unit)

    analysis = ImageAnalysis(
        file_id=file_id,
        pixel_size_nm=scale,
        sizes_nm_json=json.dumps(sizes),
        unit=unit,
        histogram_filename=hist_filename,
        analysis_text=analysis_text,
        measurement_type=mode,
    )
    db.session.add(analysis)
    db.session.commit()

    return redirect(url_for('view_particle_analysis', analysis_id=analysis.id))


MEASUREMENT_TYPE_INFO = {
    'particle_size': {'title': 'Particle size', 'back_slug': None, 'back_tab': None, 'back_label': 'Data Interpretation'},
    'step_height': {'title': 'Step height', 'back_slug': 'afm', 'back_tab': 'Topography', 'back_label': 'Topography'},
    'layer_thickness': {'title': 'Layer thickness', 'back_slug': 'tem', 'back_tab': 'Layer Thickness', 'back_label': 'Layer Thickness'},
    'saed_spacing': {'title': 'SAED d-spacing', 'back_slug': 'tem', 'back_tab': 'SAED', 'back_label': 'SAED'},
    'lattice_fringe': {'title': 'Lattice fringe spacing', 'back_slug': 'tem', 'back_tab': 'Lattice Fringes', 'back_label': 'Lattice Fringes'},
}


def combine_particle_analyses(entries):
    """Merges multiple saved measurements into one integrated write-up per measurement
    type, instead of a separate card for every single saved analysis — the same treatment
    already applied to every spectroscopy technique's Analysis tab. Strain maps are excluded
    (they're a different visual — exx/eyy/exy images, no histogram — that the caller should
    render separately from its own, unfiltered list; they don't reduce to a size-comparison
    sentence the way the others do).
    Returns a list of {label, analysis_text, histogram_filenames, entries} groups."""
    groups = {}
    order = []
    for e in entries:
        mt = e['analysis'].measurement_type
        if mt == 'strain_map':
            continue
        if mt not in groups:
            groups[mt] = []
            order.append(mt)
        groups[mt].append(e)

    combined = []
    for mt in order:
        items = groups[mt]
        type_label = MEASUREMENT_TYPE_INFO.get(mt, {}).get('title', mt.replace('_', ' ').title())
        histogram_filenames = [e['analysis'].histogram_filename for e in items if e['analysis'].histogram_filename]

        if len(items) == 1:
            e = items[0]
            label = (e['file'].label or e['file'].original_filename) if e['file'] else 'Unknown file'
            combined.append({'label': f"{type_label} — {label}", 'analysis_text': e['analysis'].analysis_text,
                              'histogram_filenames': histogram_filenames, 'entries': items})
            continue

        sentences = []
        means = []
        for e in items:
            label = (e['file'].label or e['file'].original_filename) if e['file'] else 'Unknown file'
            text = (e['analysis'].analysis_text or '').strip()
            if text and text[-1] not in '.!?':
                text += '.'
            sentences.append(f"{label} — {text}")
            try:
                sizes = json.loads(e['analysis'].sizes_nm_json)
                if sizes:
                    means.append((label, float(np.mean(sizes)), e['analysis'].unit))
            except Exception:
                pass

        text = f"Across {len(items)} {type_label.lower()} measurements: " + " ".join(sentences)
        if len(means) >= 2:
            unit = means[0][2]
            biggest = max(means, key=lambda m: m[1])
            smallest = min(means, key=lambda m: m[1])
            if biggest[0] != smallest[0] and smallest[1]:
                spread_pct = (biggest[1] - smallest[1]) / smallest[1] * 100
                if spread_pct > 20:
                    text += (f" {biggest[0]} shows a notably larger mean ({biggest[1]:.1f} {unit}) than {smallest[0]} "
                             f"({smallest[1]:.1f} {unit}) — worth checking whether that's a real difference between "
                             f"samples/regions rather than a difference in measurement settings.")
                else:
                    text += f" Mean values stay broadly consistent across these measurements ({smallest[1]:.1f}–{biggest[1]:.1f} {unit})."

        combined.append({'label': f"All {len(items)} {type_label.lower()} measurements", 'analysis_text': text,
                          'histogram_filenames': histogram_filenames, 'entries': items})

    return combined


def compute_saed_spot_angles(config):
    """Given saved SAED spot coordinates (relative to the pattern's transmitted-beam
    center), returns the angle (degrees) between every pair of measured spots —
    genuinely useful for orientation-relationship work, computed directly from the same
    click coordinates already used for the d-spacing itself, no extra measurement needed."""
    spots = config.get('spots', [])
    angles = []
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            v1, v2 = spots[i], spots[j]
            dot = v1['x'] * v2['x'] + v1['y'] * v2['y']
            mag1, mag2 = np.hypot(v1['x'], v1['y']), np.hypot(v2['x'], v2['y'])
            if mag1 == 0 or mag2 == 0:
                continue
            cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angles.append({'i': i + 1, 'j': j + 1, 'degrees': float(np.degrees(np.arccos(cos_a)))})
    return angles


@app.route('/characterizations/data/particle-analysis/<int:analysis_id>')
def view_particle_analysis(analysis_id):
    analysis = ImageAnalysis.query.get_or_404(analysis_id)
    data_file = get_owned_or_404(DataFile, analysis.file_id)
    sizes = json.loads(analysis.sizes_nm_json)
    info = MEASUREMENT_TYPE_INFO.get(analysis.measurement_type, MEASUREMENT_TYPE_INFO['particle_size'])
    spot_angles = None
    if analysis.measurement_type == 'saed_spacing' and analysis.config_json:
        spot_angles = compute_saed_spot_angles(json.loads(analysis.config_json))
    return render_template(
        'view_particle_analysis.html', analysis=analysis, data_file=data_file, sizes=sizes,
        info=info, spot_angles=spot_angles,
    )


@app.route('/characterizations/data/<int:file_id>/measure-topography')
def measure_topography(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    if data_file.parse_status != 'parsed_native' or not data_file.height_data_filename:
        # no real Z data — unlike lateral roughness, there's no honest way to estimate a
        # height difference from pixel brightness, so this tool simply doesn't apply
        return redirect(url_for('technique_workspace', slug='afm', tab='Topography'))

    arr = np.load(os.path.join(app.config['UPLOAD_FOLDER'], data_file.height_data_filename))
    img_h, img_w = arr.shape

    return render_template(
        'measure_topography.html',
        data_file=data_file,
        img_w=int(img_w), img_h=int(img_h),
        pixel_size_nm=data_file.pixel_size_nm,
    )


@app.route('/characterizations/data/<int:file_id>/measure-topography/save', methods=['POST'])
def save_topography_measurement(file_id):
    data_file = get_owned_or_404(DataFile, file_id)

    try:
        point_pairs = json.loads(request.form.get('point_pairs', '[]'))
    except (ValueError, TypeError):
        point_pairs = []
    if not point_pairs or not data_file.height_data_filename:
        return redirect(url_for('measure_topography', file_id=file_id))

    arr = np.load(os.path.join(app.config['UPLOAD_FOLDER'], data_file.height_data_filename))
    h, w = arr.shape
    heights = []
    for (x1, y1), (x2, y2) in point_pairs:
        x1, y1 = min(int(round(x1)), w - 1), min(int(round(y1)), h - 1)
        x2, y2 = min(int(round(x2)), w - 1), min(int(round(y2)), h - 1)
        heights.append(abs(float(arr[y2, x2]) - float(arr[y1, x1])))

    unit = data_file.data_units or 'nm'

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(heights) >= 3:
        ax.hist(heights, bins=min(20, max(5, len(heights) // 2)), color='#2b6cb0', edgecolor='white')
        ax.axvline(np.mean(heights), color='#e53e3e', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(heights):.2f} {unit}')
        ax.set_xlabel(f'Step height ({unit})')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
    else:
        ax.bar(range(1, len(heights) + 1), heights, color='#2b6cb0')
        ax.set_xlabel('Measurement #')
        ax.set_ylabel(f'Step height ({unit})')
    ax.grid(alpha=0.25)
    fig.tight_layout()

    hist_filename = f"stepheight_{file_id}_{int(datetime.now().timestamp())}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], hist_filename), dpi=130)
    plt.close(fig)

    analysis = ImageAnalysis(
        file_id=file_id,
        pixel_size_nm=data_file.pixel_size_nm,
        sizes_nm_json=json.dumps(heights),
        unit=unit,
        histogram_filename=hist_filename,
        analysis_text=build_step_height_analysis(heights, unit),
        measurement_type='step_height',
    )
    db.session.add(analysis)
    db.session.commit()

    return redirect(url_for('view_particle_analysis', analysis_id=analysis.id))


DEFECT_TYPES = {
    'dislocation': 'Dislocation', 'stacking_fault': 'Stacking fault', 'grain_boundary': 'Grain boundary',
    'vacancy': 'Vacancy', 'twin_boundary': 'Twin boundary', 'other': 'Other',
}


@app.route('/characterizations/data/<int:file_id>/set-pixel-size', methods=['POST'])
def set_pixel_size(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    val = request.form.get('pixel_size_nm', '').strip()
    data_file.pixel_size_nm = float(val) if val else None
    db.session.commit()
    redirect_slug = request.form.get('slug', '').strip()
    redirect_tab = request.form.get('tab', '').strip() or None
    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab=redirect_tab))
    return redirect(url_for('data_interpretation_workspace'))


@app.route('/characterizations/data/<int:file_id>/measure-saed')
def measure_saed(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    if data_file.file_type != 'image':
        return redirect(url_for('technique_workspace', slug='tem', tab='SAED'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    with Image.open(filepath) as img:
        img_w, img_h = img.size
    return render_template(
        'measure_saed.html', data_file=data_file, img_w=img_w, img_h=img_h,
        error=session.pop('tem_error', None),
    )


@app.route('/characterizations/data/<int:file_id>/measure-saed/save', methods=['POST'])
def save_saed_measurement(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    try:
        center = json.loads(request.form.get('center', 'null'))
        spots = json.loads(request.form.get('spots', '[]'))  # [{x, y, known_d: number|null}]
    except (ValueError, TypeError):
        center, spots = None, []

    k_str = request.form.get('camera_constant', '').strip()
    k = float(k_str) if k_str else None

    if not center or not spots:
        return redirect(url_for('measure_saed', file_id=file_id))

    rel_spots = []
    for s in spots:
        dx, dy = s['x'] - center['x'], s['y'] - center['y']
        rel_spots.append({'x': dx, 'y': dy, 'r_px': float(np.hypot(dx, dy)), 'known_d': s.get('known_d')})

    if not k:
        calib = next((s for s in rel_spots if s.get('known_d')), None)
        if not calib:
            session['tem_error'] = "Provide a calibration constant, or enter the known d-spacing for at least one measured spot."
            return redirect(url_for('measure_saed', file_id=file_id))
        k = float(calib['known_d']) * calib['r_px']

    d_spacings = []
    for s in rel_spots:
        d = k / s['r_px'] if s['r_px'] else None
        s['d_nm'] = d
        if d is not None:
            d_spacings.append(d)

    spot_angles = compute_saed_spot_angles({'spots': rel_spots})

    analysis = ImageAnalysis(
        file_id=file_id,
        pixel_size_nm=k,
        sizes_nm_json=json.dumps(d_spacings),
        unit='nm',
        analysis_text=build_saed_analysis(d_spacings, k, spot_angles),
        measurement_type='saed_spacing',
        config_json=json.dumps({'spots': rel_spots}),
    )
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('view_particle_analysis', analysis_id=analysis.id))


@app.route('/characterizations/data/<int:file_id>/measure-lattice-fringe')
def measure_lattice_fringe(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    if data_file.file_type != 'image':
        return redirect(url_for('technique_workspace', slug='tem', tab='Lattice Fringes'))
    if not data_file.pixel_size_nm:
        session['tem_error'] = "This image has no calibrated pixel size — enter one for it on the Select images tab before measuring lattice fringes."
        return redirect(url_for('technique_workspace', slug='tem', tab='Lattice Fringes'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    gray = np.array(Image.open(filepath).convert('L'))
    crop = data_file.image_crop_bottom or gray.shape[0]
    gray = gray[:crop, :]
    log_mag, _ = compute_fft_power_spectrum(gray)
    norm = (log_mag - log_mag.min()) / (np.ptp(log_mag) or 1) * 255
    fft_filename = f"fft_{file_id}_{int(datetime.now().timestamp())}.png"
    Image.fromarray(norm.astype('uint8')).save(os.path.join(app.config['UPLOAD_FOLDER'], fft_filename))

    h, w = gray.shape
    return render_template(
        'measure_spots_fft.html', data_file=data_file, fft_filename=fft_filename,
        img_w=w, img_h=h, pixel_size_nm=data_file.pixel_size_nm, mode='lattice_fringe',
    )


@app.route('/characterizations/data/<int:file_id>/measure-lattice-fringe/save', methods=['POST'])
def save_lattice_fringe_measurement(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    try:
        spots = json.loads(request.form.get('spots', '[]'))  # [{x, y}] absolute FFT-image pixel coords
    except (ValueError, TypeError):
        spots = []
    if not spots or not data_file.pixel_size_nm:
        return redirect(url_for('measure_lattice_fringe', file_id=file_id))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    with Image.open(filepath) as img:
        img_w, img_h = img.size
    crop = data_file.image_crop_bottom or img_h

    spacings = []
    for s in spots:
        d = fft_spot_to_spacing(s['x'], s['y'], img_w, crop, data_file.pixel_size_nm)
        if d:
            spacings.append(d)

    if not spacings:
        return redirect(url_for('measure_lattice_fringe', file_id=file_id))

    analysis = ImageAnalysis(
        file_id=file_id,
        pixel_size_nm=data_file.pixel_size_nm,
        sizes_nm_json=json.dumps(spacings),
        unit='nm',
        analysis_text=build_lattice_fringe_analysis(spacings, 'nm'),
        measurement_type='lattice_fringe',
    )
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('view_particle_analysis', analysis_id=analysis.id))


@app.route('/characterizations/data/<int:file_id>/measure-strain')
def measure_strain(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    if data_file.file_type != 'image':
        return redirect(url_for('technique_workspace', slug='tem', tab='Strain Mapping'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    gray = np.array(Image.open(filepath).convert('L'))
    crop = data_file.image_crop_bottom or gray.shape[0]
    gray = gray[:crop, :]
    log_mag, _ = compute_fft_power_spectrum(gray)
    norm = (log_mag - log_mag.min()) / (np.ptp(log_mag) or 1) * 255
    fft_filename = f"fft_{file_id}_{int(datetime.now().timestamp())}.png"
    Image.fromarray(norm.astype('uint8')).save(os.path.join(app.config['UPLOAD_FOLDER'], fft_filename))

    h, w = gray.shape
    return render_template(
        'measure_strain.html', data_file=data_file, fft_filename=fft_filename, img_w=w, img_h=h,
        error=session.pop('tem_error', None),
    )


@app.route('/characterizations/data/<int:file_id>/measure-strain/save', methods=['POST'])
def save_strain_measurement(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    try:
        g1 = json.loads(request.form.get('g1', 'null'))
        g2 = json.loads(request.form.get('g2', 'null'))
        ref_rect = json.loads(request.form.get('ref_rect', 'null'))
    except (ValueError, TypeError):
        g1 = g2 = ref_rect = None

    if not g1 or not g2 or not ref_rect:
        return redirect(url_for('measure_strain', file_id=file_id))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file.stored_filename)
    gray = np.array(Image.open(filepath).convert('L'))
    crop = data_file.image_crop_bottom or gray.shape[0]
    gray = gray[:crop, :]

    try:
        images, stats = generate_strain_map_overlay(
            data_file, gray, (g1['x'], g1['y']), (g2['x'], g2['y']),
            (ref_rect['x0'], ref_rect['y0'], ref_rect['x1'], ref_rect['y1']),
        )
    except ValueError as e:
        session['tem_error'] = str(e)
        return redirect(url_for('measure_strain', file_id=file_id))

    analysis_text = (
        f"Mean/max strain — εxx: {stats['exx']['mean']:.4f}/{stats['exx']['max']:.4f}, "
        f"εyy: {stats['eyy']['mean']:.4f}/{stats['eyy']['max']:.4f}, "
        f"εxy: {stats['exy']['mean']:.4f}/{stats['exy']['max']:.4f}. "
        "Geometric phase analysis is only as reliable as the g-vector picks and a genuinely "
        "defect-free/unstrained reference region — treat as comparative/qualitative unless "
        "validated against a known standard."
    )

    analysis = ImageAnalysis(
        file_id=file_id,
        sizes_nm_json='[]',
        unit='strain',
        analysis_text=analysis_text,
        measurement_type='strain_map',
        config_json=json.dumps({'g1': g1, 'g2': g2, 'ref_rect': ref_rect, 'stats': stats}),
        extra_images_json=json.dumps(images),
    )
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('view_strain_map', analysis_id=analysis.id))


@app.route('/characterizations/data/strain-map/<int:analysis_id>')
def view_strain_map(analysis_id):
    analysis = ImageAnalysis.query.get_or_404(analysis_id)
    data_file = get_owned_or_404(DataFile, analysis.file_id)
    images = json.loads(analysis.extra_images_json) if analysis.extra_images_json else []
    config = json.loads(analysis.config_json) if analysis.config_json else {}
    return render_template('view_strain_map.html', analysis=analysis, data_file=data_file, images=images, config=config)


@app.route('/characterizations/data/<int:file_id>/defects/add', methods=['POST'])
def add_defect_annotation(file_id):
    get_owned_or_404(DataFile, file_id)
    try:
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
    except (TypeError, ValueError):
        return redirect(url_for('technique_workspace', slug='tem', tab='Defects'))

    defect_type = request.form.get('defect_type', 'other')
    if defect_type not in DEFECT_TYPES:
        defect_type = 'other'
    note = request.form.get('note', '').strip() or None

    db.session.add(DefectAnnotation(file_id=file_id, x=x, y=y, defect_type=defect_type, note=note))
    db.session.commit()
    return redirect(url_for('technique_workspace', slug='tem', tab='Defects'))


@app.route('/characterizations/data/defects/<int:annotation_id>/delete', methods=['POST'])
def delete_defect_annotation(annotation_id):
    annotation = DefectAnnotation.query.get_or_404(annotation_id)
    get_owned_or_404(DataFile, annotation.file_id)
    db.session.delete(annotation)
    db.session.commit()
    return redirect(url_for('technique_workspace', slug='tem', tab='Defects'))


@app.route('/characterizations/data/<int:file_id>/delete', methods=['POST'])
def delete_data_file(file_id):
    data_file = get_owned_or_404(DataFile, file_id)
    technique_name = data_file.technique_name
    file_type = data_file.file_type
    for fname in [data_file.stored_filename, data_file.plot_filename, data_file.preview_filename,
                  data_file.height_data_filename]:
        if fname:
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    db.session.delete(data_file)
    db.session.commit()

    redirect_slug = request.form.get('slug', '').strip()
    redirect_tab = request.form.get('tab', '').strip()
    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        if technique_name == 'AFM':
            select_tab = 'Select data'
        else:
            select_tab = 'Select images' if file_type == 'image' else 'Select files'
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab=redirect_tab or select_tab))

    return redirect(url_for('data_interpretation_workspace', tab=redirect_tab or 'select'))


@app.route('/characterizations/data/bulk-delete', methods=['POST'])
def bulk_delete_data_files():
    file_ids = [int(i) for i in request.form.getlist('file_ids') if i.isdigit()]
    technique_name = None
    file_type = None
    for fid in file_ids:
        data_file = DataFile.query.get(fid)
        if not data_file or data_file.user_id != session['user_id']:
            continue
        technique_name = data_file.technique_name
        file_type = data_file.file_type
        for fname in [data_file.stored_filename, data_file.plot_filename, data_file.preview_filename,
                      data_file.height_data_filename]:
            if fname:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
        db.session.delete(data_file)
    db.session.commit()

    redirect_slug = request.form.get('slug', '').strip()
    redirect_tab = request.form.get('tab', '').strip()
    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        select_tab = redirect_tab or ('Select data' if technique_name == 'AFM' else ('Select images' if file_type == 'image' else 'Select files'))
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab=select_tab))

    return redirect(url_for('data_interpretation_workspace', tab=redirect_tab or 'select'))


@app.route('/characterizations/data/overlay', methods=['GET', 'POST'])
def select_overlay_files():
    """Step 1: pick which uploaded tabular files to overlay together."""
    tabular_files = DataFile.query.filter_by(file_type='tabular', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()

    if request.method == 'POST':
        selected_ids = request.form.getlist('file_ids')
        if len(selected_ids) < 2:
            return render_template(
                'select_overlay_files.html',
                files=tabular_files,
                error="Pick at least 2 files to overlay.",
            )
        ids_str = ",".join(selected_ids)
        return redirect(url_for('configure_overlay', ids=ids_str))

    return render_template('select_overlay_files.html', files=tabular_files)


@app.route('/characterizations/data/overlay/configure')
def configure_overlay():
    """Step 2: review auto-picked X/Y columns and choose a fit type for each selected file."""
    ids_str = request.args.get('ids', '')
    file_ids = [int(i) for i in ids_str.split(',') if i.strip().isdigit()]
    files = DataFile.query.filter(DataFile.id.in_(file_ids), DataFile.user_id == session['user_id']).all()
    # preserve the order the user selected them in
    files_by_id = {f.id: f for f in files}
    ordered_files = [files_by_id[i] for i in file_ids if i in files_by_id]

    file_columns = []
    for f in ordered_files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()

        try:
            df = read_tabular_file(filepath, ext, delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'), header_row_override=f.parse_header_row)
            x_col, y_col = pick_xy_columns(df)
            read_error = None if x_col else "Couldn't find two columns with usable numeric data."
        except Exception as e:
            x_col, y_col = None, None
            read_error = str(e)

        file_columns.append({'file': f, 'x_col': x_col, 'y_col': y_col, 'error': read_error})

    return render_template('configure_overlay.html', file_columns=file_columns, ids_str=ids_str)

    return render_template('configure_overlay.html', file_columns=file_columns, ids_str=ids_str)


@app.route('/characterizations/data/overlay/generate', methods=['POST'])
def generate_overlay():
    """Step 3: build the combined plot from the chosen per-file columns/fits."""
    file_ids = request.form.getlist('file_id')
    label = request.form.get('label', '').strip()

    series_list = []
    config_entries = []
    errors = []

    for fid in file_ids:
        f = DataFile.query.get(int(fid))
        if not f or f.user_id != session['user_id']:
            continue

        x_col = request.form.get(f'x_column_{fid}')
        y_col = request.form.get(f'y_column_{fid}')
        fit_type = request.form.get(f'fit_type_{fid}', 'none')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.stored_filename)
        ext = os.path.splitext(f.stored_filename)[1].lower()
        try:
            df = read_tabular_file(filepath, ext, delimiter_override=DELIMITER_MAP.get(f.parse_delimiter or 'auto'), header_row_override=f.parse_header_row)
            x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
            y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 2:
                errors.append(f"{f.original_filename}: columns '{x_col}'/'{y_col}' had fewer than 2 valid numeric rows after parsing.")
                continue
        except Exception as e:
            errors.append(f"{f.original_filename}: {e}")
            continue

        series_label = f.label or f.original_filename
        series_list.append({'x': x, 'y': y, 'label': series_label, 'fit_type': fit_type})
        config_entries.append({'file_id': f.id, 'x_column': x_col, 'y_column': y_col, 'fit_type': fit_type})

    if len(series_list) < 2:
        error_detail = " ".join(errors) if errors else "At least 2 files need valid numeric data in the chosen columns."
        return render_template('select_overlay_files.html',
                                files=DataFile.query.filter_by(file_type='tabular', user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all(),
                                error=f"Couldn't generate an overlay — {error_detail}")

    plot_filename = f"overlay_{int(datetime.now().timestamp())}.png"
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    fit_results = generate_overlay_plot(series_list, plot_path)

    overlay = OverlayPlot(
        user_id=session['user_id'],
        label=label or None,
        config_json=json.dumps(config_entries),
        fit_results_json=json.dumps(fit_results),
        plot_filename=plot_filename,
    )
    db.session.add(overlay)
    db.session.commit()

    return redirect(url_for('view_overlay', overlay_id=overlay.id))


@app.route('/characterizations/data/overlay/<int:overlay_id>')
def view_overlay(overlay_id):
    overlay = get_owned_or_404(OverlayPlot, overlay_id)
    fit_results = json.loads(overlay.fit_results_json) if overlay.fit_results_json else []
    for d in fit_results:
        d['analysis'] = generate_fit_analysis(d.get('fit_type'), d.get('fit_params'), d.get('r_squared'), y_col=d.get('label', 'Y'))
    return render_template('view_overlay.html', overlay=overlay, details=fit_results)


@app.route('/characterizations/data/overlay/<int:overlay_id>/delete', methods=['POST'])
def delete_overlay(overlay_id):
    overlay = get_owned_or_404(OverlayPlot, overlay_id)
    if overlay.plot_filename:
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], overlay.plot_filename)
        if os.path.exists(fpath):
            os.remove(fpath)
    db.session.delete(overlay)
    db.session.commit()
    return redirect(url_for('data_interpretation_workspace'))


STOPWORDS = set("""
a an the and or but if while is are was were be been being of to in on for with
at by from as this that these those it its it's into over under again further
then once here there all any both each few more most other some such no nor not
only own same so than too very can will just should now not
""".split())


def extract_text_from_pdf(file_storage, max_pages=25):
    text_parts = []
    with pdfplumber.open(file_storage) as pdf:
        for page in pdf.pages[:max_pages]:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


FACT_SIGNAL_WORDS = {
    "showed", "shows", "found", "demonstrated", "demonstrates", "achieved", "observed",
    "increased", "decreased", "improved", "reduced", "reported", "measured", "resulted",
    "concluded", "revealed", "confirmed", "indicated", "yielded", "obtained", "exhibited",
}


CITATION_FRAGMENT_RE = re.compile(
    r'[A-Z][a-zA-Z\-]+\s+[A-Z]\.(\s?[A-Z]\.)*,?\s*(et al\.?)?\s*\(?(19|20)\d{2}\)?'
)
# the naive "split on '. '" below can't tell an abbreviation from a sentence end, so a citation
# like "...by Giannazzo F. 2013, ..." gets cut right after the initial — catch what's left over:
# a short fragment that ENDS in "Surname X." with no year in view
TRAILING_INITIAL_RE = re.compile(r'\b[A-Z][a-zA-Z\-]+,?\s+[A-Z]\.$')


def strip_references_section(text):
    """Cuts full paper text off at the start of its References/Bibliography section, which is
    dense with author-name-and-year fragments that otherwise score artificially high in the
    extractive summarizer below (lots of keyword repetition, no real findings). Only cuts if
    the heading shows up well into the text, so a false match near the top can't gut it."""
    match = re.search(r'\n\s*(references|bibliography|works cited)\s*\n', text, re.IGNORECASE)
    if match and match.start() > 500:
        return text[:match.start()]
    return text


def simple_extractive_summary(text, num_sentences=5):
    """Pulls out short, fact-dense sentences using word-frequency + heuristic scoring.
    Not real AI comprehension — just a lightweight local heuristic since no API key is available."""
    text = re.sub(r'\s+', ' ', text).strip()
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        length = len(s)
        # keep only reasonably short, self-contained sentences — long ones tend to be
        # discussion/methods prose rather than standalone facts
        if 30 <= length <= 220:
            # skip likely reference/citation-list lines (heavy on brackets, numbers, "et al.",
            # "Surname Initial. Year" patterns, or too digit-heavy — all characteristic of a
            # bibliography line rather than a real sentence, which full-PDF text is full of)
            bracket_ratio = (s.count('[') + s.count('(')) / max(length, 1)
            digit_ratio = sum(c.isdigit() for c in s) / max(length, 1)
            words_list = s.split()
            word_count = len(re.findall(r'[a-zA-Z]{2,}', s))
            looks_like_citation = CITATION_FRAGMENT_RE.search(s) is not None
            starts_lowercase = s[0].islower()
            ends_with_initial = TRAILING_INITIAL_RE.search(s) is not None
            has_pdf_artifact = '(cid:' in s
            # a paper-title-style string of citations reads as almost every word capitalized
            # ("Quantum Capacitance Limited Vertical Xu H.") — real sentences aren't like that
            cap_ratio = (sum(1 for w in words_list[1:] if w[:1].isupper()) / max(len(words_list) - 1, 1)) if len(words_list) >= 4 else 0
            looks_like_title_list = cap_ratio > 0.6 and len(words_list) <= 12
            if (bracket_ratio < 0.03 and digit_ratio < 0.12 and word_count >= 5
                    and s.lower().count("et al.") < 2 and not looks_like_citation and not starts_lowercase
                    and not ends_with_initial and not has_pdf_artifact and not looks_like_title_list):
                sentences.append(s)

    if not sentences:
        # fall back to the original (longer) sentences if strict filtering removed everything
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 30][:num_sentences]
        if not sentences:
            return "Could not extract readable text from this document."

    word_freq = {}
    for sentence in sentences:
        words = re.findall(r'[a-zA-Z]{3,}', sentence.lower())
        for w in words:
            if w not in STOPWORDS:
                word_freq[w] = word_freq.get(w, 0) + 1

    if not word_freq:
        top_sentences = sentences[:num_sentences]
    else:
        max_freq = max(word_freq.values())
        for w in word_freq:
            word_freq[w] /= max_freq

        scored = []
        for idx, sentence in enumerate(sentences):
            words = re.findall(r'[a-zA-Z]{3,}', sentence.lower())
            if not words:
                continue
            # average word score keeps short punchy sentences competitive against long ones
            avg_score = sum(word_freq.get(w, 0) for w in words) / len(words)

            # bonus for sentences containing result/finding language — these read as "facts"
            fact_bonus = 1.4 if any(w in FACT_SIGNAL_WORDS for w in words) else 1.0

            # mild preference for earlier sentences (abstracts/intros/conclusions are fact-dense)
            position_bonus = 1.15 if idx < 8 else 1.0

            scored.append((avg_score * fact_bonus * position_bonus, idx, sentence))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:num_sentences]
        top_sentences = [sentence for _, idx, sentence in sorted(top, key=lambda x: x[1])]

    return "\n".join(f"• {s}" for s in top_sentences)


def lookup_semantic_scholar_by_url(url):
    """Look up a paper on Semantic Scholar using its URL/DOI, returning title + tl;dr/abstract if found."""
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/URL:{url}"
    for attempt in range(3):
        try:
            resp = requests.get(
                api_url,
                params={"fields": "title,abstract,tldr,citationCount,fieldsOfStudy,openAccessPdf"},
                headers={"User-Agent": "LabLogbook/1.0"},
                timeout=10,
            )
        except requests.RequestException:
            return None

        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title"),
                "tldr": data.get("tldr", {}).get("text") if data.get("tldr") else None,
                "abstract": data.get("abstract"),
                "citations": data.get("citationCount"),
                "fields_of_study": data.get("fieldsOfStudy") or [],
                "pdf_url": (data.get("openAccessPdf") or {}).get("url"),
            }

        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue

        break
    return None


def fetch_pdf_text(pdf_url, max_pages=15, max_bytes=20_000_000):
    """Best-effort full-text fetch for an open-access PDF, so the analysis can be based on
    the whole paper rather than just its abstract. Fails silently (returns None) on anything
    unexpected — paywalls, timeouts, non-PDF responses — since this is only ever a bonus on
    top of the abstract-based summary, never a hard requirement."""
    if not pdf_url:
        return None
    try:
        resp = requests.get(pdf_url, timeout=15, headers={"User-Agent": "LabLogbook/1.0"})
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            return None
        if len(resp.content) > max_bytes:
            return None
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            parts = []
            for page in pdf.pages[:max_pages]:
                text = page.extract_text()
                if text:
                    parts.append(text)
        full_text = "\n".join(parts) if parts else None
        return strip_references_section(full_text) if full_text else None
    except Exception:
        return None


def search_crossref(query, limit=8):
    """CrossRef indexes nearly all major journals (ACS, RSC, Wiley, Elsevier, IEEE, Springer, Nature) — very reliable, no key needed.
    Filtered to actual journal articles with real authors, since CrossRef also returns low-quality noise
    (bare journal titles, unnamed book-chapter entries) that aren't useful search results."""
    results = []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": limit * 3,   # over-fetch since we'll filter a chunk out
                "filter": "type:journal-article",
                "select": "title,author,issued,container-title,URL,DOI,abstract,is-referenced-by-count",
                "sort": "relevance",
            },
            headers={"User-Agent": "LabLogbook/1.0 (mailto:example@example.com)"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                title_list = item.get("title") or []
                if not title_list or len(title_list[0].strip()) < 15:
                    continue

                authors = []
                for a in item.get("author", [])[:5]:
                    name = " ".join(filter(None, [a.get("given"), a.get("family")]))
                    if name:
                        authors.append({"name": name})

                # skip entries with no real author list — usually junk/incomplete metadata
                if not authors:
                    continue

                venue = (item.get("container-title") or [None])[0]
                # skip entries where the "article" is really just a bare journal-title stub
                if venue and title_list[0].strip().lower() == venue.strip().lower():
                    continue

                year = None
                date_parts = item.get("issued", {}).get("date-parts", [[None]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]

                results.append({
                    "title": title_list[0],
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "url": item.get("URL"),
                    "abstract": (item.get("abstract") or "").replace("<jats:p>", "").replace("</jats:p>", "").strip() or None,
                    "tldr": None,
                    "citations": item.get("is-referenced-by-count"),
                    "fields_of_study": [],
                    "pdf_url": None,
                    "source": "CrossRef",
                })

                if len(results) >= limit:
                    break
    except requests.RequestException:
        pass
    return results


def search_arxiv(query, limit=5):
    """arXiv covers physics, CS, math, and related preprints."""
    results = []
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                link_el = entry.find("atom:id", ns)
                published_el = entry.find("atom:published", ns)
                authors = [
                    {"name": a.find("atom:name", ns).text}
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ][:5]
                year = published_el.text[:4] if published_el is not None and published_el.text else None
                link = link_el.text if link_el is not None else None
                results.append({
                    "title": title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled",
                    "authors": authors,
                    "year": year,
                    "venue": "arXiv",
                    "url": link,
                    "abstract": summary_el.text.strip().replace("\n", " ") if summary_el is not None else None,
                    "tldr": None,
                    "source": "arXiv",
                    "citations": None,
                    "fields_of_study": [],
                    # arXiv abstract pages are at /abs/<id> — the PDF is the same path under /pdf/
                    "pdf_url": link.replace('/abs/', '/pdf/') if link and '/abs/' in link else None,
                })
    except (requests.RequestException, ET.ParseError):
        pass
    return results


def search_semantic_scholar(query, limit=8):
    results = []
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,url,venue,tldr,externalIds,openAccessPdf,citationCount,fieldsOfStudy",
    }

    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                headers={"User-Agent": "LabLogbook/1.0"},
                timeout=10,
            )
        except requests.RequestException:
            break

        if resp.status_code == 200:
            for paper in resp.json().get("data", []):
                # fall back to a usable link if 'url' is missing
                if not paper.get("url"):
                    if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
                        paper["url"] = paper["openAccessPdf"]["url"]
                    elif paper.get("externalIds", {}).get("DOI"):
                        paper["url"] = f"https://doi.org/{paper['externalIds']['DOI']}"
                paper["source"] = "Semantic Scholar"
                paper["citations"] = paper.get("citationCount")
                paper["fields_of_study"] = paper.get("fieldsOfStudy") or []
                paper["pdf_url"] = (paper.get("openAccessPdf") or {}).get("url")
                results.append(paper)
            break

        if resp.status_code == 429:
            # rate-limited — brief backoff and retry, a couple of times
            time.sleep(1.5 * (attempt + 1))
            continue

        break

    return results


def search_all_sources(query, limit_per_source=8):
    """Combine multiple free, no-key literature sources for broader coverage than any single one."""
    combined = []
    combined.extend(search_semantic_scholar(query, limit_per_source))
    combined.extend(search_crossref(query, limit_per_source))
    combined.extend(search_arxiv(query, min(limit_per_source, 10)))

    # de-duplicate by normalized title
    seen = set()
    deduped = []
    for paper in combined:
        title = (paper.get("title") or "").strip().lower()
        key = re.sub(r'[^a-z0-9]', '', title)[:80]
        if key and key not in seen:
            seen.add(key)
            deduped.append(paper)

    return deduped


@app.route('/misc-analyse', methods=['POST'])
def misc_analyse():
    """AJAX endpoint: given a paper link or an uploaded PDF, return extracted key facts."""
    paper_link = request.form.get('paper_link', '').strip()
    uploaded_file = request.files.get('pdf_file')

    if uploaded_file and uploaded_file.filename:
        try:
            text = extract_text_from_pdf(uploaded_file)
            if not text.strip():
                return jsonify({"success": False, "message": "Couldn't read any text from this PDF (it may be a scanned image)."})
            summary = simple_extractive_summary(text)
            return jsonify({"success": True, "summary": summary, "source": "pdf"})
        except Exception:
            return jsonify({"success": False, "message": "Failed to process this PDF file."})

    if paper_link:
        result = lookup_semantic_scholar_by_url(paper_link)
        summary_text = (result.get("tldr") or result.get("abstract")) if result else None
        if summary_text:
            payload = {"success": True, "summary": summary_text, "source": "semantic_scholar"}
            if result.get("title"):
                payload["title"] = result["title"]
            return jsonify(payload)
        return jsonify({"success": False, "message": "Couldn't find this paper on Semantic Scholar. Try uploading the PDF instead."})

    return jsonify({"success": False, "message": "Provide a paper link or upload a PDF first."})


SEARCH_CACHE = {}    # normalized query -> {'results': [...], 'ts': float}
SEARCH_CACHE_MAX = 50
SEARCH_CACHE_TTL = 3600
SEARCH_PAGE_SIZE = 10

# Best-effort venue -> publisher-family classifier, for the journal filter. CrossRef/Semantic
# Scholar/arXiv only give us the journal/venue NAME (e.g. "Nanoscale"), not the publisher, so this
# is a keyword heuristic over well-known journal names rather than a real publisher lookup —
# anything it doesn't recognize falls into 'Other'.
JOURNAL_GROUP_KEYWORDS = [
    ('ACS', ['acs ', 'journal of the american chemical society', 'jacs', 'chemistry of materials',
             'analytical chemistry', 'nano letters', 'langmuir', 'inorganic chemistry',
             'j. am. chem. soc', 'accounts of chemical research', 'chemical reviews']),
    ('RSC', ['royal society of chemistry', 'chemical communications', 'chem. commun', 'nanoscale',
             'journal of materials chemistry', 'dalton transactions', 'soft matter', 'analyst',
             'green chemistry', 'materials horizons', 'rsc advances', 'chemical science']),
    ('Wiley', ['wiley', 'advanced materials', 'angewandte chemie', 'advanced functional materials',
               'chemistry a european journal', 'advanced energy materials', 'advanced science',
               'small', 'macromolecular']),
    ('Nature', ['nature', 'npj ', 'scientific reports']),
    ('Elsevier', ['elsevier', 'materials today', 'applied surface science',
                  'journal of colloid and interface science', 'chemical engineering journal',
                  'sensors and actuators', 'electrochimica acta', 'computational and theoretical chemistry',
                  'carbon', 'biomaterials']),
    ('Springer', ['springer', 'microchimica acta', 'journal of materials science', 'nano research',
                  'applied physics a', 'colloid and polymer science']),
    ('IEEE', ['ieee']),
]


def classify_journal_group(venue, source):
    if source == 'arXiv':
        return 'arXiv'
    if not venue:
        return 'Other'
    v = venue.lower()
    for group, keywords in JOURNAL_GROUP_KEYWORDS:
        if any(kw in v for kw in keywords):
            return group
    return 'Other'


def get_search_results(query):
    """Cache the full (multi-source) result set per query so paging/filtering through results
    doesn't re-hit the external APIs — and doesn't cost a fresh set of rate-limited
    Semantic Scholar calls — every time the user clicks a page number or a filter."""
    key = query.lower()
    cached = SEARCH_CACHE.get(key)
    if cached and time.time() - cached['ts'] < SEARCH_CACHE_TTL:
        return cached['results']

    results = search_all_sources(query, limit_per_source=25)
    for paper in results:
        paper['journal_group'] = classify_journal_group(paper.get('venue'), paper.get('source'))
        # normalize year to int — CrossRef/Semantic Scholar give an int, arXiv gives a 4-char string
        try:
            paper['year'] = int(paper['year']) if paper.get('year') else None
        except (TypeError, ValueError):
            paper['year'] = None

    if len(SEARCH_CACHE) >= SEARCH_CACHE_MAX:
        oldest_key = min(SEARCH_CACHE, key=lambda k: SEARCH_CACHE[k]['ts'])
        del SEARCH_CACHE[oldest_key]
    SEARCH_CACHE[key] = {'results': results, 'ts': time.time()}
    return results


RECENT_SEARCHES_MAX = 8


@app.route('/search-papers')
def search_papers():
    query = request.args.get('q', '').strip()
    page = max(1, request.args.get('page', 1, type=int) or 1)
    filter_year = request.args.get('year', '').strip()
    filter_journal = request.args.get('journal', '').strip()
    filter_source = request.args.get('source', '').strip()
    min_citations = request.args.get('min_citations', type=int)

    if query:
        recent = session.get('recent_searches', [])
        recent = [q for q in recent if q.lower() != query.lower()]
        recent.insert(0, query)
        session['recent_searches'] = recent[:RECENT_SEARCHES_MAX]

    recent_searches = [q for q in session.get('recent_searches', []) if q.lower() != query.lower()]

    all_results = get_search_results(query) if query else []

    filtered = all_results
    if filter_year:
        filtered = [p for p in filtered if str(p.get('year') or '') == filter_year]
    if filter_journal:
        filtered = [p for p in filtered if p['journal_group'] == filter_journal]
    if filter_source:
        filtered = [p for p in filtered if p.get('source') == filter_source]
    if min_citations is not None:
        filtered = [p for p in filtered if (p.get('citations') or 0) >= min_citations]

    total_results = len(filtered)
    total_pages = max(1, -(-total_results // SEARCH_PAGE_SIZE))   # ceil division
    page = min(page, total_pages)

    start = (page - 1) * SEARCH_PAGE_SIZE
    results = filtered[start:start + SEARCH_PAGE_SIZE]

    # filter option lists reflect the full fetched pool (not just the filtered set), so
    # switching one filter doesn't make the others disappear
    available_years = sorted({p['year'] for p in all_results if p.get('year')}, reverse=True)
    available_journals = sorted({p['journal_group'] for p in all_results})
    available_sources = sorted({p['source'] for p in all_results if p.get('source')})

    return render_template(
        'search_papers.html', query=query, results=results, page_title='Search Papers',
        page=page, total_pages=total_pages, total_results=total_results,
        filter_year=filter_year, filter_journal=filter_journal, filter_source=filter_source,
        min_citations=min_citations,
        available_years=available_years, available_journals=available_journals, available_sources=available_sources,
        recent_searches=recent_searches,
    )


@app.route('/search-papers/analyze', methods=['GET', 'POST'])
def analyze_search_result():
    """Renders a standalone analysis page for one search result (opened in a new tab via a
    click-handled window.open(), with a same-tab fallback — see search_papers.html). Accepts
    GET too, defensively, in case some environment's handling of that drops the method.

    Builds the richest analysis it can from free sources: a real TL;DR from Semantic Scholar
    when available, a multi-point extractive summary — run over the full paper text when an
    open-access PDF can be fetched, falling back to the abstract otherwise — plus citation
    count and fields of study, so the page holds more than just the abstract restated."""
    paper_title = request.values.get('paper_title', '').strip()
    paper_link = request.values.get('paper_link', '').strip()
    abstract = request.values.get('abstract', '').strip()
    venue = request.values.get('venue', '').strip()
    authors = request.values.get('authors', '').strip()
    year = request.values.get('year', '').strip()
    query = request.values.get('q', '').strip()
    citations = request.values.get('citations', '').strip()
    fields_of_study = [f.strip() for f in request.values.get('fields_of_study', '').split(',') if f.strip()]
    pdf_url = request.values.get('pdf_url', '').strip()

    tldr = None
    if paper_link:
        result = lookup_semantic_scholar_by_url(paper_link)
        if result:
            tldr = result.get('tldr')
            abstract = abstract or result.get('abstract') or ''
            pdf_url = pdf_url or result.get('pdf_url') or ''
            if not citations and result.get('citations') is not None:
                citations = str(result['citations'])
            if not fields_of_study and result.get('fields_of_study'):
                fields_of_study = result['fields_of_study']

    # abstract-based bullets are the reliable baseline — always compute them when there's an
    # abstract. Full-text bullets are a bonus on top when an open-access PDF is fetchable, shown
    # as a separate section rather than replacing the abstract one (full-PDF extraction is noisier
    # — figure captions, garbled multi-column reading order — so it shouldn't be the only summary).
    abstract_points = simple_extractive_summary(abstract, num_sentences=6) if abstract else None

    full_text = fetch_pdf_text(pdf_url) if pdf_url else None
    full_text_points = simple_extractive_summary(full_text, num_sentences=8) if full_text else None

    key_facts = '\n\n'.join(filter(None, [tldr, abstract_points, full_text_points]))

    return render_template(
        'paper_analysis.html', page_title='Paper Analysis',
        paper_title=paper_title, paper_link=paper_link, abstract=abstract,
        venue=venue, authors=authors, year=year, query=query,
        tldr=tldr, abstract_points=abstract_points, full_text_points=full_text_points, key_facts=key_facts,
        citations=citations, fields_of_study=fields_of_study, pdf_url=pdf_url,
    )



def fill_misc_entry_from_form(entry, existing_pdf_filename=None):
    date_str = request.form.get('entry_date')
    entry.entry_date = datetime.fromisoformat(date_str).date() if date_str else datetime.now().date()
    entry.research_topic = request.form['research_topic']
    entry.journal = request.form.get('journal', '')
    entry.paper_title = request.form.get('paper_title', '')
    entry.paper_link = request.form.get('paper_link', '')
    entry.key_facts = request.form.get('key_facts', '')

    # handle optional PDF upload — keep the existing file if none is uploaded this time
    uploaded_file = request.files.get('pdf_file')
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(f"{datetime.now().timestamp()}_{uploaded_file.filename}")
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        entry.pdf_filename = filename
    elif existing_pdf_filename is not None:
        entry.pdf_filename = existing_pdf_filename


@app.route('/miscellaneous', methods=['GET', 'POST'])
def miscellaneous():
    if request.method == 'POST':
        new_entry = MiscEntry(user_id=session['user_id'])
        fill_misc_entry_from_form(new_entry)
        db.session.add(new_entry)
        db.session.commit()
        return redirect(url_for('miscellaneous'))

    entries = MiscEntry.query.filter_by(user_id=session['user_id']).order_by(MiscEntry.entry_date.desc()).all()

    # group by journal, for the "papers collected" view
    grouped = {}
    for e in entries:
        grouped.setdefault(e.journal or 'Unspecified', []).append(e)

    return render_template(
        'miscellaneous.html',
        page_title='Research Papers',
        entries=entries,
        grouped_entries=grouped,
        today=datetime.now().strftime('%Y-%m-%d'),
        journal_options=JOURNAL_OPTIONS,
        banner_image='images/miscellaneous-banner.png',
    )


@app.route('/misc-entry/<int:entry_id>/edit', methods=['GET', 'POST'])
def edit_misc_entry(entry_id):
    entry = get_owned_or_404(MiscEntry, entry_id)

    if request.method == 'POST':
        fill_misc_entry_from_form(entry, existing_pdf_filename=entry.pdf_filename)
        db.session.commit()
        return redirect(url_for('miscellaneous'))

    return render_template(
        'edit_misc_entry.html',
        entry=entry,
        journal_options=JOURNAL_OPTIONS,
    )


@app.route('/misc-entry/<int:entry_id>/delete', methods=['POST'])
def delete_misc_entry(entry_id):
    entry = get_owned_or_404(MiscEntry, entry_id)
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for('miscellaneous'))


@app.route('/entry/<int:entry_id>/edit', methods=['GET', 'POST'])
def edit_entry(entry_id):
    entry = get_owned_or_404(LogEntry, entry_id)

    if request.method == 'POST':
        fill_entry_from_form(entry)
        db.session.commit()
        return redirect(url_for(entry.category))

    return render_template(
        'edit_entry.html',
        entry=entry,
        status_options=STATUS_OPTIONS,
        exp_type_options=EXP_TYPE_OPTIONS,
    )


@app.route('/entry/<int:entry_id>/delete', methods=['POST'])
def delete_entry(entry_id):
    entry = get_owned_or_404(LogEntry, entry_id)
    category = entry.category
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for(category))


@app.route('/elements')
def elements():
    tab = request.args.get('tab', 'protocols')
    if tab not in ('protocols', 'projects'):
        tab = 'protocols'

    uid = session['user_id']
    protocols = Protocol.query.filter_by(user_id=uid).order_by(Protocol.created_at.desc()).all()
    projects = Project.query.filter_by(user_id=uid).order_by(Project.created_at.desc()).all()
    experiments_list = LogEntry.query.filter_by(category='experiments', user_id=uid).order_by(LogEntry.entry_datetime.desc()).all()
    standalone_tasks = Task.query.filter_by(project_id=None, user_id=uid).order_by(Task.is_done, Task.due_date).all()

    return render_template(
        'elements.html',
        tab=tab,
        protocols=protocols,
        projects=projects,
        experiments_list=experiments_list,
        standalone_tasks=standalone_tasks,
        today=datetime.now().strftime('%Y-%m-%d'),
        banner_image='images/Elements-banner.png',
    )


@app.route('/elements/protocols/new', methods=['POST'])
def new_protocol():
    title = request.form['title'].strip()
    steps = request.form.get('steps', '').strip()
    linked_experiment_id = request.form.get('linked_experiment_id', type=int) or None

    protocol = Protocol(
        user_id=session['user_id'],
        title=title,
        equipment=(request.form.get('equipment', '').strip() or None),
        category=(request.form.get('category', '').strip() or None),
        current_version=1,
        linked_experiment_id=linked_experiment_id,
    )
    db.session.add(protocol)
    db.session.flush()
    db.session.add(ProtocolVersion(protocol_id=protocol.id, version_number=1, steps=steps, change_note='Initial version'))
    db.session.commit()
    return redirect(url_for('elements', tab='protocols'))


@app.route('/elements/protocols/<int:protocol_id>/new-version', methods=['POST'])
def new_protocol_version(protocol_id):
    protocol = get_owned_or_404(Protocol, protocol_id)
    steps = request.form.get('steps', '').strip()
    change_note = request.form.get('change_note', '').strip() or None

    protocol.current_version += 1
    db.session.add(ProtocolVersion(
        protocol_id=protocol.id,
        version_number=protocol.current_version,
        steps=steps,
        change_note=change_note,
    ))
    db.session.commit()
    return redirect(url_for('elements', tab='protocols'))


@app.route('/elements/protocols/<int:protocol_id>/clone', methods=['POST'])
def clone_protocol(protocol_id):
    source = get_owned_or_404(Protocol, protocol_id)
    latest = source.latest_version
    linked_experiment_id = request.form.get('linked_experiment_id', type=int) or None

    clone = Protocol(
        user_id=session['user_id'],
        title=source.title,
        equipment=source.equipment,
        category=source.category,
        current_version=1,
        cloned_from_id=source.id,
        linked_experiment_id=linked_experiment_id,
    )
    db.session.add(clone)
    db.session.flush()
    db.session.add(ProtocolVersion(
        protocol_id=clone.id,
        version_number=1,
        steps=(latest.steps if latest else ''),
        change_note=f'Cloned from "{source.title}" (v{latest.version_number if latest else 1})',
    ))
    db.session.commit()
    return redirect(url_for('elements', tab='protocols'))


@app.route('/elements/protocols/<int:protocol_id>/delete', methods=['POST'])
def delete_protocol(protocol_id):
    protocol = get_owned_or_404(Protocol, protocol_id)
    db.session.delete(protocol)
    db.session.commit()
    return redirect(url_for('elements', tab='protocols'))


@app.route('/elements/projects/new', methods=['POST'])
def new_project():
    def parse_date(field):
        val = request.form.get(field, '').strip()
        return datetime.strptime(val, '%Y-%m-%d').date() if val else None

    project = Project(
        user_id=session['user_id'],
        name=request.form['name'].strip(),
        description=(request.form.get('description', '').strip() or None),
        start_date=parse_date('start_date'),
        end_date=parse_date('end_date'),
        grant_name=(request.form.get('grant_name', '').strip() or None),
        funding_start=parse_date('funding_start'),
        funding_end=parse_date('funding_end'),
        funding_amount=(request.form.get('funding_amount', type=float) or None),
    )
    db.session.add(project)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/projects/<int:project_id>/delete', methods=['POST'])
def delete_project(project_id):
    project = get_owned_or_404(Project, project_id)
    db.session.delete(project)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/milestones/new', methods=['POST'])
def new_milestone():
    project = get_owned_or_404(Project, request.form.get('project_id', type=int))
    due_date_str = request.form.get('due_date', '').strip()
    milestone = Milestone(
        project_id=project.id,
        title=request.form['title'].strip(),
        due_date=(datetime.strptime(due_date_str, '%Y-%m-%d').date() if due_date_str else None),
        deliverable=(request.form.get('deliverable', '').strip() or None),
    )
    db.session.add(milestone)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/milestones/<int:milestone_id>/toggle', methods=['POST'])
def toggle_milestone(milestone_id):
    milestone = Milestone.query.get_or_404(milestone_id)
    if milestone.project.user_id != session['user_id']:
        abort(404)
    milestone.is_done = not milestone.is_done
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/milestones/<int:milestone_id>/delete', methods=['POST'])
def delete_milestone(milestone_id):
    milestone = Milestone.query.get_or_404(milestone_id)
    if milestone.project.user_id != session['user_id']:
        abort(404)
    db.session.delete(milestone)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/tasks/new', methods=['POST'])
def new_task():
    due_date_str = request.form.get('due_date', '').strip()
    project_id = request.form.get('project_id', type=int) or None
    if project_id is not None:
        get_owned_or_404(Project, project_id)
    task = Task(
        user_id=session['user_id'],
        title=request.form['title'].strip(),
        due_date=(datetime.strptime(due_date_str, '%Y-%m-%d').date() if due_date_str else None),
        project_id=project_id,
        linked_experiment_id=(request.form.get('linked_experiment_id', type=int) or None),
    )
    db.session.add(task)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/tasks/<int:task_id>/toggle', methods=['POST'])
def toggle_task(task_id):
    task = get_owned_or_404(Task, task_id)
    task.is_done = not task.is_done
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/elements/tasks/<int:task_id>/delete', methods=['POST'])
def delete_task(task_id):
    task = get_owned_or_404(Task, task_id)
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for('elements', tab='projects'))


@app.route('/docs')
def docs():
    return render_template('docs.html')


@app.route('/shortcuts')
def shortcuts():
    return render_template('shortcuts.html')


@app.route('/data-privacy')
def data_privacy():
    return render_template('data_privacy.html')


@app.route('/feedback', methods=['GET', 'POST'])
@limiter.limit('20 per hour', methods=['POST'])
def feedback():
    submitted = False
    if request.method == 'POST':
        entry = Feedback(
            category=(request.form.get('category') or 'bug'),
            message=request.form['message'].strip(),
            email=(request.form.get('email', '').strip() or None),
            page_url=(request.form.get('page_url', '').strip() or None),
        )
        db.session.add(entry)
        db.session.commit()
        submitted = True
    return render_template('feedback.html', submitted=submitted)


def _model_to_dict(obj):
    result = {}
    for col in obj.__table__.columns:
        val = getattr(obj, col.name)
        if hasattr(val, 'isoformat'):
            val = val.isoformat()
        result[col.name] = val
    return result


@app.route('/export-data')
def export_data():
    uid = session['user_id']
    payload = {
        'exported_at': datetime.now().isoformat(),
        'app_version': APP_VERSION,
        'experiments': [_model_to_dict(e) for e in LogEntry.query.filter_by(category='experiments', user_id=uid).all()],
        'characterizations': [_model_to_dict(e) for e in LogEntry.query.filter_by(category='characterizations', user_id=uid).all()],
        'research_papers': [_model_to_dict(e) for e in MiscEntry.query.filter_by(user_id=uid).all()],
        'protocols': [
            dict(_model_to_dict(p), versions=[_model_to_dict(v) for v in p.versions])
            for p in Protocol.query.filter_by(user_id=uid).all()
        ],
        'projects': [
            dict(_model_to_dict(pr), milestones=[_model_to_dict(m) for m in pr.milestones])
            for pr in Project.query.filter_by(user_id=uid).all()
        ],
        'tasks': [_model_to_dict(t) for t in Task.query.filter_by(user_id=uid).all()],
        'samples': [
            dict(_model_to_dict(s), properties=[_model_to_dict(p) for p in s.properties])
            for s in Sample.query.filter_by(user_id=uid).all()
        ],
        'analysis_snapshots': [_model_to_dict(s) for s in AnalysisSnapshot.query.filter_by(user_id=uid).all()],
    }
    buf = io.BytesIO(json.dumps(payload, indent=2, default=str).encode('utf-8'))
    return send_file(
        buf,
        mimetype='application/json',
        as_attachment=True,
        download_name=f'lablogbook-export-{datetime.now().strftime("%Y%m%d-%H%M%S")}.json',
    )


@app.route('/samples')
def samples():
    all_samples = (
        Sample.query.filter_by(user_id=session['user_id'])
        .order_by(Sample.created_at.desc())
        .all()
    )
    return render_template('samples.html', samples=all_samples)


@app.route('/samples/new', methods=['POST'])
def new_sample():
    sample = Sample(
        user_id=session['user_id'],
        name=request.form['name'].strip(),
        description=(request.form.get('description', '').strip() or None),
    )
    db.session.add(sample)
    db.session.commit()
    return redirect(url_for('sample_detail', sample_id=sample.id))


@app.route('/samples/<int:sample_id>')
def sample_detail(sample_id):
    sample = get_owned_or_404(Sample, sample_id)
    by_technique = {}
    for prop in sample.properties:
        by_technique.setdefault(prop.technique_name, []).append(prop)
    return render_template('sample_detail.html', sample=sample, by_technique=by_technique)


@app.route('/samples/<int:sample_id>/delete', methods=['POST'])
def delete_sample(sample_id):
    sample = get_owned_or_404(Sample, sample_id)
    db.session.delete(sample)
    db.session.commit()
    return redirect(url_for('samples'))


@app.route('/samples/<int:sample_id>/properties/new', methods=['POST'])
def new_sample_property(sample_id):
    sample = get_owned_or_404(Sample, sample_id)
    prop = SampleProperty(
        sample_id=sample.id,
        technique_name=request.form['technique_name'].strip(),
        property_name=request.form['property_name'].strip(),
        value=request.form.get('value', type=float),
        unit=(request.form.get('unit', '').strip() or None),
        note=(request.form.get('note', '').strip() or None),
    )
    db.session.add(prop)
    db.session.commit()
    return redirect(url_for('sample_detail', sample_id=sample.id))


@app.route('/samples/<int:sample_id>/properties/<int:prop_id>/delete', methods=['POST'])
def delete_sample_property(sample_id, prop_id):
    sample = get_owned_or_404(Sample, sample_id)
    prop = SampleProperty.query.get_or_404(prop_id)
    if prop.sample_id != sample.id:
        abort(404)
    db.session.delete(prop)
    db.session.commit()
    return redirect(url_for('sample_detail', sample_id=sample.id))


def _sample_property_axes():
    """Distinct (technique, property, unit) combos the user has logged, used to
    populate the two correlation dropdowns — each entry becomes one selectable axis."""
    rows = (
        db.session.query(
            SampleProperty.technique_name, SampleProperty.property_name, SampleProperty.unit
        )
        .join(Sample, SampleProperty.sample_id == Sample.id)
        .filter(Sample.user_id == session['user_id'])
        .distinct()
        .order_by(SampleProperty.technique_name, SampleProperty.property_name)
        .all()
    )
    seen = {}
    for technique_name, property_name, unit in rows:
        key = f"{technique_name}|{property_name}"
        if key not in seen:
            label = f"{technique_name} — {property_name}" + (f" ({unit})" if unit else '')
            seen[key] = {'key': key, 'label': label}
    return list(seen.values())


@app.route('/samples/correlate')
def sample_correlate():
    axes = _sample_property_axes()
    axis_a = request.args.get('axis_a', '')
    axis_b = request.args.get('axis_b', '')
    plot_filename = None
    points = []
    stats = None
    error = None

    if axis_a and axis_b:
        if axis_a == axis_b:
            error = "Pick two different properties to correlate."
        else:
            tech_a, prop_a = axis_a.split('|', 1)
            tech_b, prop_b = axis_b.split('|', 1)
            user_samples = Sample.query.filter_by(user_id=session['user_id']).all()
            for sample in user_samples:
                val_a = next(
                    (p.value for p in sample.properties
                     if p.technique_name == tech_a and p.property_name == prop_a), None
                )
                val_b = next(
                    (p.value for p in sample.properties
                     if p.technique_name == tech_b and p.property_name == prop_b), None
                )
                if val_a is not None and val_b is not None:
                    points.append({'name': sample.name, 'x': val_a, 'y': val_b})

            if len(points) < 2:
                error = "Fewer than 2 samples have both properties logged — need at least 2 to plot, 3+ for a trend line."
            else:
                xs = np.array([p['x'] for p in points])
                ys = np.array([p['y'] for p in points])

                fig, ax = plt.subplots(figsize=(7, 5.5))
                ax.scatter(xs, ys, s=70, color='#2a6f2a', zorder=3)
                for p in points:
                    ax.annotate(p['name'], (p['x'], p['y']), fontsize=8,
                                xytext=(6, 6), textcoords='offset points', color='#444')

                if len(points) >= 3 and np.std(xs) > 0:
                    slope, intercept = np.polyfit(xs, ys, 1)
                    fit_x = np.linspace(xs.min(), xs.max(), 100)
                    ax.plot(fit_x, slope * fit_x + intercept, '--', color='#c0392b', linewidth=1.5, zorder=2)
                    r = np.corrcoef(xs, ys)[0, 1]
                    stats = {'r': round(float(r), 3), 'r2': round(float(r ** 2), 3), 'n': len(points)}
                else:
                    stats = {'r': None, 'r2': None, 'n': len(points)}

                ax.set_xlabel(next(a['label'] for a in axes if a['key'] == axis_a))
                ax.set_ylabel(next(a['label'] for a in axes if a['key'] == axis_b))
                ax.grid(alpha=0.25)
                fig.tight_layout()

                plot_filename = f"correlate_{int(datetime.now().timestamp())}.png"
                fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
                plt.close(fig)

    return render_template(
        'sample_correlate.html',
        axes=axes, axis_a=axis_a, axis_b=axis_b,
        plot_filename=plot_filename, points=points, stats=stats, error=error,
    )


@app.route('/samples/drift')
def sample_drift():
    axes = _sample_property_axes()
    axis = request.args.get('axis', '')
    plot_filename = None
    points = []
    drift = None
    anomalies = []
    summary = None
    error = None

    if axis:
        tech, prop = axis.split('|', 1)
        rows = (
            db.session.query(SampleProperty, Sample.name)
            .join(Sample, SampleProperty.sample_id == Sample.id)
            .filter(
                Sample.user_id == session['user_id'],
                SampleProperty.technique_name == tech,
                SampleProperty.property_name == prop,
            )
            .order_by(SampleProperty.created_at)
            .all()
        )
        points = [{'name': name, 'value': sp.value, 'created_at': sp.created_at} for sp, name in rows]

        if len(points) < 4:
            error = f"Only {len(points)} measurement(s) logged for this property — need at least 4 to check for drift or anomalies."
        else:
            t0 = points[0]['created_at']
            days = np.array([(p['created_at'] - t0).total_seconds() / 86400 for p in points])
            values = np.array([p['value'] for p in points])

            # drift: does the value trend over time, or just wobble around a constant?
            slope, intercept, r = 0.0, None, None
            if np.ptp(days) > 0:
                slope, intercept = np.polyfit(days, values, 1)
                with np.errstate(invalid='ignore'):
                    r = float(np.corrcoef(days, values)[0, 1])
                if np.isnan(r):
                    r = None
            significant = r is not None and abs(r) >= 0.6
            drift = {
                'slope': float(slope), 'r': round(r, 3) if r is not None else None,
                'direction': 'increasing' if slope > 0 else 'decreasing', 'significant': significant,
            }

            # anomalies: modified z-score (median/MAD) — robust to a single outlier skewing a
            # plain mean/std the way it would with this few points
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad > 0:
                mod_z = 0.6745 * (values - median) / mad
            else:
                std = np.std(values)
                mod_z = (values - np.mean(values)) / std if std > 0 else np.zeros_like(values)

            for i, p in enumerate(points):
                p['z'] = round(float(mod_z[i]), 2)
                if abs(mod_z[i]) > 3.5:
                    anomalies.append(p)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(days, values, '-o', color='#2a6f2a', markersize=6, linewidth=1.4, zorder=2)
            for i, p in enumerate(points):
                if abs(mod_z[i]) > 3.5:
                    ax.scatter([days[i]], [values[i]], color='#c0392b', s=120, zorder=4, marker='X')
                ax.annotate(p['name'], (days[i], values[i]), fontsize=7, xytext=(5, 5),
                            textcoords='offset points', color='#666')
            if significant:
                fit_x = np.linspace(days.min(), days.max(), 100)
                ax.plot(fit_x, slope * fit_x + intercept, '--', color='#888', linewidth=1.3, zorder=1)
            axis_label = next(a['label'] for a in axes if a['key'] == axis)
            ax.set_xlabel('Days since first measurement')
            ax.set_ylabel(axis_label)
            ax.grid(alpha=0.25)
            fig.tight_layout()

            plot_filename = f"drift_{int(datetime.now().timestamp())}.png"
            fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
            plt.close(fig)

            parts = [
                f"{len(points)} measurements of {prop} ({tech}) logged between "
                f"{points[0]['created_at'].strftime('%Y-%m-%d')} and {points[-1]['created_at'].strftime('%Y-%m-%d')}."
            ]
            if significant:
                span_days = float(np.ptp(days))
                total_change = slope * span_days
                if span_days >= 1:
                    span_desc = f"{span_days:.1f} days"
                elif span_days * 24 >= 1:
                    span_desc = f"{span_days * 24:.1f} hours"
                else:
                    span_desc = f"{span_days * 24 * 60:.0f} minutes"
                parts.append(
                    f"Values show a {drift['direction']} drift over time (r = {drift['r']}), changing by "
                    f"about {total_change:.4g} over the {span_desc} spanned by these measurements — worth "
                    f"checking instrument calibration or sample stability if that's not expected."
                )
            elif r is not None:
                parts.append(f"No significant drift over time (r = {drift['r']}) — values look stable.")
            else:
                parts.append("All measurements were logged on the same day, so a time trend can't be assessed yet.")
            if anomalies:
                names = ", ".join(f"{a['name']} ({a['value']:.4g})" for a in anomalies)
                parts.append(f"{len(anomalies)} measurement(s) stand out as anomalous versus the rest: {names}.")
            else:
                parts.append("No individual measurements stand out as anomalous.")
            summary = " ".join(parts)

    return render_template(
        'sample_drift.html',
        axes=axes, axis=axis, plot_filename=plot_filename,
        points=points, drift=drift, anomalies=anomalies, summary=summary, error=error,
    )


# ---- Collaboration: share records with people, lab groups, or anyone with a link ----------

SHAREABLE_TYPES = {
    'experiment': (LogEntry, 'Experiment'),
    'characterization': (CharEntry, 'Characterization'),
    'paper': (MiscEntry, 'Research paper'),
    'protocol': (Protocol, 'Protocol / SOP'),
    'sample': (Sample, 'Sample'),
    'snapshot': (AnalysisSnapshot, 'Analysis snapshot'),
}
SHARE_EXPIRY_OPTIONS = [('7', '7 days'), ('30', '30 days'), ('90', '90 days'), ('', 'Never')]


def current_user_email():
    email = session.get('user_email')
    if not email:
        user = db.session.get(User, session['user_id'])
        email = user.email if user else ''
    return email.strip().lower()


def _active_shares():
    return or_(Share.expires_at.is_(None), Share.expires_at > datetime.now())


def user_group_ids(email):
    return [m.group_id for m in GroupMember.query.filter_by(email=email).all()]


def item_title(item_type, obj):
    if item_type == 'experiment':
        return obj.title
    if item_type == 'characterization':
        return f"{obj.technique_name} characterization ({obj.date_scheduled.strftime('%Y-%m-%d')})"
    if item_type == 'paper':
        return obj.paper_title or obj.research_topic
    if item_type == 'protocol':
        return obj.title
    if item_type == 'sample':
        return obj.name
    return obj.title   # snapshot


def item_payload(item_type, obj):
    """A read-only, template-friendly description of any shareable item — one generic shape
    (title, meta chips, text sections, steps, table, image/pdf/link) so a single view page can
    render all of them without exposing the owner's edit controls."""
    p = {'title': item_title(item_type, obj), 'meta': [], 'sections': [], 'steps': [],
         'table': None, 'image': None, 'pdf': None, 'link': None}

    def section(heading, text):
        if text and str(text).strip():
            p['sections'].append((heading, str(text)))

    if item_type == 'experiment':
        p['meta'] = [m for m in (obj.exp_type, obj.status, obj.entry_datetime.strftime('%Y-%m-%d')) if m]
        for heading, field in (('Objective', obj.objective), ('Materials', obj.materials), ('Procedure', obj.procedure),
                               ('Input parameters', obj.input_params), ('Output / results', obj.output_results),
                               ('Observations', obj.observations), ('Conclusion', obj.conclusion)):
            section(heading, field)
    elif item_type == 'characterization':
        p['meta'] = [m for m in (obj.date_scheduled.strftime('%Y-%m-%d'),
                                 f"{obj.num_samples} sample(s)" if obj.num_samples else None,
                                 obj.sample_prep, obj.outcome) if m]
        section('Interpretation', obj.interpretation)
    elif item_type == 'paper':
        p['meta'] = [m for m in (obj.journal, obj.entry_date.strftime('%Y-%m-%d')) if m]
        section('Research topic', obj.research_topic)
        section('Key facts', obj.key_facts)
        p['link'] = obj.paper_link
        p['pdf'] = obj.pdf_filename
    elif item_type == 'protocol':
        p['meta'] = [m for m in (f"v{obj.current_version}", obj.equipment, obj.category) if m]
        latest = obj.latest_version
        if latest:
            p['steps'] = [s.strip() for s in latest.steps.split('\n') if s.strip()]
        changes = [f"v{v.version_number} ({v.created_at.strftime('%Y-%m-%d')}): {v.change_note}"
                   for v in obj.versions if v.change_note]
        section('Version notes', "\n".join(changes))
    elif item_type == 'sample':
        section('Description', obj.description)
        if obj.properties:
            p['table'] = {
                'headers': ['Technique', 'Property', 'Value', 'Note'],
                'rows': [[pr.technique_name, pr.property_name,
                          f"{pr.value:g}" + (f" {pr.unit}" if pr.unit else ''), pr.note or ''] for pr in obj.properties],
            }
    else:  # snapshot
        p['meta'] = [obj.created_at.strftime('%Y-%m-%d %H:%M'), obj.plot_type]
        p['image'] = obj.plot_filename
        section('Note', obj.note)
        section('Overall interpretation', obj.overall_analysis)
        results = json.loads(obj.results_json or '[]')
        shaped = [r for r in results if r.get('shape')]
        section('Shape suggestions', "\n".join(
            f"{r['label']} looks like {r['shape']['friendly']} (R² ≈ {r['shape']['r_squared']:.3f})" for r in shaped))
        if results:
            p['table'] = {
                'headers': ['File', 'n', 'Mean', 'Std', 'Min', 'Max'],
                'rows': [[r['label'], r['stats']['count'], r['stats']['mean'], r['stats']['std'],
                          r['stats']['min'], r['stats']['max']] for r in results],
            }
    return p


def viewer_access(item_type, item_id, owner_id):
    """'owner', 'shared' (granted by email or group, not expired), or None."""
    uid = session['user_id']
    if owner_id == uid:
        return 'owner'
    email = current_user_email()
    grants = [Share.grantee_email == email]
    gids = user_group_ids(email)
    if gids:
        grants.append(Share.group_id.in_(gids))
    hit = Share.query.filter(
        Share.item_type == item_type, Share.item_id == item_id, Share.owner_id == owner_id,
        or_(*grants), _active_shares(),
    ).first()
    return 'shared' if hit else None


def _load_item_or_404(item_type, item_id):
    if item_type not in SHAREABLE_TYPES:
        abort(404)
    obj = SHAREABLE_TYPES[item_type][0].query.get_or_404(item_id)
    return obj


def _describe_target(share):
    if share.link_token:
        return 'Anyone with the link'
    if share.group_id:
        group = db.session.get(LabGroup, share.group_id)
        return f"Group: {group.name}" if group else 'Group (deleted)'
    return share.grantee_email


@app.route('/collaborate')
def collaborate():
    uid = session['user_id']
    email = current_user_email()
    gids = user_group_ids(email)

    grants = [Share.grantee_email == email]
    if gids:
        grants.append(Share.group_id.in_(gids))
    incoming = []
    for s in Share.query.filter(Share.owner_id != uid, or_(*grants), _active_shares()).order_by(Share.created_at.desc()).all():
        obj = SHAREABLE_TYPES[s.item_type][0].query.get(s.item_id)
        owner = db.session.get(User, s.owner_id)
        if obj:
            incoming.append({'share': s, 'title': item_title(s.item_type, obj), 'kind': SHAREABLE_TYPES[s.item_type][1],
                             'owner': owner.name if owner else 'Someone'})

    outgoing = []
    for s in Share.query.filter_by(owner_id=uid).order_by(Share.created_at.desc()).all():
        obj = SHAREABLE_TYPES[s.item_type][0].query.get(s.item_id)
        outgoing.append({
            'share': s, 'title': item_title(s.item_type, obj) if obj else '(deleted item)',
            'kind': SHAREABLE_TYPES[s.item_type][1], 'target': _describe_target(s),
            'link': url_for('shared_link_view', token=s.link_token, _external=True) if s.link_token else None,
            'expired': bool(s.expires_at and s.expires_at <= datetime.now()),
        })

    my_groups = LabGroup.query.filter_by(owner_id=uid).order_by(LabGroup.created_at.desc()).all()
    member_groups = LabGroup.query.filter(LabGroup.id.in_(gids), LabGroup.owner_id != uid).all() if gids else []

    items = {}
    for item_type, (model, _) in SHAREABLE_TYPES.items():
        rows = model.query.filter_by(user_id=uid).all()
        items[item_type] = [{'id': o.id, 'title': item_title(item_type, o)} for o in rows]

    return render_template(
        'collaborate.html', incoming=incoming, outgoing=outgoing, my_groups=my_groups, member_groups=member_groups,
        share_targets=[{'id': g.id, 'name': g.name} for g in my_groups + member_groups],
        items=items, type_labels={k: v[1] for k, v in SHAREABLE_TYPES.items()}, expiry_options=SHARE_EXPIRY_OPTIONS,
        preselect_type=request.args.get('type', ''), preselect_id=request.args.get('id', type=int),
        message=session.pop('collab_msg', None), error=session.pop('collab_error', None),
    )


@app.route('/collaborate/share', methods=['POST'])
def create_share():
    uid = session['user_id']
    item_type = request.form.get('item_type', '')
    item_id = request.form.get('item_id', type=int)
    obj = _load_item_or_404(item_type, item_id)
    if obj.user_id != uid:
        abort(404)

    kind = request.form.get('kind', 'email')
    days = request.form.get('expiry_days', '')
    share = Share(
        owner_id=uid, item_type=item_type, item_id=item_id,
        note=(request.form.get('note', '').strip()[:300] or None),
        expires_at=(datetime.now() + timedelta(days=int(days))) if days.isdigit() else None,
    )

    if kind == 'link':
        share.link_token = secrets.token_urlsafe(24)
        session['collab_msg'] = 'Link created — copy it from "Shared by me" below and send it to anyone.'
    elif kind == 'group':
        group_id = request.form.get('group_id', type=int)
        group = db.session.get(LabGroup, group_id) if group_id else None
        if not group or (group.owner_id != uid and group.id not in user_group_ids(current_user_email())):
            session['collab_error'] = 'Pick one of your groups.'
            return redirect(url_for('collaborate'))
        share.group_id = group.id
        session['collab_msg'] = f'Shared with the group "{group.name}".'
    else:
        target = request.form.get('email', '').strip().lower()
        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', target):
            session['collab_error'] = "That doesn't look like an email address."
            return redirect(url_for('collaborate'))
        if target == current_user_email():
            session['collab_error'] = "That's your own email."
            return redirect(url_for('collaborate'))
        duplicate = Share.query.filter_by(owner_id=uid, item_type=item_type, item_id=item_id, grantee_email=target).first()
        if duplicate:
            duplicate.expires_at, duplicate.note = share.expires_at, share.note
            db.session.commit()
            session['collab_msg'] = f'Already shared with {target} — updated the expiry and note.'
            return redirect(url_for('collaborate'))
        share.grantee_email = target
        owner = db.session.get(User, uid)
        registered = User.query.filter(func.lower(User.email) == target).first() is not None
        access_hint = (f"Sign in and open Collaborate to see it: {url_for('collaborate', _external=True)}" if registered else
                       f"Create an account with this email address to see it: {url_for('register', _external=True)}")
        sent = send_email(
            target, f"{owner.name} shared \"{item_title(item_type, obj)}\" with you on LabLogbook",
            f"{owner.name} shared a {SHAREABLE_TYPES[item_type][1].lower()} with you.\n\n"
            + (f"Their note: {share.note}\n\n" if share.note else '') + access_hint + "\n",
        )
        session['collab_msg'] = (f'Shared with {target}.' + (' They were emailed.' if sent else '')
                                 + ('' if registered else ' They have no account yet — they will see it after registering with that email, or you can make a guest link instead.'))

    db.session.add(share)
    db.session.commit()
    return redirect(url_for('collaborate'))


@app.route('/collaborate/share/<int:share_id>/revoke', methods=['POST'])
def revoke_share(share_id):
    share = get_owned_or_404(Share, share_id, owner_field='owner_id')
    db.session.delete(share)
    db.session.commit()
    session['collab_msg'] = 'Access removed.'
    return redirect(url_for('collaborate'))


@app.route('/collaborate/item/<item_type>/<int:item_id>')
def collab_item_view(item_type, item_id):
    obj = _load_item_or_404(item_type, item_id)
    access = viewer_access(item_type, item_id, obj.user_id)
    if not access:
        abort(404)
    owner = db.session.get(User, obj.user_id)
    comments = ShareComment.query.filter_by(item_type=item_type, item_id=item_id).order_by(ShareComment.created_at).all()
    authors = {u.id: u.name for u in User.query.filter(User.id.in_({c.user_id for c in comments})).all()} if comments else {}
    return render_template(
        'shared_view.html', payload=item_payload(item_type, obj), item_type=item_type, item_id=item_id,
        kind_label=SHAREABLE_TYPES[item_type][1], owner_name=owner.name if owner else 'Someone',
        is_owner=(access == 'owner'), guest=False, comments=comments, authors=authors,
    )


@app.route('/collaborate/item/<item_type>/<int:item_id>/comment', methods=['POST'])
def add_share_comment(item_type, item_id):
    obj = _load_item_or_404(item_type, item_id)
    if not viewer_access(item_type, item_id, obj.user_id):
        abort(404)
    body = request.form.get('body', '').strip()
    if body:
        db.session.add(ShareComment(item_type=item_type, item_id=item_id, user_id=session['user_id'], body=body[:2000]))
        db.session.commit()
    return redirect(url_for('collab_item_view', item_type=item_type, item_id=item_id) + '#comments')


@app.route('/collaborate/comment/<int:comment_id>/delete', methods=['POST'])
def delete_share_comment(comment_id):
    comment = ShareComment.query.get_or_404(comment_id)
    obj = _load_item_or_404(comment.item_type, comment.item_id)
    if session['user_id'] not in (comment.user_id, obj.user_id):
        abort(404)
    db.session.delete(comment)
    db.session.commit()
    return redirect(url_for('collab_item_view', item_type=comment.item_type, item_id=comment.item_id) + '#comments')


@app.route('/collaborate/item/protocol/<int:item_id>/copy', methods=['POST'])
def copy_shared_protocol(item_id):
    source = _load_item_or_404('protocol', item_id)
    if not viewer_access('protocol', item_id, source.user_id):
        abort(404)
    latest = source.latest_version
    clone = Protocol(
        user_id=session['user_id'], title=source.title, equipment=source.equipment,
        category=source.category, current_version=1, cloned_from_id=source.id,
    )
    db.session.add(clone)
    db.session.flush()
    db.session.add(ProtocolVersion(
        protocol_id=clone.id, version_number=1, steps=(latest.steps if latest else ''),
        change_note=f'Copied from a shared protocol (v{latest.version_number if latest else 1})',
    ))
    db.session.commit()
    return redirect(url_for('elements', tab='protocols'))


@app.route('/shared/<token>')
def shared_link_view(token):
    """Guest access: read-only, no account, no comments — just whoever holds the unguessable link."""
    share = Share.query.filter_by(link_token=token).first_or_404()
    if share.expires_at and share.expires_at <= datetime.now():
        return render_template('error.html', title='Link expired', message='This shared link has expired. Ask the person who sent it for a new one.'), 410
    obj = SHAREABLE_TYPES[share.item_type][0].query.get(share.item_id)
    if not obj:
        abort(404)
    owner = db.session.get(User, share.owner_id)
    response = app.make_response(render_template(
        'shared_view.html', payload=item_payload(share.item_type, obj), item_type=share.item_type, item_id=share.item_id,
        kind_label=SHAREABLE_TYPES[share.item_type][1], owner_name=owner.name if owner else 'Someone',
        is_owner=False, guest=True, comments=[], authors={},
    ))
    response.headers['X-Robots-Tag'] = 'noindex, nofollow'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@app.route('/collaborate/groups/new', methods=['POST'])
def create_group():
    name = request.form.get('name', '').strip()
    if not name:
        session['collab_error'] = 'A group needs a name.'
    else:
        db.session.add(LabGroup(owner_id=session['user_id'], name=name[:150],
                                description=(request.form.get('description', '').strip()[:300] or None)))
        db.session.commit()
        session['collab_msg'] = f'Group "{name}" created — add members below.'
    return redirect(url_for('collaborate'))


@app.route('/collaborate/groups/<int:group_id>/members/add', methods=['POST'])
def add_group_member(group_id):
    group = get_owned_or_404(LabGroup, group_id, owner_field='owner_id')
    emails = [e.strip().lower() for e in re.split(r'[,\s;]+', request.form.get('emails', '')) if e.strip()]
    added = 0
    for e in emails:
        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', e) or e == current_user_email():
            continue
        if not GroupMember.query.filter_by(group_id=group.id, email=e).first():
            db.session.add(GroupMember(group_id=group.id, email=e))
            added += 1
    db.session.commit()
    session['collab_msg'] = f'Added {added} member(s) to "{group.name}".' if added else 'No new valid emails to add.'
    return redirect(url_for('collaborate'))


@app.route('/collaborate/groups/<int:group_id>/members/<int:member_id>/remove', methods=['POST'])
def remove_group_member(group_id, member_id):
    group = get_owned_or_404(LabGroup, group_id, owner_field='owner_id')
    member = GroupMember.query.get_or_404(member_id)
    if member.group_id != group.id:
        abort(404)
    db.session.delete(member)
    db.session.commit()
    return redirect(url_for('collaborate'))


@app.route('/collaborate/groups/<int:group_id>/delete', methods=['POST'])
def delete_group(group_id):
    group = get_owned_or_404(LabGroup, group_id, owner_field='owner_id')
    Share.query.filter_by(group_id=group.id).delete()
    db.session.delete(group)
    db.session.commit()
    session['collab_msg'] = 'Group deleted, along with anything shared through it.'
    return redirect(url_for('collaborate'))


if __name__ == '__main__':
    # With the dev reloader, this block runs in both the watcher and the serving child;
    # only one of them should touch the schema.
    if PRODUCTION_MODE or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        ensure_schema()
    port = int(os.environ.get('PORT', 5000))
    if PRODUCTION_MODE:
        # Flask's own server (below) is explicitly unfit for production, even with
        # debug off — no concurrency, no hardening against slow/malformed clients.
        # waitress is a real WSGI server and works the same on Windows and Linux.
        from waitress import serve
        app.logger.info(f'Serving with waitress on 0.0.0.0:{port}')
        serve(app, host='0.0.0.0', port=port, threads=int(os.environ.get('WAITRESS_THREADS', 8)))
    else:
        # use_reloader is kept on regardless — it's just a local file-watcher and,
        # unlike debug, carries no risk of exposing the interactive debugger.
        app.run(debug=DEBUG_MODE, port=port, use_reloader=True)
