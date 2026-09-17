from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, abort
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import io
import re
import time
import secrets
import json
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
from scipy.signal import find_peaks, savgol_filter
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.restoration import unwrap_phase
from afm_formats import try_parse_afm_native, NATIVE_EXTENSIONS as AFM_NATIVE_EXTENSIONS, UNPARSED_EXTENSIONS as AFM_UNPARSED_EXTENSIONS

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

app = Flask(__name__)
app.secret_key = _load_secret_key()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
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
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

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


class CustomPlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    title = db.Column(db.String(200), nullable=True)
    plot_type = db.Column(db.String(30), nullable=False)   # scatter/line/bar/histogram/box
    config_json = db.Column(db.Text, nullable=False)       # series definitions + customization options
    analysis_json = db.Column(db.Text, nullable=True)      # per-series stats/analysis text
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


with app.app_context():
    db.create_all()

    # lightweight additive migration: db.create_all() only creates missing tables,
    # it won't add new columns to an existing sqlite file, so add missing ones by hand
    def _add_missing_columns(table, columns):
        existing = {row[1] for row in db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()}
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


# Every route requires a signed-in session except these — the pages you need
# before you can have one, plus Flask's own static file server and the favicon
# every browser requests automatically regardless of what page is loaded.
PUBLIC_ENDPOINTS = {'index', 'register', 'login', 'static', 'favicon'}


@app.before_request
def require_login():
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


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='images/lablogbook-logo1.svg'))


@app.route('/')
def index():
    return render_template('index.html', banner_image='images/homepage-banner.png')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="An account with this email already exists.")

        if len(password) < 8:
            return render_template('register.html', error="Password must be at least 8 characters.")

        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
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
    parts.append(f"Cyclic voltammetry analysis across {n} scan(s):")

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

    def add_result(label, y_values, note=None):
        stats = compute_series_stats(y_values)
        analysis = build_stats_analysis(label, stats, plot_type)
        if note:
            analysis = note + " " + analysis
        results.append({'label': label, 'stats': stats, 'analysis': analysis})

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

            add_result(s['label'], s['y'])

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
    ("Mass & Separation", ["Mass Spec (LC-MS, MALDI)", "HPLC / GC", "GPC / SEC"]),
    ("Microscopy & Imaging", ["SEM", "AFM", "TEM", "Confocal / Fluorescence"]),
    ("Crystallography & Surface", ["XPS", "XRD", "BET Nitrogen Sorption"]),
    ("Biophysics & Kinetics", ["SPR / BLI", "ITC", "DLS"]),
    ("Cellular & Phenotypic", ["FACS / Flow Cytometry", "scRNA-seq / Omics"]),
    ("Mechanics & Electrochemistry", ["UTM / Nanoindentation", "Rheometer", "Cyclic Voltammetry (CV)"]),
]

# Tabs shown per technique — tailored to what's actually relevant for that measurement type.
# Techniques not listed explicitly fall back to DEFAULT_TECHNIQUE_TABS.
DEFAULT_TECHNIQUE_TABS = ["Select files", "Plot", "Format", "Analysis"]

TECHNIQUE_TABS = {
    "FTIR": ["Select files", "Plot Spectrum", "Peak Picking", "Baseline Correction", "Format", "Analysis"],
    "UV-Vis": ["Select files", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "Fluorescence": ["Select files", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "Raman": ["Select files", "Plot Spectrum", "Peak Picking", "Baseline Correction", "Format", "Analysis"],
    "NMR (1H, 13C)": ["Select files", "Plot Spectrum", "Peak Picking", "Integration", "Format", "Analysis"],
    "CD (Circular Dichroism)": ["Select files", "Plot Spectrum", "Format", "Analysis"],

    "Mass Spec (LC-MS, MALDI)": ["Select files", "Plot Spectrum", "Peak Picking", "Format", "Analysis"],
    "HPLC / GC": ["Select files", "Plot Chromatogram", "Peak Integration", "Format", "Analysis"],
    "GPC / SEC": ["Select files", "Plot Chromatogram", "Molecular Weight", "Format", "Analysis"],

    "SEM": ["Select images", "Measure Particles", "Porosity", "Roughness", "Analysis"],
    "AFM": ["Select data", "Topography", "Mechanical", "Electrical", "Magnetic", "Chemical / Frictional", "Biological", "Analysis"],
    "TEM": ["Select images", "Measure Particles", "Layer Thickness", "Defects", "SAED", "Lattice Fringes", "Strain Mapping", "Analysis"],
    "Confocal / Fluorescence": ["Select images", "Measure Particles", "Analysis"],

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
}


def slugify_technique(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


TECHNIQUE_SLUGS = {slugify_technique(t): t for cat, techs in TECHNIQUE_CATEGORIES for t in techs}

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
}


def tech_state_key(slug):
    return f'tech_state_{slug}'


def tech_get_state(slug):
    key = tech_state_key(slug)
    state = session.get(key)
    if not state:
        modes = TECHNIQUE_PLOT_MODES.get(slug, [("line_spectrum", "Line Graph")])
        state = {
            'file_ids': [],
            'plot_type': modes[0][0],
            'derivative': False,
            'format': {
                'legend': True, 'legend_loc': 'best', 'legend_orientation': 'vertical', 'legend_scale': 1.0,
                'line_width': 1.6, 'marker_size': 18, 'tick_width': 1.0, 'label_size': 11,
                'bold_labels': False, 'grid': True, 'log_x': False, 'log_y': False, 'colormap': 'default',
                'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None,
            },
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


def assign_peaks(peaks, table):
    """Matches each (x, y) peak against a reference range table, returns list of (x, y, description)."""
    assigned = []
    for x, y in peaks:
        match = next((desc for lo, hi, desc in table if lo <= x <= hi), None)
        assigned.append((x, y, match))
    return assigned


def generate_spectroscopy_analysis(technique_name, series_list):
    """Produces a genuine technique-specific interpretation: peak assignments for
    vibrational/NMR/CD techniques, or computed quantities (λmax, band gap, Stokes shift)
    for UV-Vis/Fluorescence. Falls back to basic stats if the technique isn't specially handled."""
    if not series_list:
        return []

    results = []

    if technique_name == 'FTIR':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.1 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, FTIR_TABLE)
            lines = [f"{x:.0f} cm⁻¹ → {desc}" if desc else f"{x:.0f} cm⁻¹ → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} peak(s) detected. " + ("; ".join(lines) + "." if lines else "No significant peaks found.")
            results.append({'label': s['label'], 'analysis': text})

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
            results.append({'label': s['label'], 'analysis': text})

    elif technique_name == 'NMR (1H, 13C)':
        for s in series_list:
            idx, _ = find_peaks(s['y'], prominence=(max(s['y']) - min(s['y'])) * 0.1 or None)
            peaks = [(float(s['x'][i]), float(s['y'][i])) for i in idx]
            assigned = assign_peaks(peaks, NMR_1H_TABLE)
            lines = [f"{x:.2f} ppm → {desc}" if desc else f"{x:.2f} ppm → unassigned" for x, y, desc in assigned]
            text = f"{len(peaks)} peak(s) detected (assuming 1H shifts). " + ("; ".join(lines) + "." if lines else "No significant peaks found.")
            results.append({'label': s['label'], 'analysis': text})

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
            results.append({'label': s['label'], 'analysis': text})

    elif technique_name == 'UV-Vis':
        for s in series_list:
            peak_idx = int(np.argmax(s['y']))
            lam_max = float(s['x'][peak_idx])
            peak_intensity = float(s['y'][peak_idx])
            text = f"λmax ≈ {lam_max:.1f} nm (absorbance {peak_intensity:.3g})."
            if 190 <= lam_max <= 1100:
                gap_ev = 1240 / lam_max
                text += f" Approximate optical transition energy ≈ {gap_ev:.2f} eV (E = 1240/λmax — a rough estimate, not a substitute for Tauc analysis)."
            results.append({'label': s['label'], 'analysis': text})

    elif technique_name == 'Fluorescence':
        peak_positions = []
        for s in series_list:
            peak_idx = int(np.argmax(s['y']))
            lam_em = float(s['x'][peak_idx])
            peak_positions.append((s['label'], lam_em, float(s['y'][peak_idx])))
            results.append({'label': s['label'], 'analysis': f"Emission maximum ≈ {lam_em:.1f} nm (intensity {s['y'][peak_idx]:.3g})."})

        if len(peak_positions) >= 2:
            lam_values = [p[1] for p in peak_positions]
            shift = max(lam_values) - min(lam_values)
            if shift > 2:
                direction = "red-shifted" if peak_positions[-1][1] > peak_positions[0][1] else "blue-shifted"
                results.append({'label': 'Overall', 'analysis': f"Emission maxima span {shift:.1f} nm across samples — later samples appear {direction} relative to the first, which can indicate changes in the local environment, conjugation, or aggregation state."})

    return results



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



def tech_render_spectrum_plot(state, mark_peaks=False):
    """Renders a technique's plot — line spectrum, calibration scatter+fit, or an honest
    placeholder for 2D contour data (not supported by our flat X/Y column model)."""
    plot_type = state.get('plot_type', 'line_spectrum')

    if plot_type == 'contour_2d':
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.text(0.5, 0.5, "2D contour/heatmap data (e.g. COSY/HSQC) needs a full 2D intensity\nmatrix, not flat X/Y columns — not supported by this tool yet.",
                ha='center', va='center', fontsize=11, color='#888', transform=ax.transAxes, wrap=True)
        ax.axis('off')
        fig.tight_layout()
        plot_filename = f"tech_{int(datetime.now().timestamp())}.png"
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
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

    fig, ax = plt.subplots(figsize=(8, 5.5))
    results = []
    peaks_by_label = {}

    if plot_type == 'calibration_scatter':
        for i, s in enumerate(series_list):
            ax.scatter(s['x'], s['y'], s=fmt['marker_size'], color=colors[i], alpha=0.8, label=s['label'])
            try:
                popt, _ = curve_fit(linear_fn, s['x'], s['y'])
                x_smooth = np.linspace(min(s['x']), max(s['x']), 200)
                ax.plot(x_smooth, linear_fn(x_smooth, *popt), color=colors[i], linestyle='--', linewidth=fmt['line_width'])
                y_pred = linear_fn(s['x'], *popt)
                ss_res = np.sum((s['y'] - y_pred) ** 2)
                ss_tot = np.sum((s['y'] - np.mean(s['y'])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
                analysis = f"Linear fit: y = {popt[0]:.4g}x + {popt[1]:.4g}, R² = {r_squared:.4f}." if r_squared is not None else "Linear fit could not be scored."
            except Exception:
                analysis = "Linear fit did not converge for this series."
            stats = compute_series_stats(s['y'])
            results.append({'label': s['label'], 'stats': stats, 'analysis': analysis})
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
            ax.plot(s['x'], s['y'], color=colors[i], linewidth=fmt['line_width'], label=s['label'])
            if mark_peaks and peaks_by_label.get(s['label']):
                px = [p[0] for p in peaks_by_label[s['label']]]
                py = [p[1] for p in peaks_by_label[s['label']]]
                ax.scatter(px, py, color=colors[i], marker='v', s=60, edgecolor='black', zorder=5)
            stats = compute_series_stats(s['y'])
            n_peaks = len(peaks_by_label.get(s['label'], []))
            note = f"{n_peaks} peak(s) detected." if mark_peaks else None
            analysis = build_stats_analysis(s['label'], stats, plot_type)
            if note:
                analysis = note + " " + analysis
            results.append({'label': s['label'], 'stats': stats, 'analysis': analysis})

    label_weight = 'bold' if fmt.get('bold_labels') else 'normal'
    ax.set_xlabel('X', fontsize=fmt['label_size'], fontweight=label_weight)
    ax.set_ylabel('Y', fontsize=fmt['label_size'], fontweight=label_weight)
    ax.tick_params(width=fmt['tick_width'])
    if fmt.get('log_x'):
        ax.set_xscale('log')
    if fmt.get('log_y'):
        ax.set_yscale('log')

    # explicit axis range (zoom), applied after scale type is set
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
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Emission wavelength' if state.get('wide_mode') else 'Series value')
    elif fmt.get('legend', True):
        ncol = min(len(series_list), 3) if fmt.get('legend_orientation') == 'horizontal' and len(series_list) > 3 else (len(series_list) if fmt.get('legend_orientation') == 'horizontal' else 1)
        ax.legend(fontsize=9 * fmt.get('legend_scale', 1.0), loc=fmt.get('legend_loc', 'best'), ncol=ncol, framealpha=0.9)

    fig.tight_layout()
    plot_base = f"tech_{int(datetime.now().timestamp())}"
    plot_filename = f"{plot_base}.png"
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename), dpi=130)
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{plot_base}.svg"))
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{plot_base}.pdf"))
    plt.close(fig)

    return plot_filename, results, errors, peaks_by_label



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


@app.route('/characterizations/data/technique/<slug>')
def technique_workspace(slug):
    technique_name = TECHNIQUE_SLUGS.get(slug)
    if not technique_name:
        return redirect(url_for('data_interpretation'))

    tabs = TECHNIQUE_TABS.get(technique_name, DEFAULT_TECHNIQUE_TABS)
    active_tab = request.args.get('tab', tabs[0])
    if active_tab not in tabs:
        active_tab = tabs[0]

    parent_category = next((cat for cat, techs in TECHNIQUE_CATEGORIES if technique_name in techs), None)

    # Spectroscopy techniques get the fully wired workflow; others still show the placeholder for now.
    if parent_category == 'Spectroscopy':
        state = tech_get_state(slug)
        all_files = DataFile.query.filter_by(file_type='tabular', technique_name=technique_name, user_id=session['user_id']).order_by(DataFile.uploaded_at.desc()).all()
        selected_files = [f for f in all_files if f.id in state['file_ids']]
        plot_modes = TECHNIQUE_PLOT_MODES.get(slug, [('line_spectrum', 'Line Graph')])

        plot_filename, results, plot_errors, peaks_by_label = (None, [], [], {})
        if active_tab in ('Plot Spectrum', 'Peak Picking', 'Format', 'Analysis') and state['file_ids']:
            mark_peaks = (active_tab in ('Peak Picking', 'Analysis'))
            plot_filename, results, plot_errors, peaks_by_label = tech_render_spectrum_plot(state, mark_peaks=mark_peaks)

        spectroscopy_analysis = []
        if active_tab == 'Analysis' and state['file_ids'] and state.get('plot_type') != 'contour_2d':
            if state.get('wide_mode'):
                analysis_series, _, _ = tech_build_wide_series(state)
            else:
                analysis_series, _ = dp_build_series(state)
            spectroscopy_analysis = generate_spectroscopy_analysis(technique_name, analysis_series)

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
            plot_filename=plot_filename,
            results=results,
            plot_errors=plot_errors,
            peaks_by_label=peaks_by_label,
            spectroscopy_analysis=spectroscopy_analysis,
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

        defect_results = []
        if active_tab == 'Defects' and selected_images:
            image_ids = [f.id for f in selected_images]
            annotations = DefectAnnotation.query.filter(DefectAnnotation.file_id.in_(image_ids)).order_by(DefectAnnotation.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in annotations:
                defect_results.append({'annotation': a, 'file': files_by_id.get(a.file_id)})

        tem_analyses = []
        if active_tab == 'Analysis' and selected_images:
            image_ids = [f.id for f in selected_images]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(image_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in analyses:
                extra_images = json.loads(a.extra_images_json) if a.extra_images_json else []
                tem_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id), 'extra_images': extra_images})

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
            defect_types=DEFECT_TYPES,
            defect_results=defect_results,
            tem_analyses=tem_analyses,
            tem_error=session.pop('tem_error', None),
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
        if active_tab == 'Analysis' and selected_images:
            image_ids = [f.id for f in selected_images]
            analyses = ImageAnalysis.query.filter(ImageAnalysis.file_id.in_(image_ids)).order_by(ImageAnalysis.created_at.desc()).all()
            files_by_id = {f.id: f for f in selected_images}
            for a in analyses:
                particle_analyses.append({'analysis': a, 'file': files_by_id.get(a.file_id)})

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


@app.route('/characterizations/data/technique/<slug>/select-files', methods=['POST'])
def tech_select_files(slug):
    state = tech_get_state(slug)
    selected = request.form.getlist('file_ids')
    new_file_ids = [int(i) for i in selected if i.isdigit()]

    if new_file_ids != state['file_ids']:
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            state['format'][key] = None

    state['file_ids'] = new_file_ids
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Plot Spectrum'))


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
    return redirect(url_for('technique_workspace', slug=slug, tab='Plot Spectrum'))


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
    state = tech_get_state(slug)
    fmt = state['format']
    fmt['legend'] = request.form.get('legend') == 'on'
    fmt['legend_loc'] = request.form.get('legend_loc', 'best')
    fmt['legend_orientation'] = request.form.get('legend_orientation', 'vertical')
    fmt['legend_scale'] = float(request.form.get('legend_scale', 1.0) or 1.0)
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
    tech_save_state(slug, state)
    return redirect(url_for('technique_workspace', slug=slug, tab='Format'))


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

    return render_template(
        'data_interpretation_workspace.html',
        page_title='Data Interpretation',
        tab=tab,
        state=state,
        all_files=all_files,
        image_files=image_files,
        selected_files=selected_files,
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


def build_particle_size_analysis(sizes, unit):
    n = len(sizes)
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))
    cv = std_size / mean_size if mean_size else 0
    parts = [f"Manual sizing of {n} particle{'s' if n != 1 else ''}: mean diameter = {mean_size:.1f} {unit} (± {std_size:.1f} {unit} std dev), range {min(sizes):.1f}–{max(sizes):.1f} {unit}."]
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
    if redirect_slug and redirect_slug in TECHNIQUE_SLUGS:
        if technique_name == 'AFM':
            select_tab = 'Select data'
        else:
            select_tab = 'Select images' if file_type == 'image' else 'Select files'
        return redirect(url_for('technique_workspace', slug=redirect_slug, tab=select_tab))

    return redirect(url_for('data_interpretation_workspace'))


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

    return redirect(url_for('data_interpretation_workspace'))


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


if __name__ == '__main__':
    if os.environ.get('PRODUCTION', '0').lower() in ('1', 'true', 'yes'):
        # Flask's own server (below) is explicitly unfit for production, even with
        # debug off — no concurrency, no hardening against slow/malformed clients.
        # waitress is a real WSGI server and works the same on Windows and Linux.
        from waitress import serve
        port = int(os.environ.get('PORT', 5000))
        app.logger.info(f'Serving with waitress on 0.0.0.0:{port}')
        serve(app, host='0.0.0.0', port=port)
    else:
        # use_reloader is kept on regardless — it's just a local file-watcher and,
        # unlike debug, carries no risk of exposing the interactive debugger.
        app.run(debug=DEBUG_MODE, use_reloader=True)
