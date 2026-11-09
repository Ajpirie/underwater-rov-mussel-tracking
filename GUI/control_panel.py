"""Desktop supervisory panel for recorded and live mussel tracking.

Launch from repository root with: python -m GUI
"""
from __future__ import annotations

from collections import deque

import json
import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

import numpy as np
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from Integration.tracking_session import (ROOT, Frame, Snapshot, TrackingSession, discover_datasets,
                               load_csv_frames, recommended_dt)
from Integration.live_input import live_source
from Integration.observation_session import ObservationSession
from Integration.session_export import export_session
from Integration.rov_state_machine import ROVController, ROVState
from Integration.rov_simulator import SimulatedROV

CONFIG_PATH = Path.home() / '.mussel_prediction' / 'control_panel.json'
BG = '#F4F7FB'
WHITE = '#FFFFFF'
INK = '#17283D'
MUTED = '#62758A'
BLUE = '#2563EB'
TEAL = '#0F927F'
AMBER = '#BA7417'
RED = '#BE3944'
GRID = '#E2E8F0'
PALETTE = colormaps['tab10'].colors

DEFAULTS = {
    'mode': 'Test', 'dataset': 'Data/Real/RD_004.csv',
    'grouping': 'Auto', 'frame_dt': '0.5',
    'horizon': '5', 'stale': '3', 'max_id': '999', 'speed': '1×',
    'lull_threshold': '0.015', 'lull_window': '2', 'calm_min': '1', 'target_z': '0.35',
    'xy_tol': '0.05', 'z_tol': '0.05',
    'selected_track': 'Cluster centroid',
    'show_raw': True, 'show_estimate': True, 'show_forecast': True,
}


def load_settings():
    try:
        if CONFIG_PATH.exists():
            value = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            settings = {**DEFAULTS, **(value if isinstance(value, dict) else {})}
            if settings.get('stale_units') != 'seconds':
                # Convert previously saved frame counts into seconds.
                try:
                    settings['stale'] = f"{float(settings['stale']) * float(settings['frame_dt']):g}"
                except (ValueError, TypeError):
                    settings['stale'] = DEFAULTS['stale']
            settings['stale_units'] = 'seconds'
            return settings
    except (OSError, ValueError):
        pass
    return DEFAULTS.copy()


def save_settings(settings):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    except OSError:
        pass  # Failure to persist preferences must not stop tracking.


def pos_number(value: str, name: str, minimum=0.0, maximum=None):
    try:
        number = float(value)
        if not math.isfinite(number) or number <= minimum or (maximum is not None and number > maximum):
            raise ValueError()
        return number
    except ValueError as exc:
        raise ValueError(f'{name} must be greater than {minimum:g}' +
                         (f' and no more than {maximum:g}' if maximum is not None else '')) from exc


def format_mm(value):
    return '—' if value is None else f'{1000 * value:,.1f} mm'


def theme_axes(ax):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in getattr(ax, 'spines', {}).values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=.7, linewidth=.7)


class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Mussel Motion · Control Panel')
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f'{min(1390, max(980, screen_w - 90))}x{min(880, max(660, screen_h - 100))}')
        self.minsize(min(960, screen_w), min(620, screen_h))
        self.configure(background=BG)
        self._font = next((name for name in ('Segoe UI', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans')
                          if name in tkfont.families()), 'TkDefaultFont')
        self.settings = load_settings()
        self.vars = {name: tk.StringVar(value=str(self.settings.get(name, default)))
                     for name, default in DEFAULTS.items() if not isinstance(default, bool)}
        self.show_raw = tk.BooleanVar(value=bool(self.settings['show_raw']))
        self.show_estimate = tk.BooleanVar(value=bool(self.settings['show_estimate']))
        self.show_forecast = tk.BooleanVar(value=bool(self.settings['show_forecast']))
        self.session = None
        self.frames = []
        self.cursor = -1
        self.playing = False
        self.live_observations = []
        self.output_saved = False
        self.source_label = ""
        self.last_rejected_count = 0
        self.timer_id = None
        self.seek_id = None
        self.seek_target = None
        self.last_packet = None
        self.lock_ids = None
        self.last_display_draw = 0
        self._ignore_slider = False
        self._dataset_keys = {}
        self.rov_controller = ROVController(SimulatedROV())
        self._rov_logged = 0
        self._build_theme()
        self._build_layout()
        self.bind('<Configure>', self._responsive_layout, add='+')
        self._refresh_datasets()
        self._set_mode()
        self._reset_dashboard()
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.after(80, self._poll_live)

    def _build_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure('.', font=(self._font, 10), background=BG, foreground=INK)
        style.configure('TFrame', background=BG)
        style.configure('Surface.TFrame', background=WHITE)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('Surface.TLabel', background=WHITE, foreground=INK)
        style.configure('Hint.TLabel', background=BG, foreground=MUTED, font=(self._font, 9))
        style.configure('Sidebar.TLabel', background=WHITE, foreground=INK)
        style.configure('SidebarHint.TLabel', background=WHITE, foreground=MUTED, font=(self._font, 9))
        style.configure('Heading.TLabel', background=WHITE, foreground=INK, font=(self._font, 11, 'bold'))
        style.configure('MetricTitle.TLabel', background=WHITE, foreground=MUTED, font=(self._font, 9))
        style.configure('MetricValue.TLabel', background=WHITE, foreground=INK, font=(self._font, 15, 'bold'))
        style.configure('TButton', padding=(12, 8), background=WHITE, foreground=INK)
        style.map('TButton', background=[('active', '#E7EEF9'), ('disabled', '#EFF3F7')])
        style.configure('Accent.TButton', padding=(12, 9), background=BLUE, foreground=WHITE,
                        font=(self._font, 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#1D4ED8'), ('disabled', '#AFC3E7')],
                  foreground=[('disabled', WHITE)])
        style.configure('TEntry', padding=5, fieldbackground=WHITE)
        style.configure('TCombobox', padding=4, fieldbackground=WHITE)
        style.configure('TNotebook', background=BG, borderwidth=0, tabmargins=(0, 6, 0, 0))
        style.configure('TNotebook.Tab', padding=(17, 10), background='#E8EEF6', foreground=MUTED)
        style.map('TNotebook.Tab', background=[('selected', WHITE)], foreground=[('selected', BLUE)])
        style.configure('TCheckbutton', background=WHITE)
        style.configure('TRadiobutton', background=WHITE)
        style.configure('TScale', background=BG)

    def _build_layout(self):
        header = ttk.Frame(self, padding=(22, 14, 20, 12))
        header.pack(fill='x')
        self.header_title = ttk.Label(header, text='MUSSEL MOTION', font=(self._font, 18, 'bold'))
        self.header_title.pack(side='left')
        self.header_hint = ttk.Label(header, text='  /  Tracking & prediction', style='Hint.TLabel')
        self.header_hint.pack(side='left', pady=4)
        self.live_status = tk.Label(header, text='●  IDLE', bg=BG, fg=MUTED,
                                    font=(self._font, 10, 'bold'))
        self.live_status.pack(side='right', padx=(12, 0))
        # Playback controls stay visible even when the settings sidebar is scrolled.
        global_controls = ttk.Frame(header)
        global_controls.pack(side='right')
        self.start_button = ttk.Button(global_controls, text='Start', style='Accent.TButton', command=self._start)
        self.start_button.pack(side='left', padx=3)
        self.pause_button = ttk.Button(global_controls, text='Pause', command=self._pause_resume)
        self.pause_button.pack(side='left', padx=3)
        self.step_button = ttk.Button(global_controls, text='Step ›', command=self._step)
        self.step_button.pack(side='left', padx=3)
        self.stop_button = ttk.Button(global_controls, text='Stop', command=self._stop)
        self.stop_button.pack(side='left', padx=3)
        ttk.Separator(self).pack(fill='x')

        content = ttk.Frame(self, padding=(14, 10, 14, 12))
        content.pack(fill='both', expand=True)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        sidebar_wrap = ttk.Frame(content, style='Surface.TFrame', width=314)
        sidebar_wrap.grid(row=0, column=0, sticky='ns', padx=(0, 12))
        sidebar_wrap.grid_propagate(False)
        self.sidebar_canvas = tk.Canvas(sidebar_wrap, background=WHITE, highlightthickness=0, width=300)
        scroll = ttk.Scrollbar(sidebar_wrap, orient='vertical', command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.sidebar_canvas.pack(side='left', fill='both', expand=True)
        side = ttk.Frame(self.sidebar_canvas, style='Surface.TFrame', padding=(16, 14, 15, 12))
        sid = self.sidebar_canvas.create_window(0, 0, window=side, anchor='nw')
        side.bind('<Configure>', lambda _: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox('all')))
        self.sidebar_canvas.bind('<Configure>', lambda e: self.sidebar_canvas.itemconfig(sid, width=e.width))
        self.sidebar_canvas.bind('<Enter>', lambda _: self.sidebar_canvas.bind_all('<MouseWheel>', self._mouse_scroll))
        self.sidebar_canvas.bind('<Leave>', lambda _: self.sidebar_canvas.unbind_all('<MouseWheel>'))

        self._section(side, '01  MODE')
        mode = ttk.Frame(side, style='Surface.TFrame')
        mode.pack(fill='x', pady=(0, 8))
        for value, label in [('Test', 'Test · replay'), ('Run', 'Run · live')]:
            ttk.Radiobutton(mode, text=label, value=value, variable=self.vars['mode'],
                            command=self._set_mode).pack(side='left', padx=(0, 10))
        self.source_frame = ttk.Frame(side, style='Surface.TFrame')
        self.source_frame.pack(fill='x')
        self.test_source = ttk.Frame(self.source_frame, style='Surface.TFrame')
        self.live_source = ttk.Frame(self.source_frame, style='Surface.TFrame')
        self._label(self.test_source, 'Recorded dataset')
        self.dataset_combo = ttk.Combobox(self.test_source, state='readonly', width=30)
        self.dataset_combo.pack(fill='x', pady=(2, 5))
        self.dataset_combo.bind('<<ComboboxSelected>>', self._dataset_changed)
        ttk.Button(self.test_source, text='Open CSV…', command=self._browse_csv).pack(fill='x')
        self._label(self.test_source, 'Frame grouping')
        ttk.Combobox(self.test_source, state='readonly', textvariable=self.vars['grouping'],
                     values=['Auto', 'Rows', 'Repeated IDs'], width=26).pack(fill='x', pady=(2, 4))
        ttk.Label(self.test_source, text='RD: 2 fps  ·  260721: 15 fps + frame log',
                  style='SidebarHint.TLabel', wraplength=265).pack(anchor='w')

        ttk.Label(self.live_source, text='Individual (ID, X, Y, Z) observations arrive through Integration.live_input.\nNo frame markers required. ROV sequence is simulation only.',
                  style='SidebarHint.TLabel', wraplength=265).pack(anchor='w', pady=6)

        self._section(side, '02  PREDICTION')
        ttk.Label(side, text='UKF · active prediction method',
                  style='SidebarHint.TLabel').pack(anchor='w', pady=(2, 7))
        self._field_pair(side, 'Nominal frame period (s)', 'frame_dt', 'Horizon (s)', 'horizon')
        self._field_pair(side, 'ID timeout (s)', 'stale', 'Max ID', 'max_id')

        self._section(side, '03  ALIGNMENT · ADVISORY')
        self._field_pair(side, 'Target depth Z (m)', 'target_z', 'XY tolerance (m)', 'xy_tol')
        self._field_pair(side, 'Z tolerance (m)', 'z_tol', 'Calm speed (m/s)', 'lull_threshold')
        self._field_pair(side, 'Lull window (s)', 'lull_window', 'Min calm time (s)', 'calm_min')
        ttk.Label(side, text='Target is X = 0, Y = 0, and your chosen Z. Calm threshold is illustrative; calibrate for sea state.',
                  style='SidebarHint.TLabel', wraplength=265).pack(anchor='w', pady=(6, 4))
        ttk.Button(side, text='Apply alignment settings',
                   command=self._apply_alignment_settings).pack(fill='x', pady=(3, 0))

        self._section(side, '04  DISPLAY & EXPORT')
        test_controls = ttk.Frame(side, style='Surface.TFrame')
        test_controls.pack(fill='x', pady=(0, 8))
        self.speed_frame = test_controls
        ttk.Label(test_controls, text='Replay speed', style='SidebarHint.TLabel').pack(side='left')
        self.speed_combo = ttk.Combobox(test_controls, textvariable=self.vars['speed'], state='readonly',
                                        values=['0.25×', '0.5×', '1×', '2×', '4×', '8×', '16×', 'Max'], width=8)
        self.speed_combo.pack(side='right')
        self.speed_combo.bind('<<ComboboxSelected>>', lambda _: self._reschedule())
        self.lock_button = ttk.Button(side, text='Lock current cluster', command=self._toggle_lock)
        self.lock_button.pack(fill='x', pady=(2, 3))
        self.alignment_var = tk.StringVar(value='No tracking data')
        tk.Label(side, textvariable=self.alignment_var, bg=WHITE, fg=MUTED,
                 justify='left', anchor='w', wraplength=260,
                 font=(self._font, 9)).pack(fill='x', pady=(3, 8))
        ttk.Button(side, text='Save session to Outputs/', command=self._export).pack(fill='x', pady=2)
        self.motion_var = tk.StringVar(value='Motion and outliers available after tracking begins.')
        tk.Label(side, textvariable=self.motion_var, bg=WHITE, fg=MUTED,
                 justify='left', anchor='w', wraplength=260,
                 font=(self._font, 9)).pack(fill='x', pady=(6, 5))
        self._section(side, '05  ROV · SIMULATION ONLY')
        ttk.Label(side, text='No ROV commands are sent. This tests operator approval, simulated feedback, timeouts and stop/reset only.',
                  style='SidebarHint.TLabel', wraplength=265).pack(anchor='w', pady=(3, 6))
        self.rov_status_var = tk.StringVar(value='IDLE · DISARMED')
        tk.Label(side, textvariable=self.rov_status_var, bg=WHITE, fg=MUTED,
                 justify='left', anchor='w', wraplength=260,
                 font=(self._font, 9, 'bold')).pack(fill='x', pady=(3, 6))
        self.rov_arm_button = ttk.Button(side, text='Arm simulation', command=self._arm_rov)
        self.rov_arm_button.pack(fill='x', pady=2)
        self.align_button = ttk.Button(side, text='Simulate approach', command=self._command_alignment)
        self.align_button.pack(fill='x', pady=2)
        self.collect_button = ttk.Button(side, text='Approve simulated collection', command=self._command_collection)
        self.collect_button.pack(fill='x', pady=2)
        self.rov_stop_button = ttk.Button(side, text='Safe stop (simulation)', command=self._safe_stop_rov)
        self.rov_stop_button.pack(fill='x', pady=2)
        self.rov_reset_button = ttk.Button(side, text='Reset simulated fault', command=self._reset_rov)
        self.rov_reset_button.pack(fill='x', pady=2)
        ttk.Button(side, text='Input format / interpretation', command=self._show_help).pack(fill='x', pady=2)

        main = ttk.Frame(content)
        main.grid(row=0, column=1, sticky='nsew')
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        metrics = ttk.Frame(main)
        metrics.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        for i in range(5):
            metrics.grid_columnconfigure(i, weight=1, uniform='metrics')
        self.metrics = {}
        for i, (key, title) in enumerate([('time', 'DATA TIME'), ('active', 'VISIBLE / IDS'),
                                         ('horizon', 'FORECAST'), ('rmse', '3D RMSE'),
                                         ('gain', 'VS. HOLD')]):
            card = ttk.Frame(metrics, style='Surface.TFrame', padding=(11, 8))
            card.grid(row=0, column=i, sticky='nsew', padx=(0 if i == 0 else 4, 0))
            ttk.Label(card, text=title, style='MetricTitle.TLabel').pack(anchor='w')
            v = tk.StringVar(value='—')
            self.metrics[key] = v
            ttk.Label(card, textvariable=v, style='MetricValue.TLabel').pack(anchor='w', pady=(3, 1))

        self.pages = ttk.Notebook(main)
        self.pages.grid(row=1, column=0, sticky='nsew')
        self.page_scene = ttk.Frame(self.pages, style='Surface.TFrame')
        self.page_tracks = ttk.Frame(self.pages, style='Surface.TFrame')
        self.page_accuracy = ttk.Frame(self.pages, style='Surface.TFrame')
        self.page_motion = ttk.Frame(self.pages, style='Surface.TFrame')
        self.page_events = ttk.Frame(self.pages, style='Surface.TFrame')
        for page, name in [(self.page_scene, 'Scene 3D'), (self.page_tracks, 'Position over time'),
                           (self.page_accuracy, 'Accuracy'), (self.page_motion, 'Motion'),
                           (self.page_events, 'Events')]:
            self.pages.add(page, text=name)
        self.pages.bind('<<NotebookTabChanged>>', lambda _: self._draw(force=True))
        self.scene_fig, self.scene_canvas = self._figure(self.page_scene)

        track_controls = ttk.Frame(self.page_tracks, style='Surface.TFrame', padding=(10, 8))
        track_controls.pack(fill='x')
        ttk.Label(track_controls, text='Track', style='Surface.TLabel').pack(side='left', padx=(0, 8))
        self.track_combo = ttk.Combobox(track_controls, textvariable=self.vars['selected_track'],
                                        state='readonly', width=19, values=['Cluster centroid'])
        self.track_combo.pack(side='left', padx=(0, 12))
        self.track_combo.bind('<<ComboboxSelected>>', lambda _: self._draw(force=True))
        for var, text in [(self.show_raw, 'Measured'), (self.show_estimate, 'Filtered'),
                          (self.show_forecast, 'Forecast')]:
            ttk.Checkbutton(track_controls, variable=var, text=text,
                            command=lambda: self._draw(force=True)).pack(side='left', padx=5)
        self.track_fig, self.track_canvas = self._figure(self.page_tracks)
        self.accuracy_fig, self.accuracy_canvas = self._figure(self.page_accuracy)
        self.motion_fig, self.motion_canvas = self._figure(self.page_motion)
        self.log_box = tk.Text(self.page_events, background=WHITE, foreground=INK, relief='flat',
                               font=(self._font, 10), padx=20, pady=16, wrap='word', state='disabled')
        self.log_box.pack(fill='both', expand=True)
        self._log('Test: recorded frames. Run: asynchronous observations (no frame markers). ROV sequence is simulation-only.')

        transport = ttk.Frame(main, padding=(8, 10, 8, 0))
        transport.grid(row=2, column=0, sticky='ew')
        transport.grid_columnconfigure(0, weight=1)
        self.timeline = ttk.Scale(transport, from_=0, to=1, command=self._on_scrub)
        self.timeline.grid(row=0, column=0, sticky='ew', padx=(0, 12))
        self.progress_var = tk.StringVar(value='Frame 0 / 0')
        ttk.Label(transport, textvariable=self.progress_var,
                  style='Hint.TLabel').grid(row=0, column=1, sticky='e')
        self.footer_var = tk.StringVar(value='Forecast error is scored against future observed positions, not current samples.')
        self.footer_label = ttk.Label(main, textvariable=self.footer_var, style='Hint.TLabel',
                                      wraplength=750)
        self.footer_label.grid(row=3, column=0, sticky='w', pady=(3, 0))

    def _responsive_layout(self, event):
        if event.widget is not self:
            return
        if event.width < 1180 and self.header_hint.winfo_manager():
            self.header_hint.pack_forget()
        elif event.width >= 1180 and not self.header_hint.winfo_manager():
            self.header_hint.pack(side='left', pady=4, after=self.header_title)
        self.footer_label.configure(wraplength=max(320, event.width - 380))

    def _figure(self, parent):
        holder = ttk.Frame(parent, style='Surface.TFrame')
        holder.pack(fill='both', expand=True, padx=6, pady=6)
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=WHITE)
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, holder, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
        # Matplotlib's stock toolbar supports pan, zoom, rotation (3D), reset and export.
        return fig, canvas

    def _mouse_scroll(self, event):
        amount = -1 if getattr(event, 'delta', 0) > 0 else 1
        self.sidebar_canvas.yview_scroll(amount * 2, 'units')

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style='Heading.TLabel').pack(anchor='w', pady=(13, 7))
        ttk.Separator(parent).pack(fill='x', pady=(0, 8))

    def _label(self, parent, text):
        ttk.Label(parent, text=text, style='SidebarHint.TLabel').pack(anchor='w', pady=(5, 0))

    def _entry(self, parent, key):
        box = ttk.Entry(parent, textvariable=self.vars[key])
        box.pack(fill='x', pady=(2, 3))
        return box

    def _field_pair(self, parent, first, first_key, second, second_key):
        row = ttk.Frame(parent, style='Surface.TFrame')
        row.pack(fill='x')
        row.columnconfigure(0, weight=1, uniform='pair')
        row.columnconfigure(1, weight=1, uniform='pair')
        for index, (title, key) in enumerate([(first, first_key), (second, second_key)]):
            cell = ttk.Frame(row, style='Surface.TFrame')
            cell.grid(row=0, column=index, sticky='ew', padx=(0, 6) if index == 0 else (6, 0))
            self._label(cell, title)
            self._entry(cell, key)

    def _refresh_datasets(self):
        datasets = discover_datasets()
        self._dataset_keys.clear()
        for path in datasets:
            label = str(path.relative_to(ROOT))
            self._dataset_keys[label] = path
        saved = self.vars['dataset'].get()
        saved_path = Path(saved).expanduser()
        if saved_path.is_absolute() and saved_path.is_file():
            self._dataset_keys[str(saved_path)] = saved_path
        self.dataset_combo['values'] = list(self._dataset_keys)
        if saved in self._dataset_keys:
            self.dataset_combo.set(saved)
        elif self._dataset_keys:
            self.dataset_combo.set(next(iter(self._dataset_keys)))
            self.vars['dataset'].set(self.dataset_combo.get())

    def _browse_csv(self):
        filename = filedialog.askopenfilename(title='Choose tracking CSV',
                                               filetypes=[('CSV tracking data', '*.csv'), ('All files', '*.*')])
        if not filename:
            return
        path = Path(filename).resolve()
        label = str(path)
        self._dataset_keys[label] = path
        self.dataset_combo['values'] = list(self._dataset_keys)
        self.dataset_combo.set(label)
        self._dataset_changed()

    def _dataset_changed(self, event=None):
        key = self.dataset_combo.get()
        self.vars['dataset'].set(key)
        path = self._dataset_keys.get(key)
        if path:
            dt = recommended_dt(path)
            self.vars['frame_dt'].set(f'{dt:.6g}')
            self.vars['stale'].set(f'{6 * dt:.6g}')  # Matches the previous six-frame setting.
            # Ten real frames gives 5 s for RD or about 0.667 s for 15 fps.
            # This is a starting point: the operator may override either value.
            self.vars['horizon'].set(f'{10 * dt:.6g}')
        self._log(f'Dataset: {key} (change the frame period if your recording differs).')

    def _set_mode(self):
        if self.session is not None:
            self._stop(clear=True)
        mode = self.vars['mode'].get()
        self.test_source.pack_forget()
        self.live_source.pack_forget()
        if mode == 'Test':
            self.test_source.pack(fill='x')
        else:
            self.live_source.pack(fill='x')
        if mode == 'Run':
            self.speed_frame.pack_forget()
            self.step_button.pack_forget()
        else:
            self.speed_frame.pack(fill='x', pady=(0, 8), before=self.lock_button)
            self.step_button.pack(side='left', padx=3, before=self.stop_button)
        self.pause_button.configure(state='disabled')
        self.timeline.configure(state='normal' if mode == 'Test' else 'disabled')
        self._set_status('IDLE', MUTED)
        self.footer_var.set('Test: recorded data with live replay & evaluation.' if mode == 'Test' else
                            'Run: individual tracking observations. ROV sequence is simulation only; no hardware commands.')
        self._update_rov_controls()

    def _collect_config(self):
        dt = pos_number(self.vars['frame_dt'].get(), 'Frame period', maximum=10)
        horizon = pos_number(self.vars['horizon'].get(), 'Forecast horizon', maximum=120)
        stale = pos_number(self.vars['stale'].get(), 'ID timeout', maximum=10000)
        max_id = int(float(self.vars['max_id'].get()))
        if max_id < 0 or max_id > 10000000:
            raise ValueError('Maximum ID must be between 0 and 10,000,000')
        target_z = float(self.vars['target_z'].get())
        if not math.isfinite(target_z):
            raise ValueError('Target depth must be finite')
        xy = pos_number(self.vars['xy_tol'].get(), 'XY tolerance')
        zt = pos_number(self.vars['z_tol'].get(), 'Z tolerance')
        threshold = pos_number(self.vars['lull_threshold'].get(), 'Calm speed threshold')
        window = pos_number(self.vars['lull_window'].get(), 'Lull window', maximum=120)
        min_calm = pos_number(self.vars['calm_min'].get(), 'Minimum calm time', maximum=120)
        return {'dt': dt, 'horizon': horizon, 'stale': stale, 'max_id': max_id,
                'target_z': target_z, 'xy_tol': xy, 'z_tol': zt, 'threshold': threshold,
                'lull_window': window, 'calm_min': min_calm,
                }

    def _apply_alignment_settings(self):
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror('Invalid alignment settings', str(exc))
            return
        if self.session:
            self.config.update({key: cfg[key] for key in ('target_z', 'xy_tol', 'z_tol', 'threshold')})
            if (self.session.lull_threshold_mps != cfg['threshold'] or
                    self.session.calm_window_s != cfg['lull_window'] or
                    self.session.calm_min_s != cfg['calm_min']):
                self.session.lull_threshold_mps = cfg['threshold']
                self.session.calm_window_s = cfg['lull_window']
                self.session.calm_min_s = cfg['calm_min']
                self.session.centroid_window.clear()
                self.session.calm_streak = 0
                self.session.calm_elapsed_s = 0.0
            if self.cursor >= 0:
                self._show_snapshot(self.session.snapshots[self.cursor], force=True)
        self._save_current()
        self._log(f'Alignment set: X/Y ±{cfg["xy_tol"]:.3f} m, '
                  f'Z {cfg["target_z"]:.3f} ±{cfg["z_tol"]:.3f} m, '
                  f'calm threshold {cfg["threshold"]:.4f} m/s.')

    def _start(self):
        self._stop(quiet=True, clear=True)
        try:
            cfg = self._collect_config()
            if self.vars['mode'].get() == 'Test':
                path = self._dataset_keys.get(self.dataset_combo.get())
                if not path:
                    raise ValueError('Select a dataset CSV first')
                group = {'Auto': 'auto', 'Rows': 'rows', 'Repeated IDs': 'repeated'}[self.vars['grouping'].get()]
                self.frames = load_csv_frames(path, cfg['dt'], group)
            else:
                self.frames = []
                live_source.clear()
            session_cls = TrackingSession if self.vars['mode'].get() == 'Test' else ObservationSession
            self.session = session_cls(cfg['dt'], cfg['horizon'], cfg['stale'], cfg['max_id'],
                                       cfg['threshold'], cfg['lull_window'], cfg['calm_min'])
            self.config = cfg
            self.source_label = path.name if self.vars['mode'].get() == 'Test' else 'camera_tracking'
            self.live_observations = []
            self.output_saved = False
            self.last_rejected_count = 0
            self.lock_ids = None
            self.lock_button.configure(text='Lock current cluster')
            self.cursor = -1
            self._reset_dashboard()
            self._save_current()
            if self.vars['mode'].get() == 'Test':
                self.timeline.configure(from_=0, to=max(1, len(self.frames) - 1))
                self._set_status('PLAYING', TEAL)
                self.playing = True
                self.pause_button.configure(state='normal', text='Pause')
                self._log(f'Replay started · {path.name} · {len(self.frames):,} frames · '
                          f'{cfg["dt"]:g} s/frame · UKF · forecast +{self.session.horizon_s:.3f} s.')
                if self.session.steps >= len(self.frames) - 1:
                    duration = self.frames[-1].time_s - self.frames[0].time_s
                    self._log(f'Accuracy unavailable at this horizon: recording is {duration:.2f} s, '
                              f'forecast is +{self.session.horizon_s:.2f} s. '
                              'Choose a shorter horizon to score predictions.')
                self._tick()
            else:
                self._set_status('WAITING FOR TRACKING', BLUE)
                self.playing = True
                self.pause_button.configure(state='disabled')
                self._log(f'Run started · independent observations · nominal forecast step {cfg["dt"]:g} s; '
                          'ROV sequence simulation only, with explicit operator approval.')
                self._update_rov_controls()
        except (ValueError, KeyError, OSError, ImportError) as exc:
            self._stop(quiet=True, clear=True)
            messagebox.showerror('Cannot start', str(exc))
            self._set_status('ERROR', RED)

    def _reschedule(self):
        if self.session and self.playing and self.vars['mode'].get() == 'Test':
            if self.timer_id:
                self.after_cancel(self.timer_id)
                self.timer_id = None
            self._schedule_tick()

    def _schedule_tick(self):
        if not self.playing or self.vars['mode'].get() != 'Test':
            return
        speed_name = self.vars['speed'].get()
        speed = None if speed_name == 'Max' else float(speed_name.rstrip('×'))
        interval = 1 if speed is None else max(1, int(1000 * self.config['dt'] / speed))
        self.timer_id = self.after(interval, self._tick)

    def _tick(self):
        self.timer_id = None
        if not self.playing or not self.session:
            return
        next_index = self.cursor + 1
        if next_index >= len(self.frames):
            self._pause(end=True)
            return
        try:
            if next_index < len(self.session.snapshots):
                snap = self.session.snapshots[next_index]
            else:
                snap = self.session.process(self.frames[next_index])
            self.cursor = next_index
            self._show_snapshot(snap)
            if self.cursor == len(self.frames) - 1:
                self._pause(end=True)
            else:
                self._schedule_tick()
        except Exception as exc:
            self._pause()
            self._set_status('ERROR', RED)
            self._log(f'Processing error: {exc}')
            messagebox.showerror('Tracking failed', str(exc))

    def _pause(self, end=False):
        self.playing = False
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.pause_button.configure(text='Resume' if not end else 'Finished',
                                    state='normal' if not end else 'disabled')
        self._set_status('COMPLETE' if end else 'PAUSED', TEAL if end else AMBER)
        if end:
            try:
                self._save_outputs()
            except OSError as exc:
                self._log(f'Could not save output: {exc}')
            self._log('Replay complete. Use the timeline to review frames and export evaluated results.')

    def _pause_resume(self):
        if self.vars['mode'].get() != 'Test' or self.session is None:
            return
        if self.playing:
            self._pause()
        else:
            if self.cursor >= len(self.frames) - 1:
                self._seek_to(0)
            self.playing = True
            self.pause_button.configure(text='Pause', state='normal')
            self._set_status('PLAYING', TEAL)
            self._schedule_tick()

    def _step(self):
        if self.vars['mode'].get() != 'Test':
            return
        if self.session is None:
            self._start()
        if self.session is None:
            return
        self._pause()
        if self.cursor + 1 < len(self.frames):
            try:
                self.cursor += 1
                if self.cursor >= len(self.session.snapshots):
                    self.session.process(self.frames[self.cursor])
                self._show_snapshot(self.session.snapshots[self.cursor], force=True)
            except Exception as exc:
                messagebox.showerror('Step failed', str(exc))

    def _on_scrub(self, value):
        if self._ignore_slider or not self.session or self.vars['mode'].get() != 'Test':
            return
        requested = int(round(float(value)))
        if requested == self.cursor and self.seek_target is None:
            return
        self._pause()
        self._seek_to(requested)

    def _seek_to(self, target):
        if not self.session:
            return
        self.seek_target = max(0, min(target, len(self.frames) - 1))
        if self.seek_id:
            self.after_cancel(self.seek_id)
        self._seek_batch()

    def _seek_batch(self):
        self.seek_id = None
        if not self.session or self.seek_target is None:
            return
        target = self.seek_target
        try:
            deadline = time.monotonic() + 0.035
            while len(self.session.snapshots) <= target and time.monotonic() < deadline:
                i = len(self.session.snapshots)
                self.session.process(self.frames[i])
            if len(self.session.snapshots) <= target:
                self.progress_var.set(f'Calculating frame {len(self.session.snapshots):,} / {target + 1:,}…')
                self.seek_id = self.after(3, self._seek_batch)
            else:
                self.cursor = target
                self.seek_target = None
                self._show_snapshot(self.session.snapshots[target], force=True)
        except Exception as exc:
            self.seek_target = None
            self._set_status('ERROR', RED)
            messagebox.showerror('Seek failed', str(exc))

    def _poll_live(self):
        if self.playing and self.session and self.vars['mode'].get() == 'Run':
            if live_source.dropped_observations:
                self._log('Live observation buffer overflow. Stopping: data was lost.')
                self._stop()
                self._set_status('DATA LOSS', RED)
            else:
                last_snap = None
                for obs in live_source.read(256):
                    try:
                        last_snap = self.session.observe(obs)
                        self.live_observations.append(obs)
                        self.last_packet = time.monotonic()
                    except (ValueError, KeyError, OSError) as exc:
                        self._log(f'Live input error: {exc}')
                        self._stop()
                        self._set_status('ERROR', RED)
                        break
                if self.playing:
                    if last_snap is None:
                        last_snap = self.session.expire_at(live_source.elapsed_time())
                    if last_snap is not None:
                        self.cursor = last_snap.index
                        self._show_snapshot(last_snap)
                        self._set_status('LIVE' if last_snap.fresh else 'WAITING',
                                         TEAL if last_snap.fresh else AMBER)
                    if (self.last_packet and time.monotonic() - self.last_packet >
                            max(2.0, self.config['stale'] * 2)):
                        self._set_status('NO SIGNAL', AMBER)
                        self.alignment_var.set('Tracking data stopped arriving; alignment unavailable.')
                        if self.rov_controller.state in (ROVState.APPROACH, ROVState.COLLECT):
                            self.rov_controller.stop('Tracking signal lost')
        self._update_rov_controls()
        self.after(80, self._poll_live)

    def _stop(self, quiet=False, clear=False):
        if (self.rov_controller.state in (ROVState.APPROACH, ROVState.COLLECT) or
                self.rov_controller.armed):
            self.rov_controller.stop('Tracking session stopped')
        self.playing = False
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        if self.seek_id:
            self.after_cancel(self.seek_id)
            self.seek_id = None
        self.seek_target = None
        if self.session and self.session.snapshots and not self.output_saved:
            try:
                self._save_outputs()
            except OSError as exc:
                self._log(f'Could not save output: {exc}')
        self.last_packet = None
        self.pause_button.configure(state='disabled', text='Pause')
        if clear:
            self.session = None
            self.frames = []
            self.live_observations = []
            self.cursor = -1
            self.lock_ids = None
            self._reset_dashboard()
        self._update_rov_controls()
        if not quiet:
            self._set_status('STOPPED' if self.session else 'IDLE', MUTED)
            self._log('Tracking stopped. Results remain available for review/export; Start begins a new session.')

    def _reset_dashboard(self):
        for value in self.metrics.values():
            value.set('—')
        self.progress_var.set('No observations')
        self.alignment_var.set('No tracking data')
        self.motion_var.set('Motion and outliers available after tracking begins.')
        self._ignore_slider = True
        self.timeline.set(0)
        self._ignore_slider = False
        self.track_combo['values'] = ['Cluster centroid']
        self.vars['selected_track'].set('Cluster centroid')
        for fig, canvas in [(self.scene_fig, self.scene_canvas), (self.track_fig, self.track_canvas),
                            (self.accuracy_fig, self.accuracy_canvas),
                            (self.motion_fig, self.motion_canvas)]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(.5, .5, 'Start a session to see live data', va='center', ha='center',
                    color=MUTED, fontsize=13, transform=ax.transAxes)
            canvas.draw_idle()

    def _set_status(self, status, color):
        self.live_status.configure(text=f'●  {status}', fg=color)

    def _save_current(self):
        self.settings = {key: var.get() for key, var in self.vars.items()}
        self.settings['stale_units'] = 'seconds'
        self.settings['dataset'] = self.dataset_combo.get()
        self.settings['show_raw'] = self.show_raw.get()
        self.settings['show_estimate'] = self.show_estimate.get()
        self.settings['show_forecast'] = self.show_forecast.get()
        save_settings(self.settings)

    def _toggle_lock(self):
        if self.lock_ids is not None:
            self.lock_ids = None
            self.lock_button.configure(text='Lock current cluster')
            self._log('Cluster unlocked. Alignment tracks all currently visible mussels.')
        elif self.session and self.cursor >= 0:
            snap = self.session.snapshots[self.cursor]
            self.lock_ids = set(snap.tracks) if self.vars['mode'].get() == 'Run' else set(
                mid for mid, row in snap.tracks.items() if row['fresh'])
            if not self.lock_ids:
                messagebox.showinfo('No active tracks', 'Wait for a valid mussel observation.')
                return
            self.lock_button.configure(text=f'Unlock cluster ({len(self.lock_ids)} IDs)')
            self._log(f'Cluster locked to IDs: {sorted(self.lock_ids)}. New IDs will not change the target.')
        else:
            messagebox.showinfo('No data', 'Start Test or Run mode first.')
            return
        self._draw(force=True)

    def _evaluate_alignment(self, snap):
        if self.lock_ids is None:
            return 'Lock a cluster to inspect alignment and calm conditions.'
        items = [item for mid, item in snap.tracks.items() if mid in self.lock_ids]
        fresh = any(item['fresh'] for item in items)
        if not items or (not fresh and self.vars['mode'].get() == 'Test') or (
                self.vars['mode'].get() == 'Run' and self.last_packet and
                                     time.monotonic() - self.last_packet > max(2.0, self.config['stale'] * 2)):
            return 'Locked target unavailable · no fresh observations.'
        cx, cy, cz = np.mean([r['filtered'] for r in items], axis=0)
        xy = self.config['xy_tol']
        ztol = self.config['z_tol']
        dz = cz - self.config['target_z']
        aligned = abs(cx) <= xy and abs(cy) <= xy and abs(dz) <= ztol
        status = ('CONDITIONS MET (advisory only)' if aligned and snap.is_calm else
                  'Aligned; waiting for low motion' if aligned else 'Not aligned')
        if self.vars['mode'].get() == 'Run':
            status += ' · asynchronous / unverified' if aligned else ' · asynchronous'
        return (f'{status}\nΔX {cx:+.3f} m   ΔY {cy:+.3f} m\n'
                f'ΔZ {dz:+.3f} m   ·   motion: '
                f'{f"{snap.speed_mps:.3f} m/s" if snap.speed_mps is not None else "unknown"}\n'
                'Scene-level calm is indicative only; no ROV command is sent.')

    def _show_snapshot(self, snap: Snapshot, force=False):
        if not self.session:
            return
        self.metrics['time'].set(f'{snap.time_s:.2f} s')
        visible = snap.active if self.vars['mode'].get() == 'Run' else snap.fresh
        self.metrics['active'].set(f'{visible} / {snap.total}')
        self.metrics['horizon'].set(f'+{self.session.horizon_s:.2f} s')
        self.metrics['rmse'].set(format_mm(snap.rmse_m))
        gain = None if not snap.hold_rmse_m or snap.rmse_m is None else 100 * (1 - snap.rmse_m / snap.hold_rmse_m)
        self.metrics['gain'].set(f'{gain:+.1f} %' if gain is not None else '—')
        if self.vars['mode'].get() == 'Test':
            progress = f'Frame {snap.index + 1:,} / {len(self.frames):,}'
        elif snap.index:
            progress = f'Observation {len(self.live_observations):,} · live' 
        else:
            progress = 'Observation 1 · live'
        self.progress_var.set(progress)
        self.alignment_var.set(self._evaluate_alignment(snap))
        mid = self._selected_mid()
        velocity = snap.tracks.get(mid, {}).get('velocity') if mid is not None else None
        if velocity is not None:
            speed_mps = float(np.linalg.norm(velocity))
            extra = f'Estimated mussel speed: {speed_mps * 1000:.1f} mm/s'
        else:
            extra = 'Select a mussel for its filtered velocity'
        calm = 'CALM' if snap.is_calm else 'MOVING / UNCONFIRMED'
        self.motion_var.set(f'{extra}\nCluster speed: '
                            f'{snap.speed_mps * 1000:.1f} mm/s' if snap.speed_mps is not None
                            else f'{extra}\nCluster speed: —')
        self.motion_var.set(self.motion_var.get() + f' · {calm}\nRejected measurements: {snap.rejected_total}')
        if mid is not None and self.vars['mode'].get() == 'Run' and mid in snap.tracks:
            dt = snap.tracks[mid]['frame_interval_s']
            self.motion_var.set(self.motion_var.get() + f'\nLast ID interval: {dt * 1000:.1f} ms')
        if snap.rejected_total > self.last_rejected_count and snap.index == len(self.session.snapshots) - 1:
            self._log(f'Outlier gate: {snap.rejected_total - self.last_rejected_count} observation(s) rejected.')
        self.last_rejected_count = max(self.last_rejected_count, snap.rejected_total)
        ids = ['Cluster centroid'] + [f'Mussel {i}' for i in sorted(self.session.tracks)]
        if list(self.track_combo['values']) != ids:
            self.track_combo['values'] = ids
            if self.vars['selected_track'].get() not in ids:
                self.vars['selected_track'].set('Cluster centroid')
        if self.vars['mode'].get() == 'Test':
            self._ignore_slider = True
            self.timeline.set(snap.index)
            self._ignore_slider = False
        self.footer_var.set(f'Accuracy: {snap.n_scores:,} forecast–target comparisons · '
                            f'3D RMSE {format_mm(snap.rmse_m)} · hold baseline {format_mm(snap.hold_rmse_m)}. '
                            f'Rejected observations: {snap.rejected_total}. Forecasts shown at target times.')
        self._draw(force=force)

    def _selected_mid(self):
        choice = self.vars['selected_track'].get()
        try:
            return int(choice.removeprefix('Mussel ')) if choice.startswith('Mussel ') else None
        except ValueError:
            return None

    def _draw(self, force=False):
        if not self.session or self.cursor < 0:
            return
        if not force and time.monotonic() - self.last_display_draw < .085:
            return
        self.last_display_draw = time.monotonic()
        page = self.pages.index(self.pages.select())
        snap = self.session.snapshots[self.cursor]
        if page == 0:
            self._draw_scene(snap)
        elif page == 1:
            self._draw_tracks(snap)
        elif page == 2:
            self._draw_accuracy(snap)
        elif page == 3:
            self._draw_motion(snap)

    def _draw_scene(self, snap):
        fig = self.scene_fig
        old_ax = fig.axes[0] if fig.axes and hasattr(fig.axes[0], 'elev') else None
        view = (old_ax.elev, old_ax.azim) if old_ax is not None else (22, -65)
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(*view)
        fig.suptitle('Camera-relative 3D positions and forecasts (m)', color=INK, fontsize=12, y=.97)
        ax.set_xlabel('X (m) · left/right', labelpad=3, fontsize=9)
        ax.set_ylabel('Y (m) · up/down', labelpad=3, fontsize=9)
        ax.set_zlabel('Z (m) · forward/back', labelpad=3, fontsize=9)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=4))
            axis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=8, pad=0)
        current = list(snap.tracks.items())
        points = [[0, 0, self.config['target_z']]]
        max_labels = 18
        for index, (mid, row) in enumerate(current):
            pos, future = row['filtered'], row['forecast']
            points.extend([pos, future])
            is_locked = self.lock_ids is None or mid in self.lock_ids
            color = PALETTE[index % len(PALETTE)] if is_locked else '#B6BEC9'
            if self.show_estimate.get():
                ax.scatter(*pos, color=color, s=39 if row['fresh'] else 24,
                           marker='o' if row['fresh'] else 'x', alpha=1 if row['fresh'] else .55)
                if index < max_labels:
                    ax.text(*pos, f' {mid}', fontsize=8, color=color)
            if self.show_forecast.get():
                ax.plot([pos[0], future[0]], [pos[1], future[1]], [pos[2], future[2]],
                        color=color, linewidth=1.1, linestyle='--', alpha=.6)
                ax.scatter(*future, color=color, s=15, marker='^', alpha=.65)
            if self.show_raw.get() and row['raw'] is not None:
                ax.scatter(*row['raw'], color='#718096', s=9, marker='.', alpha=.6)
        if snap.centroid:
            points.append(snap.centroid)
            ax.scatter(*snap.centroid, marker='*', s=220, c=[TEAL], label='Scene centroid')
        if snap.centroid_forecast:
            points.append(snap.centroid_forecast)
            ax.scatter(*snap.centroid_forecast, marker='D', s=60, c=[BLUE], label='Centroid forecast')
        ax.scatter(0, 0, self.config['target_z'], marker='+', s=190, color=RED, linewidth=2.4,
                   label='Alignment target')
        arr = np.asarray(points)
        ranges = np.ptp(arr, axis=0)
        widths = np.maximum(ranges * 1.4, [.13, .13, .13])
        mid = (np.min(arr, axis=0) + np.max(arr, axis=0)) / 2
        for f, center, width in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid, widths):
            f(float(center - width / 2), float(center + width / 2))
        try:
            ax.set_box_aspect((1.1, 1, 0.85), zoom=1.05)
        except TypeError:
            ax.set_box_aspect((1.1, 1, 0.85))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.035, .9), fontsize=8, frameon=False)
        ax.grid(True, alpha=.3)
        fig.subplots_adjust(left=.07, right=.96, bottom=.05, top=.93)
        self.scene_canvas.draw_idle()

    def _make_series(self):
        mid = self._selected_mid()
        if mid is not None:
            history = self.session.track_history(self.cursor, mid)
            rows = [(r['time_s'], r['raw'], r['filtered'], r['forecast']) for r in history]
        else:
            rows = []
            for snap in self.session.snapshots[:self.cursor + 1]:
                if snap.centroid is None:
                    continue
                measured = ([row['raw'] for row in snap.tracks.values()
                             if row['fresh'] and row['raw'] is not None]
                            if self.vars['mode'].get() == 'Test' else [])
                raw_centroid = tuple(float(x) for x in np.mean(measured, axis=0)) if measured else None
                rows.append((snap.time_s, raw_centroid, snap.centroid, snap.centroid_forecast))
        return rows

    def _draw_tracks(self, snap):
        fig = self.track_fig
        fig.clear()
        axes = fig.subplots(3, 1, sharex=True)
        series = self._make_series()
        label = self.vars['selected_track'].get()
        if series:
            t = np.asarray([r[0] for r in series])
            raw = np.array([r[1] if r[1] is not None else (np.nan,) * 3 for r in series])
            est = np.array([r[2] for r in series])
            future = np.array([r[3] for r in series])
            for dimension, ax in enumerate(axes):
                theme_axes(ax)
                if self.show_raw.get():
                    ax.plot(t, raw[:, dimension], linestyle='', marker='.', markersize=3,
                            alpha=.6, color='#8A9BB0', label='Measured')
                if self.show_estimate.get():
                    ax.plot(t, est[:, dimension], color=TEAL, linewidth=1.5, label='Filtered')
                if self.show_forecast.get():
                    ax.plot(t + self.session.horizon_s, future[:, dimension], color=BLUE,
                            alpha=.8, linestyle='--', linewidth=1.25, label='Forecast at target time')
                ax.axvline(snap.time_s, color=AMBER, linestyle=':', linewidth=1, alpha=.85)
                ax.set_ylabel(f'{"XYZ"[dimension]} (m)', color=INK, fontsize=10)
        else:
            for ax in axes:
                theme_axes(ax)
        axes[-1].set_xlabel('Dataset / sensor time (s)', color=INK, fontsize=10)
        fig.suptitle(f'{label} · measured, filtered and +{self.session.horizon_s:.2f} s forecast',
                     x=.52, y=.98, fontsize=11, color=INK)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .94),
                       ncol=3, frameon=False, fontsize=8)
        fig.subplots_adjust(left=.105, right=.96, top=.82, bottom=.18, hspace=.29)
        self.track_canvas.draw_idle()

    def _draw_accuracy(self, snap):
        fig = self.accuracy_fig
        fig.clear()
        ax = fig.add_subplot(211)
        bar = fig.add_subplot(212)
        mid = self._selected_mid()
        scores = self.session.scores_at(self.cursor, mid)
        theme_axes(ax)
        theme_axes(bar)
        ax.set_title('Future-target error · lower is better', fontsize=12, color=INK)
        ax.set_ylabel('3D error (mm)', fontsize=9, color=INK)
        if scores:
            t = [s.time_s for s in scores]
            ax.plot(t, [1000 * s.error_m for s in scores], color=BLUE, linewidth=1.1,
                    alpha=.8, label='UKF forecast')
            ax.plot(t, [1000 * s.hold_m for s in scores], color='#97A4B7', linewidth=1.1,
                    alpha=.85, label='Hold baseline')
            ax.legend(loc='upper right', frameon=False, fontsize=8)
            errors = np.array([s.forecast_error_m for s in scores])
            hold = np.array([s.hold_error_m for s in scores])
            xs = np.arange(3)
            w = .34
            bar.bar(xs - w / 2, 1000 * np.sqrt(np.mean(errors**2, axis=0)), width=w,
                    color=BLUE, label='Forecast')
            bar.bar(xs + w / 2, 1000 * np.sqrt(np.mean(hold**2, axis=0)), width=w,
                    color='#B5C0CE', label='Hold')
            bar.set_xticks(xs, ['X', 'Y', 'Z'])
            bar.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, 'No forecasts have reached a valid target yet',
                    ha='center', va='center', color=MUTED, transform=ax.transAxes)
            bar.set_xticks([0, 1, 2], ['X', 'Y', 'Z'])
        bar.set_ylabel('Per-axis RMSE (mm)', fontsize=9, color=INK)
        count = len(scores)
        stats = self.session.metrics_at(self.cursor, mid)
        gain = (f'{stats["improvement_pct"]:+.1f}%'
                if stats['improvement_pct'] is not None else '—')
        subtitle = (f'{count:,} evaluated targets   •   RMSE {format_mm(stats["rmse_m"])}   •   '
                    f'Hold {format_mm(stats["hold_rmse_m"])}   •   Gain {gain}')
        fig.text(.5, .015, subtitle, ha='center', color=MUTED, fontsize=9)
        fig.subplots_adjust(left=.11, right=.96, top=.92, bottom=.16, hspace=.50)
        self.accuracy_canvas.draw_idle()

    def _draw_motion(self, snap):
        fig = self.motion_fig
        fig.clear()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax)
        for panel in (ax, ax2):
            theme_axes(panel)
        mid = self._selected_mid()
        if mid is not None:
            rows = self.session.track_history(self.cursor, mid)
            label = f'Mussel {mid}'
            t = [row['time_s'] for row in rows]
            speed = [1000 * float(np.linalg.norm(row['velocity'])) for row in rows]
            rejected = [row for row in rows if row['rejected']]
            if rows:
                ax.plot(t, speed, color=TEAL, linewidth=1.5, label='Filtered speed')
                if rejected:
                    ax.scatter([row['time_s'] for row in rejected],
                               [1000 * float(np.linalg.norm(row['velocity'])) for row in rejected],
                               color=RED, marker='x', s=40, label='Rejected observation')
                ax2.step(t, [row['rejected_count'] for row in rows], where='post',
                         color=RED, label='Cumulative rejections')
        else:
            label = 'Cluster centroid'
            rows = self.session.snapshots[:self.cursor + 1]
            t = [row.time_s for row in rows if row.speed_mps is not None]
            speed = [1000 * row.speed_mps for row in rows if row.speed_mps is not None]
            if t:
                ax.plot(t, speed, color=TEAL, linewidth=1.5, label='Windowed path speed')
                ax.axhline(1000 * self.config['threshold'], color=AMBER, linestyle='--',
                           linewidth=1.0, label='Lull threshold')
            ax2.step([row.time_s for row in rows], [row.rejected_total for row in rows],
                     where='post', color=RED, label='Cumulative rejections')
            for row in rows:
                if row.is_calm:
                    ax2.axvspan(row.time_s, row.time_s + self.session.dt,
                                facecolor=TEAL, alpha=.10)
        ax.set_title(f'{label} · velocity and outliers', fontsize=12, color=INK)
        ax.set_ylabel('Speed (mm/s)', fontsize=9)
        ax2.set_ylabel('Rejected measurements', fontsize=9)
        ax2.set_xlabel('Observation time (s)', fontsize=9)
        for panel in (ax, ax2):
            if panel.get_legend_handles_labels()[0]:
                panel.legend(loc='upper right', frameon=False, fontsize=8)
        fig.subplots_adjust(left=.11, right=.96, top=.90, bottom=.13, hspace=.35)
        self.motion_canvas.draw_idle()

    def _save_outputs(self):
        if not self.session or not self.session.snapshots:
            return []
        if self.output_saved:
            return self.output_files
        frames = self.frames[:len(self.session.snapshots)] if self.vars['mode'].get() == 'Test' else self.live_observations
        files = export_session(self.session, frames, self.vars['mode'].get(), self.source_label)
        self.output_files = files
        self.output_saved = True
        self._log(f'Recorded observations, predictions and summary in Outputs/: {files[0].stem}')
        return files

    def _export(self):
        try:
            files = self._save_outputs()
            if not files:
                messagebox.showinfo('Nothing to save', 'Start a session first.')
            else:
                messagebox.showinfo('Session saved', '\n'.join(str(f) for f in files))
        except OSError as exc:
            messagebox.showerror('Export failed', str(exc))

    def _current_target(self):
        if not self.session or self.cursor < 0:
            return None
        snap = self.session.snapshots[self.cursor]
        if self.lock_ids is None:
            return snap.centroid
        positions = [row['filtered'] for mid, row in snap.tracks.items() if mid in self.lock_ids]
        if not positions:
            return None
        return tuple(float(x) for x in np.mean(positions, axis=0))

    def _update_rov_controls(self):
        """Refresh simulated controller feedback and permitted operator actions."""
        controller = self.rov_controller
        if self.vars['mode'].get() == 'Run':
            controller.tick()
        while self._rov_logged < len(controller.history):
            self._log('ROV SIM: ' + controller.history[self._rov_logged])
            self._rov_logged += 1
        run = self.vars['mode'].get() == 'Run' and self.playing and self.session is not None
        state = controller.state
        if state == ROVState.FAULT:
            status = 'FAULT · ' + str(controller.fault_reason)
        elif state == ROVState.IDLE:
            status = 'IDLE · ARMED' if controller.armed else 'IDLE · DISARMED'
        elif state == ROVState.APPROACH:
            status = 'APPROACH · READY FOR APPROVAL' if controller.arrived else 'APPROACH · WAITING FOR ARRIVAL'
        else:
            status = 'COLLECT · SIMULATING'
        self.rov_status_var.set(status)
        self.rov_arm_button.configure(state='normal' if run and state == ROVState.IDLE and not controller.armed else 'disabled')
        self.align_button.configure(state='normal' if run and state == ROVState.IDLE and controller.armed and
                                    self._current_target() is not None else 'disabled')
        self.collect_button.configure(state='normal' if run and state == ROVState.APPROACH and controller.arrived else 'disabled')
        self.rov_stop_button.configure(state='normal' if run and state != ROVState.FAULT else 'disabled')
        self.rov_reset_button.configure(state='normal' if self.vars['mode'].get() == 'Run' and
                                        state == ROVState.FAULT else 'disabled')

    def _arm_rov(self):
        try:
            if not self.playing or self.vars['mode'].get() != 'Run':
                raise RuntimeError('Start Run mode first')
            self.rov_controller.arm()
        except RuntimeError as exc:
            messagebox.showwarning('Simulation unavailable', str(exc))
        self._update_rov_controls()

    def _command_alignment(self):
        target = self._current_target()
        if target is None or not self.session or not self.session.snapshots or not self.session.snapshots[-1].active:
            messagebox.showinfo('No active target', 'Wait for a live mussel position before simulating approach.')
            return
        try:
            self.rov_controller.start_approach(target)
        except (RuntimeError, ValueError) as exc:
            messagebox.showwarning('Simulation unavailable', str(exc))
        self._update_rov_controls()

    def _command_collection(self):
        try:
            self.rov_controller.approve_collection()
        except RuntimeError as exc:
            messagebox.showwarning('Simulation unavailable', str(exc))
        self._update_rov_controls()

    def _safe_stop_rov(self):
        self.rov_controller.stop('Operator simulated stop')
        self._update_rov_controls()

    def _reset_rov(self):
        try:
            self.rov_controller.reset()
        except RuntimeError as exc:
            messagebox.showwarning('Reset unavailable', str(exc))
        self._update_rov_controls()

    def _show_help(self):
        messagebox.showinfo('Using the control panel',
            'TEST: Select a dataset and replay it with Start, Pause, Step, speed or timeline. '
            'Measured positions are plotted at observation time; forecasts at the future target time.\n\n'
            'RUN: The external CV adapter calls Integration.live_input.publish_observation(ID, x, y, z) '
            'once per measurement. No timestamps or frame-completion calls are required. Arrival time is used; '
            'the app does not receive network packets directly.\n\n'
            'COORDINATES: Metres relative to the stereo camera lens centre. '
            'X is left/right, Y is up/down, Z is forward/back. Confirm axis signs with CV integration.\n\n'
            'ACCURACY: 3D RMSE (mm) is evaluated only when the future observation arrives; '
            'Hold is the last measurement used as a baseline. The event log shows rejected measurements.\n\n'
            'ROV: The four-state controller is simulation-only. Arm, request approach, wait for simulated arrival, '
            'then explicitly approve simulated collection. Safe stop latches FAULT until operator reset. '
            'No command reaches ROV hardware and the camera-coordinate centroid is not a validated vehicle target. '
            'Session observations, predictions and a summary are saved automatically in Outputs/.')

    def _log(self, message):
        stamp = time.strftime('%H:%M:%S')
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f'[{stamp}] {message}\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _close(self):
        self._save_current()
        self._stop(quiet=True, clear=True)
        self.destroy()


def launch():
    ControlPanel().mainloop()


if __name__ == '__main__':
    launch()
