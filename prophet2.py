#!/usr/bin/env python3
"""
chaos_pad.py  —  Syntezer Wektorowy (styl Prophet VS / Behringer VS Pro)
=========================================================================
4 oscylatory A B C D w rogach pada XY — joystick miesza barwy bilinearnie.

MIDI (pełne sterowanie):
  • Dialog wyboru portu MIDI
  • Note On/Off — wielogłosowość 8 głosów
  • Pitch Bend    → strojenie ±2 półtony
  • CC 1  Mod     → głębokość LFO
  • CC 2  Breath  → oś Z / barwa
  • CC 7  Volume, CC 11 Expression
  • CC 64 Sustain, CC 66 Sostenuto
  • CC 71 Resonance, CC 72-75 ADSR, CC 74 Filter Cutoff
  • CC 76 LFO Rate, CC 77 LFO Depth
  • CC 91 Reverb
  • Channel Aftertouch → vibrato + filter
  • Poly Aftertouch    → per-note modulation
  • Program Change     → wybór presetu

Fale (24): sine, saw↓, saw↑, square, triangle, pulse 25%, pulse 12%,
  sine², sine³, FM 2:1, FM 3:1, FM 5:2, FM 7:3,
  harmoniki nieparzyste/parzyste/Fibonacci,
  formanty /a/ /e/ /o/, organ, comb/string, dzwonek FM, szkło, metaliczne

Vector Envelope: animowana ścieżka XY kroków (LPM=dodaj, PPM=usuń).
Auto-ruch: random walk, Lissajous, spirala.

Uruchomienie standalone: python chaos_pad.py
Integracja z ED_Waves:   from chaos_pad import ChaosPadWindow
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import numpy as np
import math, random, os, tempfile, threading

# ── pygame ────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 1, 2048)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ── rtmidi ────────────────────────────────────────────────────────
try:
    import rtmidi
    RTMIDI_OK = True
except ImportError:
    RTMIDI_OK = False

# ── scipy ─────────────────────────────────────────────────────────
try:
    import scipy.io.wavfile as _wavfile
    from scipy.signal import lfilter as _lfilter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

SR         = 44100
MAX_VOICES = 8
TICK_MS    = 40


# ═══════════════════════════════════════════════════════════════════
#  24 KSZTAŁTY FAL
# ═══════════════════════════════════════════════════════════════════
WAVE_NAMES = [
    "sine",           "saw ↓",          "saw ↑",           "square",
    "triangle",       "pulse 25%",      "pulse 12%",       "sine²",
    "sine³",          "FM 2:1",         "FM 3:1",          "FM 5:2",
    "FM 7:3",         "harm. niep.",    "harm. parz.",     "harm. Fib.",
    "formant /a/",    "formant /e/",    "formant /o/",     "organ",
    "comb/string",    "dzwonek FM",     "szkło",           "metaliczne",
]
WAVE_IDS = [
    "sine",   "saw_dn", "saw_up", "square", "tri",
    "p25",    "p12",    "sine2",  "sine3",
    "fm21",   "fm31",   "fm52",   "fm73",
    "odd",    "even",   "fib",
    "fa",     "fe",     "fo",
    "organ",  "comb",   "bell",   "glass",  "metal",
]
assert len(WAVE_NAMES) == len(WAVE_IDS) == 24


def gen_wave(wid: str, n: int, x: float = 0.5) -> np.ndarray:
    """Generuje jeden pełny okres fali o n próbkach. x∈[0,1] = param. dodatkowy."""
    if n < 2:
        return np.zeros(max(n, 1))
    t  = np.linspace(0, 2 * math.pi, n, endpoint=False)
    tp = np.linspace(0, 1,           n, endpoint=False)

    if   wid == "sine":   w = np.sin(t)
    elif wid == "saw_dn": w = 1.0 - 2.0 * tp
    elif wid == "saw_up": w = 2.0 * tp - 1.0
    elif wid == "square": w = np.sign(np.sin(t)); w[w == 0] = 1.0
    elif wid == "tri":    w = 2.0 * np.abs(2.0 * tp - 1.0) - 1.0
    elif wid == "p25":    w = np.where(tp < .25,  1.0, -1.0)
    elif wid == "p12":    w = np.where(tp < .125, 1.0, -1.0)
    elif wid == "sine2":  s = np.sin(t); w = s * s * np.sign(s)
    elif wid == "sine3":  w = np.sin(t) ** 3
    elif wid == "fm21":
        ix = 1.0 + x * 8.0;  w = np.sin(t + ix * np.sin(2 * t))
    elif wid == "fm31":
        ix = 1.0 + x * 6.0;  w = np.sin(t + ix * np.sin(3 * t))
    elif wid == "fm52":
        ix = 0.5 + x * 5.0;  w = np.sin(t + ix * np.sin(2.5 * t))
    elif wid == "fm73":
        ix = 0.5 + x * 4.0;  w = np.sin(t + ix * np.sin(7 / 3 * t))
    elif wid == "odd":
        w = sum(np.sin(k * t) / k for k in range(1, 16, 2))
    elif wid == "even":
        w = np.sin(t) + sum(np.sin(k * t) / k for k in range(2, 14, 2))
    elif wid == "fib":
        fibs = [1, 2, 3, 5, 8, 13]
        w = sum(np.sin(k * t) / k for k in fibs)
    elif wid == "fa":   # /a/
        w = (np.sin(t) + .6 * np.sin(2*t) + .4 * np.sin(3*t)
             + .2 * np.sin(5*t) + .1 * np.sin(8*t))
    elif wid == "fe":   # /e/
        w = (np.sin(t) + .2 * np.sin(2*t) + .5 * np.sin(4*t)
             + .3 * np.sin(6*t) + .15 * np.sin(8*t))
    elif wid == "fo":   # /o/
        w = (np.sin(t) + .7 * np.sin(2*t) + .4 * np.sin(3*t)
             + .1 * np.sin(4*t))
    elif wid == "organ":
        rs  = [1, 2, 3, 4, 5, 6, 8]
        ams = [1.0, 0.8, 0.6*x, 0.5, 0.4*x, 0.3*x, 0.2*x]
        w = sum(a * np.sin(r * t) for r, a in zip(rs, ams))
    elif wid == "comb":
        rs  = range(1, 9)
        ams = [1.0 / (k ** (0.5 + x)) for k in rs]
        phs = [0.0, .05, .12, .20, .30, .42, .55, .70]
        w = sum(a * np.sin(r * t + p) for r, a, p in zip(rs, ams, phs))
    elif wid == "bell":
        ix = 1.5 + x * 5.0;  w = np.sin(t + ix * np.sin(2.756 * t))
    elif wid == "glass":
        rs  = [1.0, 2.756, 5.404, 8.932, 13.346]
        ams = [1.0, 0.50,  0.25,  0.12,   0.06]
        w = sum(a * np.sin(r * t) for r, a in zip(rs, ams))
    elif wid == "metal":
        ix = 1.0 + x * 4.0
        w  = np.sin(t + ix * np.sin(1.414 * t) + ix * .5 * np.sin(2.732 * t))
    else:
        w = np.zeros(n)

    m = np.max(np.abs(w))
    return (w / m) if m > 1e-9 else w


# ═══════════════════════════════════════════════════════════════════
#  DSP
# ═══════════════════════════════════════════════════════════════════
def _adsr_env(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    ia  = min(int(a * SR), n)
    id_ = min(int(d * SR), n - ia)
    ir  = min(int(r * SR), n)
    isu = max(0, n - ia - id_ - ir)
    segs = []
    if ia:  segs.append(np.linspace(0, 1, ia))
    if id_: segs.append(np.linspace(1, s, id_))
    if isu: segs.append(np.full(isu, s))
    if ir:  segs.append(np.linspace(s, 0, ir))
    env = np.concatenate(segs) if segs else np.ones(n)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def biquad_lp(wave: np.ndarray, cutoff_hz: float, Q: float = 0.707) -> np.ndarray:
    """Biquad lowpass (2nd order IIR)."""
    cutoff_hz = max(30.0, min(cutoff_hz, SR / 2.0 - 10.0))
    Q         = max(0.1, Q)
    w0    = 2 * math.pi * cutoff_hz / SR
    sw, cw = math.sin(w0), math.cos(w0)
    alpha = sw / (2.0 * Q)
    b0 = (1 - cw) / 2;  b1 = 1 - cw;  b2 = b0
    a0 = 1 + alpha;      a1 = -2 * cw; a2 = 1 - alpha
    b  = np.array([b0 / a0, b1 / a0, b2 / a0])
    a  = np.array([1.0, a1 / a0, a2 / a0])
    if SCIPY_OK:
        return _lfilter(b, a, wave)
    # Pure-Python fallback (wolniejszy, ale zawsze działa)
    n   = len(wave)
    out = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = wave[i]
        y0 = b[0]*xi + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
        out[i] = y0
        x2, x1, y2, y1 = x1, xi, y1, y0
    return out


def multi_echo_reverb(wave: np.ndarray, wet: float = 0.3) -> np.ndarray:
    """Szybki multi-tap echo jako reverb (bez splotu, czysto numpy)."""
    if wet < 0.005:
        return wave
    n   = len(wave)
    out = wave.copy()
    taps = [
        (int(.029 * SR), .50), (int(.055 * SR), .30),
        (int(.089 * SR), .18), (int(.137 * SR), .10),
    ]
    for delay, g in taps:
        if delay < n:
            out[delay:] = out[delay:] + wave[:n - delay] * g
    mx = np.max(np.abs(out))
    if mx > 1e-9:
        out /= mx
    return wave * (1 - wet) + out * wet


def blend_weights(nx: float, ny: float):
    """Bilinearne wagi A(TL) B(TR) C(BR) D(BL). Suma = 1."""
    wA = (1 - nx) * (1 + ny) / 4
    wB = (1 + nx) * (1 + ny) / 4
    wC = (1 + nx) * (1 - ny) / 4
    wD = (1 - nx) * (1 - ny) / 4
    return wA, wB, wC, wD


def note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def note_name(note: int) -> str:
    ns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return ns[note % 12] + str(note // 12 - 1)


# ═══════════════════════════════════════════════════════════════════
#  Ustawienia oscylatora
# ═══════════════════════════════════════════════════════════════════
class OscSettings:
    __slots__ = ('wave', 'coarse', 'fine', 'level', 'extra')
    def __init__(self):
        self.wave   = "sine"
        self.coarse = 0
        self.fine   = 0.0
        self.level  = 1.0
        self.extra  = 0.5

    def det_st(self) -> float:
        return self.coarse + self.fine / 100.0


# ═══════════════════════════════════════════════════════════════════
#  Synteza jednego głosu
# ═══════════════════════════════════════════════════════════════════
def synthesize_voice(
        note: int, vel: int,
        oscs: dict,           # {'A':OscSettings, 'B':..., 'C':..., 'D':...}
        nx: float, ny: float,
        adsr: tuple,          # (a, d, s, r)
        pitch_bend: float = 0.0,
        flt_cut:    float = 20000.0,
        flt_res:    float = 0.0,
        reverb:     float = 0.0,
        lfo_depth:  float = 0.0,
        lfo_rate:   float = 5.0,
        aftertouch: float = 0.0,
        duration:   float = 3.5,
) -> np.ndarray:
    """Generuje bufor PCM (float32) dla jednej nuty."""
    f0 = note_to_hz(note) * (2.0 ** (pitch_bend * 2.0 / 12.0))
    n  = int(SR * duration)
    wA, wB, wC, wD = blend_weights(nx, ny)
    weights = {'A': wA, 'B': wB, 'C': wC, 'D': wD}

    wave = np.zeros(n, dtype=np.float64)
    for oid, osc in oscs.items():
        w = weights[oid] * osc.level
        if w < 1e-5:
            continue
        f  = f0 * (2.0 ** (osc.det_st() / 12.0))
        nc = max(2, int(round(SR / f)))
        raw = gen_wave(osc.wave, nc, osc.extra)
        wave += np.resize(raw, n) * w  # np.resize loops seamlessly

    # LFO vibrato / tremolo
    vib_depth = lfo_depth + aftertouch * 0.03
    if vib_depth > 0.001:
        lt  = np.linspace(0, duration, n, endpoint=False)
        lfo = vib_depth * np.sin(2 * math.pi * lfo_rate * lt)
        wave *= 1.0 + lfo * 0.015   # ±1.5% głębokość vibrato

    # ADSR
    wave *= _adsr_env(n, *adsr)

    # Filter (z lekkim podbiciem od aftertouch)
    fc = min(flt_cut * (1.0 + aftertouch * 0.6), SR / 2.0 - 10.0)
    if fc < SR / 2.0 - 200.0:
        Q  = 0.707 + flt_res * 9.0
        wave = biquad_lp(wave, fc, Q)

    # Soft clip
    wave = np.tanh(wave * 1.5) / 1.5

    # Reverb
    if reverb > 0.01:
        wave = multi_echo_reverb(wave, reverb)

    # Velocity + normalizacja
    v  = vel / 127.0
    mx = np.max(np.abs(wave))
    if mx > 1e-9:
        wave /= mx * 1.05
    wave *= v
    return wave.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Menadżer głosów (polifonia przez kanały pygame)
# ═══════════════════════════════════════════════════════════════════
class VoiceManager:
    def __init__(self):
        self._active: dict = {}   # note → pygame.Channel
        self._lock   = threading.Lock()

    def note_on(self, note, vel, oscs, nx, ny,
                adsr, pb, flt_cut, flt_res,
                reverb, lfo_depth, lfo_rate, at, duration):
        if not PYGAME_OK:
            return
        self.note_off(note, 20)
        ch = self._alloc()
        if ch is None:
            return

        def _gen():
            wave = synthesize_voice(
                note, vel, oscs, nx, ny, adsr, pb,
                flt_cut, flt_res, reverb, lfo_depth, lfo_rate, at, duration)
            arr = (wave * 32767).astype(np.int16)
            try:
                snd = pygame.sndarray.make_sound(arr)
                with self._lock:
                    self._active[note] = ch
                ch.play(snd)
            except Exception as e:
                print(f"[Voice] {e}")

        threading.Thread(target=_gen, daemon=True).start()

    def note_off(self, note: int, fade_ms: int = 160):
        with self._lock:
            ch = self._active.pop(note, None)
        if ch and PYGAME_OK:
            ch.fadeout(fade_ms)

    def all_off(self, fade_ms: int = 80):
        with self._lock:
            chs = list(self._active.values())
            self._active.clear()
        for c in chs:
            if PYGAME_OK:
                c.fadeout(fade_ms)

    def _alloc(self):
        if not PYGAME_OK:
            return None
        for i in range(pygame.mixer.get_num_channels()):
            ch = pygame.mixer.Channel(i)
            if not ch.get_busy():
                return ch
        # Zabierz najstarszy kanał
        with self._lock:
            if self._active:
                k = next(iter(self._active))
                ch = self._active.pop(k)
                ch.stop()
                return ch
        return None


# ═══════════════════════════════════════════════════════════════════
#  MIDI Engine — pełny odbiór (Note, PB, CC all, AT, PC)
# ═══════════════════════════════════════════════════════════════════
class MIDIEngine:
    def __init__(self):
        self._in   = None
        self._idx  = -1
        self._name = ""
        self.on_note_on     = None   # (note, vel)
        self.on_note_off    = None   # (note,)
        self.on_pitch_bend  = None   # (val ∈ -1..+1)
        self.on_cc          = None   # (cc, val 0..127)
        self.on_aftertouch  = None   # (val 0..127)
        self.on_poly_at     = None   # (note, val)
        self.on_prog_change = None   # (prog 0..127)

    def list_ports(self) -> list:
        if not RTMIDI_OK:
            return []
        try:
            m = rtmidi.MidiIn()
            p = m.get_ports()
            del m
            return p
        except Exception:
            return []

    def open(self, idx: int) -> bool:
        if not RTMIDI_OK:
            return False
        self.close()
        try:
            self._in = rtmidi.MidiIn()
            self._in.open_port(idx)
            self._in.set_callback(self._cb)
            self._in.ignore_types(sysex=True, timing=True, active_sense=True)
            ports     = self.list_ports()
            self._name = ports[idx] if idx < len(ports) else f"Port {idx}"
            self._idx  = idx
            return True
        except Exception as e:
            print(f"MIDI open: {e}")
            self._in = None
            return False

    def close(self):
        if self._in:
            try:
                self._in.close_port()
            except Exception:
                pass
            self._in = None
        self._idx = -1; self._name = ""

    def _cb(self, event, _=None):
        msg, _ = event
        if not msg:
            return
        st = msg[0] & 0xF0
        if st == 0x90:           # Note On
            n, v = msg[1], msg[2]
            if v == 0:
                if self.on_note_off: self.on_note_off(n)
            else:
                if self.on_note_on:  self.on_note_on(n, v)
        elif st == 0x80:         # Note Off
            if self.on_note_off: self.on_note_off(msg[1])
        elif st == 0xA0:         # Poly Aftertouch
            if self.on_poly_at:  self.on_poly_at(msg[1], msg[2])
        elif st == 0xB0:         # CC
            if self.on_cc:       self.on_cc(msg[1], msg[2])
        elif st == 0xC0:         # Program Change
            if self.on_prog_change: self.on_prog_change(msg[1])
        elif st == 0xD0:         # Channel Aftertouch
            if self.on_aftertouch:  self.on_aftertouch(msg[1])
        elif st == 0xE0:         # Pitch Bend
            raw = ((msg[2] << 7) | msg[1]) - 8192
            if self.on_pitch_bend: self.on_pitch_bend(raw / 8192.0)

    @property
    def port_name(self): return self._name
    @property
    def connected(self):  return self._in is not None


# ═══════════════════════════════════════════════════════════════════
#  Dialog wyboru portu MIDI
# ═══════════════════════════════════════════════════════════════════
class MIDIPortDialog(Gtk.Dialog):
    def __init__(self, parent, ports):
        super().__init__(title="Wybór portu MIDI", parent=parent,
                         flags=Gtk.DialogFlags.MODAL)
        self.set_default_size(440, 260)
        self.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                         "_Połącz",        Gtk.ResponseType.OK)
        self.selected_idx = -1
        box = self.get_content_area()
        box.set_spacing(6); box.set_border_width(10)

        if not RTMIDI_OK:
            box.add(Gtk.Label(label="Brak python-rtmidi. Zainstaluj:\n  pip install python-rtmidi"))
            return
        if not ports:
            box.add(Gtk.Label(label="Nie znaleziono urządzeń MIDI."))
            return

        box.add(Gtk.Label(label="Dostępne porty MIDI:"))
        ls = Gtk.ListStore(int, str)
        for i, p in enumerate(ports):
            ls.append([i, p])
        tv = Gtk.TreeView(model=ls)
        tv.append_column(Gtk.TreeViewColumn("#",    Gtk.CellRendererText(), text=0))
        tv.append_column(Gtk.TreeViewColumn("Port", Gtk.CellRendererText(), text=1))
        tv.connect("row-activated", lambda *a: self.response(Gtk.ResponseType.OK))
        sel = tv.get_selection()
        sel.set_mode(Gtk.SelectionMode.SINGLE)
        sel.connect("changed", lambda s: self._sel(s))
        if ports:
            sel.select_path(Gtk.TreePath.new_first())
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        sw.set_size_request(-1, 150)
        sw.add(tv); box.add(sw)
        box.show_all()

    def _sel(self, sel):
        model, it = sel.get_selected()
        if it:
            self.selected_idx = model[it][0]


# ═══════════════════════════════════════════════════════════════════
#  Auto-ruch joysticka
# ═══════════════════════════════════════════════════════════════════
class AutoMove:
    RANDOM    = "random"
    LISSAJOUS = "lissajous"
    SPIRAL    = "spiral"

    def __init__(self, mode=RANDOM, speed=0.008):
        self.mode  = mode
        self.speed = speed
        self.x = 0.0; self.y = 0.0
        self.vx = random.uniform(-0.01, 0.01)
        self.vy = random.uniform(-0.01, 0.01)
        self.t  = 0.0
        self.fa = random.uniform(.8, 1.6); self.fb = random.uniform(.9, 2.1)
        self.pa = random.uniform(0, math.pi * 2)
        self.pb = random.uniform(0, math.pi * 2)
        self.sr = 0.1; self.sd = 1

    def step(self):
        self.t += self.speed
        if self.mode == self.LISSAJOUS:
            self.fa = max(.5, min(3., self.fa + random.uniform(-.0003, .0003)))
            self.fb = max(.5, min(3., self.fb + random.uniform(-.0003, .0003)))
            self.x  = math.sin(self.fa * self.t + self.pa)
            self.y  = math.sin(self.fb * self.t + self.pb)
        elif self.mode == self.SPIRAL:
            self.sr += self.sd * 0.003
            if self.sr > 0.95: self.sd = -1
            if self.sr < 0.05: self.sd =  1
            self.x = self.sr * math.sin(self.t * 1.5)
            self.y = self.sr * math.cos(self.t * 1.0)
        else:  # random walk
            self.vx = max(-.03, min(.03, self.vx + random.uniform(-.0012, .0012)))
            self.vy = max(-.03, min(.03, self.vy + random.uniform(-.0012, .0012)))
            self.x += self.vx; self.y += self.vy
            r = math.sqrt(self.x * self.x + self.y * self.y)
            if r > 0.95:
                self.x *= .88; self.y *= .88
                self.vx *= -.5; self.vy *= -.5
        return self.x, self.y


# ═══════════════════════════════════════════════════════════════════
#  Vector Envelope — animowana sekwencja kroków XY
# ═══════════════════════════════════════════════════════════════════
class VectorEnvelope:
    def __init__(self):
        self.steps:  list = []   # [{'x':float,'y':float,'t':float}, ...]
        self.loop        = True
        self._running    = False
        self._idx        = 0
        self._elapsed    = 0.0
        self._cb         = None  # fn(x, y)

    def set_callback(self, fn):
        self._cb = fn

    def start(self):
        if not self.steps: return
        self._running = True; self._idx = 0; self._elapsed = 0.0

    def stop(self):
        self._running = False

    @property
    def running(self): return self._running

    def tick(self, dt: float):
        if not self._running or len(self.steps) < 2:
            return None
        cur  = self.steps[self._idx]
        nxt  = self.steps[(self._idx + 1) % len(self.steps)]
        self._elapsed += dt
        frac = min(1.0, self._elapsed / max(cur['t'], .001))
        nx   = cur['x'] + (nxt['x'] - cur['x']) * frac
        ny   = cur['y'] + (nxt['y'] - cur['y']) * frac
        if self._elapsed >= cur['t']:
            self._elapsed = 0.0
            self._idx     = (self._idx + 1) % len(self.steps)
            if self._idx == 0 and not self.loop:
                self._running = False
        if self._cb:
            self._cb(nx, ny)
        return nx, ny


# ═══════════════════════════════════════════════════════════════════
#  Widget pada XY — 4-narożny joystick (styl Prophet VS)
# ═══════════════════════════════════════════════════════════════════
class VectorPadWidget(Gtk.DrawingArea):
    """
    Pad z A(lewa-góra) B(prawa-góra) C(prawa-dół) D(lewa-dół).
    LPM/przeciągnij = ruch joysticka. PPM = powrót do centrum.
    """
    OSC_COLORS = [
        (.20, .65, 1.00),  # A — niebieski
        (.15, .90, .45),   # B — zielony
        (1.00, .55, .15),  # C — pomarańczowy
        (.90, .20, .70),   # D — fioletowo-różowy
    ]
    CORNERS = [(-1, +1), (+1, +1), (+1, -1), (-1, -1)]  # A,B,C,D
    LABELS  = ['A', 'B', 'C', 'D']

    def __init__(self, on_change=None):
        super().__init__()
        self.on_change = on_change
        self.nx = 0.0; self.ny = 0.0
        self.dragging   = False
        self.trail: list = []
        self._MAX_TRAIL  = 140
        mask = (Gdk.EventMask.BUTTON_PRESS_MASK |
                Gdk.EventMask.BUTTON_RELEASE_MASK |
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.set_events(mask)
        self.connect("draw",                 self._draw)
        self.connect("button-press-event",   self._press)
        self.connect("button-release-event", self._release)
        self.connect("motion-notify-event",  self._motion)

    def set_pos(self, nx: float, ny: float, notify: bool = True):
        self.nx = max(-1., min(1., float(nx)))
        self.ny = max(-1., min(1., float(ny)))
        self.trail.append((self.nx, self.ny))
        if len(self.trail) > self._MAX_TRAIL:
            self.trail.pop(0)
        self.queue_draw()
        if notify and self.on_change:
            self.on_change(self.nx, self.ny)

    # ── układ pikseli ─────────────────────────────────────────────
    def _pad_rect(self):
        W = self.get_allocated_width()
        H = self.get_allocated_height()
        p = 32
        return p, p, W - 2*p, H - 2*p - 20

    def _n2px(self, nx, ny):
        px, py, pw, ph = self._pad_rect()
        return px + (nx + 1) / 2 * pw,  py + (1 - ny) / 2 * ph

    def _px2n(self, x, y):
        px, py, pw, ph = self._pad_rect()
        return (max(-1., min(1., (x - px) / pw * 2 - 1)),
                max(-1., min(1., 1 - (y - py) / ph * 2)))

    # ── rysowanie ─────────────────────────────────────────────────
    def _draw(self, widget, cr):
        W = widget.get_allocated_width()
        H = widget.get_allocated_height()
        px, py, pw, ph = self._pad_rect()

        cr.set_source_rgb(.06, .06, .09); cr.paint()

        # Gradient tła — 4 trójkąty od centrum do rogów
        cx = px + pw / 2;  cy = py + ph / 2
        cpts = [self._n2px(nx, ny) for nx, ny in self.CORNERS]
        for i in range(4):
            x0, y0 = cpts[i]; x1, y1 = cpts[(i + 1) % 4]
            r, g, b = self.OSC_COLORS[i]
            cr.set_source_rgba(r, g, b, .20)
            cr.move_to(cx, cy); cr.line_to(x0, y0); cr.line_to(x1, y1)
            cr.close_path(); cr.fill()

        # Obramowanie
        cr.set_source_rgba(.42, .42, .55, .9); cr.set_line_width(1.5)
        cr.rectangle(px, py, pw, ph); cr.stroke()

        # Siatka 4×4
        cr.set_source_rgba(.20, .20, .26, .55); cr.set_line_width(.5)
        for i in range(1, 4):
            xg = px + i * pw / 4; yg = py + i * ph / 4
            cr.move_to(xg, py); cr.line_to(xg, py + ph); cr.stroke()
            cr.move_to(px, yg); cr.line_to(px + pw, yg); cr.stroke()

        # Osie środkowe
        cr.set_source_rgba(.38, .38, .50, .80); cr.set_line_width(1.)
        cr.move_to(cx, py);      cr.line_to(cx, py + ph); cr.stroke()
        cr.move_to(px, cy);      cr.line_to(px + pw, cy); cr.stroke()

        # Etykiety narożne A B C D
        cr.set_font_size(14)
        lbl_px = [(px + 5, py + 18), (px + pw - 19, py + 18),
                  (px + pw - 19, py + ph - 5), (px + 5, py + ph - 5)]
        for i, (lx, ly) in enumerate(lbl_px):
            r, g, b = self.OSC_COLORS[i]
            cr.set_source_rgba(r, g, b, .92)
            cr.move_to(lx, ly); cr.show_text(self.LABELS[i])

        # Ślad ruchu
        if len(self.trail) > 1:
            for k in range(len(self.trail) - 1):
                frac = (k + 1) / len(self.trail)
                x0, y0 = self._n2px(*self.trail[k])
                x1, y1 = self._n2px(*self.trail[k + 1])
                cr.set_source_rgba(.75, .75, 1., frac * .55)
                cr.set_line_width(1.4)
                cr.move_to(x0, y0); cr.line_to(x1, y1); cr.stroke()

        # Paski wag (linie od rogu ku kursorowi)
        wA, wB, wC, wD = blend_weights(self.nx, self.ny)
        jx, jy = self._n2px(self.nx, self.ny)
        for i, (ww, (cpx, cpy)) in enumerate(zip([wA, wB, wC, wD], cpts)):
            if ww < 0.01:
                continue
            r, g, b = self.OSC_COLORS[i]
            cr.set_source_rgba(r, g, b, .5 + ww * .4)
            cr.set_line_width(1.5 + ww * 3.5)
            cr.move_to(cpx, cpy); cr.line_to(jx, jy); cr.stroke()

        # Kursor joysticka
        bc = [sum(self.OSC_COLORS[i][c] * w
                  for i, w in enumerate([wA, wB, wC, wD]))
              for c in range(3)]
        cr.set_source_rgba(0, 0, 0, .55)
        cr.arc(jx + 2, jy + 2, 16, 0, 6.28); cr.fill()
        cr.set_source_rgba(bc[0], bc[1], bc[2], .97)
        cr.arc(jx, jy, 16, 0, 6.28); cr.fill()
        cr.set_source_rgb(1, 1, 1); cr.set_line_width(2.)
        cr.arc(jx, jy, 16, 0, 6.28); cr.stroke()
        cr.set_source_rgba(0, 0, 0, .80); cr.set_line_width(2.)
        cr.move_to(jx - 10, jy); cr.line_to(jx + 10, jy); cr.stroke()
        cr.move_to(jx, jy - 10); cr.line_to(jx, jy + 10); cr.stroke()

        # Pasek informacyjny
        cr.set_font_size(9); cr.set_source_rgba(.7, .7, .85, .9)
        wA, wB, wC, wD = blend_weights(self.nx, self.ny)
        cr.move_to(px, py + ph + 15)
        cr.show_text(f"X={self.nx:+.2f}  Y={self.ny:+.2f}  "
                     f"A={wA:.0%} B={wB:.0%} C={wC:.0%} D={wD:.0%}")

    # ── mysz ──────────────────────────────────────────────────────
    def _press(self, w, e):
        if e.button == 3:
            self.set_pos(0., 0.); return
        if e.button == 1:
            self.dragging = True
            nx, ny = self._px2n(e.x, e.y)
            self.set_pos(nx, ny)

    def _release(self, w, e):
        if e.button == 1: self.dragging = False

    def _motion(self, w, e):
        if self.dragging:
            nx, ny = self._px2n(e.x, e.y)
            self.set_pos(nx, ny)


# ═══════════════════════════════════════════════════════════════════
#  Panel jednego oscylatora (A / B / C / D)
# ═══════════════════════════════════════════════════════════════════
class OscPanel(Gtk.Frame):
    def __init__(self, label: str, color: tuple, on_change=None):
        r, g, b = color
        super().__init__()
        lw = Gtk.Label()
        lw.set_markup(
            f'<b><span foreground="#{int(r*255):02x}{int(g*255):02x}'
            f'{int(b*255):02x}"> OSC {label} </span></b>')
        self.set_label_widget(lw); self.set_border_width(3)
        self._on_change = on_change
        self._r, self._g, self._b = r, g, b
        self.s = OscSettings()

        vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        vb.set_border_width(4); self.add(vb)

        # Wybór fali
        hb = Gtk.Box(spacing=3)
        hb.pack_start(Gtk.Label(label="Fala:"), False, False, 0)
        self.wave_cb = Gtk.ComboBoxText()
        for nm in WAVE_NAMES:
            self.wave_cb.append_text(nm)
        self.wave_cb.set_active(0)
        self.wave_cb.connect("changed", self._on_wave)
        hb.pack_start(self.wave_cb, True, True, 0)
        vb.pack_start(hb, False, False, 0)

        # Strojenie
        hb2 = Gtk.Box(spacing=3)
        hb2.pack_start(Gtk.Label(label="ST:"), False, False, 0)
        self.coarse = Gtk.SpinButton()
        self.coarse.set_range(-24, 24); self.coarse.set_value(0)
        self.coarse.set_increments(1, 12); self.coarse.set_width_chars(4)
        self.coarse.connect("value-changed",
                            lambda w: self._upd("coarse", int(w.get_value())))
        hb2.pack_start(self.coarse, False, False, 0)
        hb2.pack_start(Gtk.Label(label=" ¢:"), False, False, 0)
        self.fine = Gtk.SpinButton()
        self.fine.set_range(-100, 100); self.fine.set_value(0)
        self.fine.set_increments(1, 10); self.fine.set_width_chars(5)
        self.fine.connect("value-changed",
                          lambda w: self._upd("fine", float(w.get_value())))
        hb2.pack_start(self.fine, False, False, 0)
        vb.pack_start(hb2, False, False, 0)

        # Poziom
        hb3 = Gtk.Box(spacing=3)
        hb3.pack_start(Gtk.Label(label="Lev:"), False, False, 0)
        self.level_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.level_sc.set_range(0, 1); self.level_sc.set_value(1)
        self.level_sc.set_digits(2); self.level_sc.set_draw_value(True)
        self.level_sc.connect("value-changed",
                              lambda w: self._upd("level", w.get_value()))
        hb3.pack_start(self.level_sc, True, True, 0)
        vb.pack_start(hb3, False, False, 0)

        # Extra (parametr specjalny fali)
        hb4 = Gtk.Box(spacing=3)
        hb4.pack_start(Gtk.Label(label="Ext:"), False, False, 0)
        self.extra_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.extra_sc.set_range(0, 1); self.extra_sc.set_value(0.5)
        self.extra_sc.set_digits(2); self.extra_sc.set_draw_value(True)
        self.extra_sc.set_tooltip_text(
            "Param. dodatkowy: FM index, głębokość harmonik, itp.")
        self.extra_sc.connect("value-changed",
                              lambda w: self._upd("extra", w.get_value()))
        hb4.pack_start(self.extra_sc, True, True, 0)
        vb.pack_start(hb4, False, False, 0)

        # Mini-podgląd fali
        self.preview = Gtk.DrawingArea()
        self.preview.set_size_request(-1, 38)
        self.preview.connect("draw", self._draw_preview)
        vb.pack_start(self.preview, False, False, 0)

    def _on_wave(self, cb):
        idx = cb.get_active()
        if 0 <= idx < len(WAVE_IDS):
            self.s.wave = WAVE_IDS[idx]
        self.preview.queue_draw()
        if self._on_change: self._on_change()

    def _upd(self, key, val):
        setattr(self.s, key, val)
        if key == 'extra':
            self.preview.queue_draw()
        if self._on_change: self._on_change()

    def _draw_preview(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        cr.set_source_rgb(.06, .06, .09); cr.paint()
        n    = min(w, 256)
        wave = gen_wave(self.s.wave, n, self.s.extra)
        cr.set_source_rgba(self._r, self._g, self._b, .9)
        cr.set_line_width(1.3)
        for i, v in enumerate(wave):
            x = i * w / n;  y = h / 2 - v * (h / 2 - 3)
            if i == 0: cr.move_to(x, y)
            else:      cr.line_to(x, y)
        cr.stroke()
        cr.set_source_rgba(.25, .25, .32, .5); cr.set_line_width(.5)
        cr.move_to(0, h / 2); cr.line_to(w, h / 2); cr.stroke()

    def set_from_dict(self, d: dict):
        """Wczytuje parametry z presetu."""
        self.s.wave   = d.get('wave', 'sine')
        self.s.coarse = d.get('coarse', 0)
        self.s.fine   = d.get('fine', 0.0)
        self.s.level  = d.get('level', 1.0)
        self.s.extra  = d.get('extra', 0.5)
        if self.s.wave in WAVE_IDS:
            self.wave_cb.set_active(WAVE_IDS.index(self.s.wave))
        self.coarse.set_value(self.s.coarse)
        self.fine.set_value(self.s.fine)
        self.level_sc.set_value(self.s.level)
        self.extra_sc.set_value(self.s.extra)
        self.preview.queue_draw()


# ═══════════════════════════════════════════════════════════════════
#  Presety (8 gotowych brzmień)
# ═══════════════════════════════════════════════════════════════════
PRESETS = [
    {"name": "Init",
     "A": {"wave":"sine",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "B": {"wave":"sine",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "C": {"wave":"sine",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "D": {"wave":"sine",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "adsr":[.01,.10,.80,.30], "flt":[18000.,.00], "rev":.00, "lfo":[5.,.00]},

    {"name": "Poly Pad",
     "A": {"wave":"saw_dn", "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "B": {"wave":"saw_up", "coarse": 7, "fine": 5,  "level":.80, "extra":.5},
     "C": {"wave":"tri",    "coarse":12, "fine": 0,  "level":.70, "extra":.5},
     "D": {"wave":"odd",    "coarse":-5, "fine":-5,  "level":.60, "extra":.6},
     "adsr":[.40,.20,.85,.80], "flt":[5500.,.25], "rev":.40, "lfo":[4.,.00]},

    {"name": "Deep Bass",
     "A": {"wave":"saw_dn", "coarse":-12,"fine": 0,  "level":1.0, "extra":.5},
     "B": {"wave":"square", "coarse": 0, "fine": 0,  "level":.70, "extra":.5},
     "C": {"wave":"fm21",   "coarse": 0, "fine": 3,  "level":.50, "extra":.4},
     "D": {"wave":"odd",    "coarse":-12,"fine":-5,  "level":.40, "extra":.5},
     "adsr":[.01,.15,.70,.15], "flt":[2800.,.40], "rev":.10, "lfo":[3.,.00]},

    {"name": "Bright Lead",
     "A": {"wave":"saw_dn", "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "B": {"wave":"fm21",   "coarse": 0, "fine": 7,  "level":.80, "extra":.7},
     "C": {"wave":"odd",    "coarse": 7, "fine": 0,  "level":.50, "extra":.6},
     "D": {"wave":"saw_up", "coarse":-12,"fine":-7,  "level":.30, "extra":.5},
     "adsr":[.01,.08,.75,.12], "flt":[12000.,.30], "rev":.15, "lfo":[5.,.00]},

    {"name": "Pluck",
     "A": {"wave":"comb",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.8},
     "B": {"wave":"odd",    "coarse": 0, "fine": 0,  "level":.70, "extra":.5},
     "C": {"wave":"sine",   "coarse":12, "fine": 0,  "level":.40, "extra":.5},
     "D": {"wave":"comb",   "coarse": 0, "fine":-5,  "level":.50, "extra":.6},
     "adsr":[.003,.40,.00,.08], "flt":[9000.,.20], "rev":.20, "lfo":[6.,.00]},

    {"name": "Choir",
     "A": {"wave":"fa",     "coarse": 0, "fine": 0,  "level":1.0, "extra":.6},
     "B": {"wave":"fe",     "coarse": 0, "fine": 7,  "level":.90, "extra":.6},
     "C": {"wave":"fo",     "coarse":-5, "fine": 0,  "level":.80, "extra":.6},
     "D": {"wave":"fa",     "coarse": 7, "fine":-7,  "level":.60, "extra":.7},
     "adsr":[.30,.15,.85,.40], "flt":[7000.,.25], "rev":.50, "lfo":[4.5,.00]},

    {"name": "Bell Field",
     "A": {"wave":"bell",   "coarse": 0, "fine": 0,  "level":1.0, "extra":.6},
     "B": {"wave":"glass",  "coarse":12, "fine": 0,  "level":.80, "extra":.5},
     "C": {"wave":"bell",   "coarse": 7, "fine": 5,  "level":.70, "extra":.8},
     "D": {"wave":"glass",  "coarse":-5, "fine":-5,  "level":.50, "extra":.4},
     "adsr":[.003,.80,.00,.60], "flt":[15000.,.15], "rev":.45, "lfo":[5.,.00]},

    {"name": "Ambient Glass",
     "A": {"wave":"glass",  "coarse": 0, "fine": 0,  "level":1.0, "extra":.5},
     "B": {"wave":"organ",  "coarse":12, "fine": 0,  "level":.70, "extra":.6},
     "C": {"wave":"metal",  "coarse":-5, "fine": 5,  "level":.50, "extra":.4},
     "D": {"wave":"fib",    "coarse": 0, "fine":-7,  "level":.60, "extra":.7},
     "adsr":[.60,.30,.80,.90], "flt":[8000.,.35], "rev":.65, "lfo":[3.,.00]},
]


# ═══════════════════════════════════════════════════════════════════
#  16-krokowy sekwencer
# ═══════════════════════════════════════════════════════════════════
NOTE_NAMES_ALL = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_note_name(n: int) -> str:
    return NOTE_NAMES_ALL[n % 12] + str(n // 12 - 1)

def midi_from_name(name: str) -> int:
    """'C4' → 60, 'A#3' → 46, itd."""
    try:
        if len(name) >= 2 and name[-2] == '#':
            nm = name[:-2]; oct_ = int(name[-1])
        else:
            nm = name[:-1]; oct_ = int(name[-1])
        return NOTE_NAMES_ALL.index(nm) + (oct_ + 1) * 12
    except Exception:
        return 60


# Gotowe wzorce dla sekwencera
SEQ_PATTERNS = {
    "Pusta":         [{'note':60,'vel':100,'active':False,'gate':.8,'nx':0.,'ny':0.,'len':1}]*16,
    "C-dur akordy":  [{'note':n,'vel':v,'active':True,'gate':.8,'nx':0.,'ny':0.,'len':1}
                      for n, v in zip(
                          [60,64,67,72, 60,64,67,72, 62,65,69,72, 60,65,69,72],
                          [110,90,90,80]*4)],
    "Pentatonika":   [{'note':n,'vel':100,'active':True,'gate':.7,'nx':0.,'ny':0.,'len':1}
                      for n in [60,62,64,67,69,72,74,76,72,69,67,64,62,60,62,64]],
    "Basowa oktawa": [{'note':n,'vel':v,'active':True,'gate':.9,'nx':-1.,'ny':-1.,'len':l}
                      for n,v,l in [(36,120,1),(36,80,1),(36,90,1),(36,70,1),
                                    (41,120,1),(41,80,1),(43,90,1),(43,70,1),
                                    (36,120,2),(0,0,1),(38,100,1),(0,0,1),
                                    (36,110,1),(36,75,1),(36,85,1),(36,65,1)]],
    "Techno 16th":   [{'note':36,'vel':v,'active':True,'gate':.5,'nx':-1.,'ny':-1.,'len':1}
                      for v in [120,70,90,70, 110,70,90,100, 120,70,90,70, 110,70,100,80]],
}


class SeqStep:
    """Jeden krok sekwencera z pełnym zestawem parametrów."""
    __slots__ = ('active', 'note', 'velocity', 'gate', 'nx', 'ny', 'length')

    def __init__(self, note: int = 60, velocity: int = 100):
        self.active   = True
        self.note     = note
        self.velocity = velocity
        self.gate     = 0.80    # długość nuty (ułamek długości kroku)
        self.nx       = 0.0     # X joysticka dla tego kroku
        self.ny       = 0.0     # Y joysticka dla tego kroku
        self.length   = 1       # mnożnik długości: 1=1/16, 2=1/8, 4=1/4

    def from_dict(self, d: dict):
        self.active   = d.get('active',   True)
        self.note     = d.get('note',      60)
        self.velocity = d.get('vel',       100)
        self.gate     = d.get('gate',      .8)
        self.nx       = d.get('nx',        0.)
        self.ny       = d.get('ny',        0.)
        self.length   = d.get('len',       1)


class Sequencer:
    """
    16-krokowy sekwencer z BPM, swing, zmienną długością kroków,
    per-step XY (pozycja joysticka → timbral blend).
    """
    N_STEPS = 16

    def __init__(self):
        # Wypełnij krokami ze skalą C-dur
        c_maj = [60,62,64,65,67,69,71,72,74,72,71,69,67,65,64,62]
        self.steps = [SeqStep(note=n, velocity=100) for n in c_maj]
        self.bpm        = 120.0
        self.active_len = 16      # ile kroków w pętli (1-16)
        self.swing      = 0.0     # 0=bez swing, 1=max swing
        self.running    = False
        self.current    = 0       # bieżący krok
        self._elapsed   = 0.0     # czas w bieżącym kroku [s]
        self._pending_offs: list = []  # [(czas_pozostały, note), ...]

        # Callbacki
        self.on_step_on  = None   # fn(step_idx, step: SeqStep)
        self.on_step_off = None   # fn(note)

    def step_dur(self, idx: int) -> float:
        """Czas trwania kroku w sekundach."""
        base = 60.0 / self.bpm / 4.0  # 1/16 nuty
        if self.swing > 0.01:
            if idx % 2 == 1:
                base *= (1.0 + self.swing * 0.35)   # nieparzyste dłuższe
            else:
                base *= (1.0 - self.swing * 0.20)   # parzyste krótsze
        return base * self.steps[idx].length

    def start(self):
        self.running = True; self.current = 0; self._elapsed = 0.0
        self._pending_offs.clear()

    def stop(self):
        self.running = False
        for _, note in self._pending_offs:
            if self.on_step_off: self.on_step_off(note)
        self._pending_offs.clear()

    def tick(self, dt: float) -> list:
        """Wołany z timera. Zwraca listę kroków, które właśnie wystrzeliły."""
        if not self.running:
            return []

        # Gate-off dla minionych nut
        alive = []
        for (t_rem, note) in self._pending_offs:
            t_rem -= dt
            if t_rem <= 0:
                if self.on_step_off: self.on_step_off(note)
            else:
                alive.append((t_rem, note))
        self._pending_offs = alive

        fired = []
        self._elapsed += dt
        # Obsłuż kilka kroków w jednym ticku (przy niskich BPM lub lag-spike)
        for _ in range(self.active_len):
            dur = self.step_dur(self.current)
            if self._elapsed < dur:
                break
            self._elapsed -= dur
            step = self.steps[self.current]
            if step.active:
                if self.on_step_on:
                    self.on_step_on(self.current, step)
                gate_t = dur * step.gate
                self._pending_offs.append((gate_t, step.note))
            fired.append(self.current)
            self.current = (self.current + 1) % self.active_len
        return fired

    def load_pattern(self, name: str):
        """Ładuje gotowy wzorzec."""
        pat = SEQ_PATTERNS.get(name)
        if not pat:
            return
        for i, d in enumerate(pat[:self.N_STEPS]):
            self.steps[i].from_dict(d)

    def randomize(self, scale=None):
        """Losuje nuty ze skali (domyślnie pentatonika C)."""
        if scale is None:
            scale = [48, 50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74]
        for step in self.steps:
            step.note     = random.choice(scale)
            step.velocity = random.randint(70, 127)
            step.active   = random.random() > 0.2


# ── Kolory kroków sekwencera ────────────────────────────────────────
def _seq_step_color(step: SeqStep, is_current: bool, is_sel: bool):
    """Zwraca (r, g, b, alpha) dla kroku."""
    wA, wB, wC, wD = blend_weights(step.nx, step.ny)
    bc = [sum(VectorPadWidget.OSC_COLORS[k][c] * ww
              for k, ww in enumerate([wA, wB, wC, wD]))
          for c in range(3)]
    alpha = 0.82 if step.active else 0.16
    return bc[0], bc[1], bc[2], alpha


class SeqStepWidget(Gtk.DrawingArea):
    """
    Rysuje 16 kroków sekwencera.
    LPM = zaznacz do edycji
    PPM = toggle active/inactive
    Scroll = zmień nutę zaznaczonego kroku
    """
    STEP_W = 58
    STEP_H = 80

    def __init__(self, seq: Sequencer, on_select=None):
        super().__init__()
        self.seq       = seq
        self.on_select = on_select
        self.selected  = 0
        self.set_size_request(self.STEP_W * seq.N_STEPS, self.STEP_H)
        mask = (Gdk.EventMask.BUTTON_PRESS_MASK |
                Gdk.EventMask.SCROLL_MASK       |
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.set_events(mask)
        self.connect("draw",                self._draw)
        self.connect("button-press-event",  self._press)
        self.connect("scroll-event",        self._scroll)

    def _idx(self, event_x) -> int:
        return max(0, min(self.seq.N_STEPS - 1, int(event_x / self.STEP_W)))

    def _draw(self, widget, cr):
        W = widget.get_allocated_width()
        H = widget.get_allocated_height()
        cr.set_source_rgb(.05, .05, .08); cr.paint()

        N = self.seq.N_STEPS
        sw = W / N    # auto-width per step (may differ from STEP_W if window resized)

        for i in range(N):
            step = self.seq.steps[i]
            x    = i * sw
            w    = sw - 2
            h    = H - 2

            r, g, b, alpha = _seq_step_color(step, i == self.seq.current,
                                              i == self.selected)

            # ── tło kroku ─────────────────────────────────────────
            cr.set_source_rgba(r, g, b, alpha)
            cr.rectangle(x + 1, 1, w, h); cr.fill()

            # ── pasek velocity (dół) ──────────────────────────────
            if step.active:
                bar_h = max(2, int((h - 48) * step.velocity / 127))
                cr.set_source_rgba(1, 1, 1, .35)
                cr.rectangle(x + 2, h - bar_h + 1, w - 4, bar_h)
                cr.fill()

            # ── ramka ─────────────────────────────────────────────
            if i == self.seq.current and self.seq.running:
                # bieżący grający krok — jasna żółta ramka
                cr.set_source_rgba(1., .96, .25, 1.)
                cr.set_line_width(3.5)
                cr.rectangle(x + 1, 1, w, h); cr.stroke()
            elif i == self.selected:
                # zaznaczony do edycji — przerywana cyjanowa
                cr.set_source_rgba(.35, .92, 1., .9)
                cr.set_line_width(2); cr.set_dash([4, 3])
                cr.rectangle(x + 1, 1, w, h); cr.stroke()
                cr.set_dash([])
            else:
                cr.set_source_rgba(.22, .22, .28, .8)
                cr.set_line_width(1)
                cr.rectangle(x + 1, 1, w, h); cr.stroke()

            # ── numer kroku ───────────────────────────────────────
            cr.set_font_size(8)
            cr.set_source_rgba(.6, .6, .7, .8)
            cr.move_to(x + 3, 11); cr.show_text(str(i + 1))

            # ── długość kroku ─────────────────────────────────────
            if step.length > 1:
                cr.set_font_size(7); cr.set_source_rgba(1., .9, .4, .9)
                cr.move_to(x + w - 12, 11)
                cr.show_text(f"×{step.length}")

            # ── nazwa nuty ────────────────────────────────────────
            nm = midi_note_name(step.note)
            cr.set_font_size(12 if step.active else 10)
            cr.set_source_rgba(1, 1, 1, .92 if step.active else .25)
            te = cr.text_extents(nm)
            cr.move_to(x + sw / 2 - te.width / 2, 34)
            cr.show_text(nm)

            # ── gate linia ────────────────────────────────────────
            if step.active:
                gate_w = (w - 4) * step.gate
                cr.set_source_rgba(1, 1, 1, .4)
                cr.set_line_width(1.5)
                cr.move_to(x + 2, 42); cr.line_to(x + 2 + gate_w, 42)
                cr.stroke()

            # ── mini XY dot ───────────────────────────────────────
            dot_x = x + 4 + (step.nx + 1) / 2 * (sw - 8)
            dot_y = 48 + (1 - step.ny) / 2 * 8
            cr.set_source_rgba(1, 1, .4, .9 if step.active else .2)
            cr.arc(dot_x, dot_y, 2.5, 0, 6.28); cr.fill()

        # ── linia końca aktywnej długości ─────────────────────────
        if self.seq.active_len < N:
            x_end = self.seq.active_len * sw
            cr.set_source_rgba(1., .35, .35, .85)
            cr.set_line_width(2.5)
            cr.move_to(x_end, 0); cr.line_to(x_end, H)
            cr.set_dash([5, 3]); cr.stroke(); cr.set_dash([])

    def _press(self, widget, event):
        i = self._idx(event.x)
        if event.button == 1:
            self.selected = i
            if self.on_select: self.on_select(i)
            self.queue_draw()
        elif event.button == 3:
            self.seq.steps[i].active = not self.seq.steps[i].active
            self.queue_draw()

    def _scroll(self, widget, event):
        """Scroll = zmień nutę zaznaczonego kroku."""
        step = self.seq.steps[self.selected]
        if event.direction == Gdk.ScrollDirection.UP:
            step.note = min(127, step.note + 1)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            step.note = max(0,   step.note - 1)
        self.queue_draw()


# ═══════════════════════════════════════════════════════════════════
#  Główne okno (Vector Synthesizer)
# ═══════════════════════════════════════════════════════════════════
def _sep():
    return Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)


class ChaosPadWindow(Gtk.Window):
    """
    Standalone syntezer wektorowy  lub  sub-okno ED_Waves.
    callback(wave: np.ndarray, params: dict) — wywoływany przy każdej zmianie.
    """
    TICK_MS = TICK_MS

    def __init__(self, callback=None):
        Gtk.Window.__init__(self, title="Vector Synthesizer — Prophet VS")
        self.set_default_size(1300, 980)
        self.connect("destroy", self._on_destroy)

        self.wave_callback = callback   # dla ED_Waves

        # ── silniki ──────────────────────────────────────────────
        self.midi      = MIDIEngine()
        self.voices    = VoiceManager()
        self.vec_env   = VectorEnvelope()
        self.auto_move = AutoMove()
        self.seq       = Sequencer()

        # ── sekwencer: tryb PLAY/PROG ─────────────────────────────
        self._seq_prog_mode = False  # False=PLAY, True=PROG (klawiatura → krok)
        self._seq_auto_adv  = True   # auto-advance po zaprogramowaniu kroku
        self._seq_edit_step = 0      # indeks kroku edytowanego w trybie PROG
        self._seq_link_pad  = True   # krok dziedziczy XY pada podczas wstawiania

        # ── stan globalny ─────────────────────────────────────────
        self._pb       = 0.0    # pitch bend -1..+1
        self._mod      = 0.0    # CC1 mod wheel 0..1
        self._vol      = 1.0    # CC7
        self._expr     = 1.0    # CC11
        self._sustain  = False  # CC64
        self._at       = 0.0    # aftertouch 0..1
        self._active_notes: set = set()

        self._adsr     = [.01, .1, .8, .3]
        self._flt_cut  = 18000.0
        self._flt_res  = 0.0
        self._reverb   = 0.0
        self._lfo_rate = 5.0
        self._lfo_dep  = 0.0
        self._duration = 3.5

        # ── MIDI callbacki ────────────────────────────────────────
        self.midi.on_note_on     = self._midi_on
        self.midi.on_note_off    = self._midi_off
        self.midi.on_pitch_bend  = self._midi_pb
        self.midi.on_cc          = self._midi_cc
        self.midi.on_aftertouch  = self._midi_at
        self.midi.on_poly_at     = self._midi_poly_at
        self.midi.on_prog_change = self._midi_pc

        self.vec_env.set_callback(self._vec_env_pos_cb)

        # ── sekwencer: callbacki ──────────────────────────────────
        self.seq.on_step_on  = self._seq_step_on
        self.seq.on_step_off = self._seq_step_off

        self._timer_id = None
        self._build_ui()
        self._apply_preset(PRESETS[0])
        self._timer_id = GLib.timeout_add(self.TICK_MS, self._tick)

    # ══════════════════════════════════════════════════════════════
    #  Budowanie interfejsu
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        root.set_border_width(3); self.add(root)

        # ── pasek górny ──────────────────────────────────────────
        tb = Gtk.Box(spacing=5); tb.set_border_width(3)
        root.pack_start(tb, False, False, 0)

        self.midi_btn = Gtk.Button(label="🎹 MIDI")
        self.midi_btn.connect("clicked", self._open_midi_dlg)
        tb.pack_start(self.midi_btn, False, False, 0)
        self.midi_lbl = Gtk.Label()
        self.midi_lbl.set_markup('<span foreground="red">⬤ OFF</span>')
        tb.pack_start(self.midi_lbl, False, False, 2)
        tb.pack_start(_sep(), False, False, 4)

        tb.pack_start(Gtk.Label(label="Preset:"), False, False, 0)
        self.preset_cb = Gtk.ComboBoxText()
        for p in PRESETS:
            self.preset_cb.append_text(p['name'])
        self.preset_cb.set_active(0)
        self.preset_cb.connect(
            "changed", lambda w: self._apply_preset(PRESETS[w.get_active()]))
        tb.pack_start(self.preset_cb, False, False, 0)
        tb.pack_start(_sep(), False, False, 4)

        tb.pack_start(Gtk.Label(label="Vol:"), False, False, 0)
        self.master_vol = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.master_vol.set_range(0, 1); self.master_vol.set_value(.85)
        self.master_vol.set_digits(2); self.master_vol.set_size_request(90, -1)
        self.master_vol.set_draw_value(False)
        tb.pack_start(self.master_vol, False, False, 0)
        tb.pack_start(_sep(), False, False, 4)

        for lbl, fn, tip in [
                ("▶",  "_play_test", "Test nuty A4"),
                ("⏹",  "_stop_all",  "Stop wszystkich głosów"),
                ("🔇", "_all_off",   "MIDI Panic — natychmiastowe wyciszenie"),
        ]:
            b = Gtk.Button(label=lbl); b.set_tooltip_text(tip)
            b.connect("clicked", lambda w, f=fn: getattr(self, f)())
            tb.pack_start(b, False, False, 0)
        tb.pack_start(_sep(), False, False, 4)

        self.auto_btn = Gtk.ToggleButton(label="⟳ Auto")
        self.auto_btn.connect("toggled", self._toggle_auto)
        tb.pack_start(self.auto_btn, False, False, 0)
        self.auto_mode = Gtk.ComboBoxText()
        for m in ["random", "lissajous", "spiral"]:
            self.auto_mode.append_text(m)
        self.auto_mode.set_active(0)
        self.auto_mode.connect(
            "changed", lambda w: setattr(self.auto_move, 'mode', w.get_active_text()))
        tb.pack_start(self.auto_mode, False, False, 0)
        tb.pack_start(Gtk.Label(label=" Spd:"), False, False, 0)
        self.auto_spd = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.auto_spd.set_range(.001, .05); self.auto_spd.set_value(.008)
        self.auto_spd.set_digits(3); self.auto_spd.set_size_request(80, -1)
        self.auto_spd.set_draw_value(False)
        self.auto_spd.connect(
            "value-changed", lambda w: setattr(self.auto_move, 'speed', w.get_value()))
        tb.pack_start(self.auto_spd, False, False, 0)
        tb.pack_start(_sep(), False, False, 4)

        tb.pack_start(Gtk.Label(label="Dur:"), False, False, 0)
        self.dur_sp = Gtk.SpinButton()
        self.dur_sp.set_range(.5, 8.); self.dur_sp.set_value(3.5)
        self.dur_sp.set_increments(.5, 1); self.dur_sp.set_digits(1)
        self.dur_sp.connect("value-changed",
                            lambda w: setattr(self, '_duration', w.get_value()))
        tb.pack_start(self.dur_sp, False, False, 0)
        tb.pack_start(Gtk.Label(label="s"), False, False, 0)
        tb.pack_start(_sep(), False, False, 4)
        b = Gtk.Button(label="💾 WAV")
        b.connect("clicked", self._save_wav)
        tb.pack_start(b, False, False, 0)

        # ── obszar główny ─────────────────────────────────────────
        main = Gtk.Box(spacing=3); main.set_border_width(2)
        root.pack_start(main, True, True, 0)

        # Lewa kolumna: OSC A + OSC D
        lc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        lc.set_size_request(212, -1)
        main.pack_start(lc, False, False, 0)
        self.osc_panels = {}
        for oid, col in [('A', VectorPadWidget.OSC_COLORS[0]),
                          ('D', VectorPadWidget.OSC_COLORS[3])]:
            p = OscPanel(oid, col, self._on_param_change)
            self.osc_panels[oid] = p
            lc.pack_start(p, True, True, 0)

        # Środek: pad
        mc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        main.pack_start(mc, True, True, 0)
        self.pad = VectorPadWidget(on_change=self._on_pad_change)
        self.pad.set_size_request(370, 390)
        mc.pack_start(self.pad, True, True, 0)

        # Prawa kolumna: OSC B + OSC C
        rc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        rc.set_size_request(212, -1)
        main.pack_start(rc, False, False, 0)
        for oid, col in [('B', VectorPadWidget.OSC_COLORS[1]),
                          ('C', VectorPadWidget.OSC_COLORS[2])]:
            p = OscPanel(oid, col, self._on_param_change)
            self.osc_panels[oid] = p
            rc.pack_start(p, True, True, 0)

        # Prawy panel globalnych parametrów
        gp = self._build_global_panel()
        main.pack_start(gp, False, False, 0)

        # ── dolna sekcja ──────────────────────────────────────────
        bot = Gtk.Box(spacing=4)
        root.pack_start(bot, False, False, 0)
        bot.pack_start(self._build_adsr_panel(), False, False, 0)
        bot.pack_start(self._build_vec_env_panel(), True, True, 0)

        # ── sekwencer ─────────────────────────────────────────────
        root.pack_start(self._build_seq_panel(), False, False, 0)

        # ── pasek MIDI status ─────────────────────────────────────
        self._build_midi_bar(root)

    # ── panel globalnych parametrów ───────────────────────────────
    def _build_global_panel(self):
        fr = Gtk.Frame(label=" Globalne ")
        fr.set_size_request(228, -1)
        vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        vb.set_border_width(6); fr.add(vb)

        def sl(lbl, lo, hi, val, digs, cb, tt=""):
            hb = Gtk.Box(spacing=3)
            lb = Gtk.Label(label=lbl); lb.set_width_chars(6); lb.set_xalign(0)
            hb.pack_start(lb, False, False, 0)
            sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
            sc.set_range(lo, hi); sc.set_value(val)
            sc.set_digits(digs); sc.set_draw_value(True)
            sc.set_hexpand(True)
            if tt: sc.set_tooltip_text(tt)
            sc.connect("value-changed", cb)
            hb.pack_start(sc, True, True, 0)
            vb.pack_start(hb, False, False, 0)
            return sc

        vb.pack_start(Gtk.Label(label="─── Filtr ───"), False, False, 2)
        self.flt_cut_sc = sl("Cut",  100, 20000, 18000, 0,
                             lambda w: setattr(self, '_flt_cut', w.get_value()),
                             "Odcięcie filtra LP [Hz] — CC74")
        self.flt_res_sc = sl("Res",  0, 1, 0, 2,
                             lambda w: setattr(self, '_flt_res', w.get_value()),
                             "Rezonans filtra — CC71")

        vb.pack_start(Gtk.Label(label="─── Efekty ───"), False, False, 2)
        self.reverb_sc = sl("Rev",   0, .9, 0, 2,
                            lambda w: setattr(self, '_reverb', w.get_value()),
                            "Pogłos (multi-echo) — CC91")

        vb.pack_start(Gtk.Label(label="─── LFO ───"), False, False, 2)
        self.lfo_rate_sc = sl("Rate", .1, 20, 5, 1,
                              lambda w: setattr(self, '_lfo_rate', w.get_value()),
                              "Szybkość LFO [Hz] — CC76")
        self.lfo_dep_sc  = sl("Dep",  0, 1, 0, 2,
                              lambda w: setattr(self, '_lfo_dep', w.get_value()),
                              "Głębokość vibrato — CC77")

        vb.pack_start(Gtk.Label(label="─── Pitch ───"), False, False, 2)
        self.pb_lbl = Gtk.Label(label="Bend:  0.00")
        self.pb_lbl.set_xalign(0); vb.pack_start(self.pb_lbl, False, False, 0)

        vb.pack_start(Gtk.Label(label="─── Kanał MIDI ───"), False, False, 2)
        hb_ch = Gtk.Box(spacing=3)
        hb_ch.pack_start(Gtk.Label(label="Ch:"), False, False, 0)
        self.midi_ch_sp = Gtk.SpinButton()
        self.midi_ch_sp.set_range(0, 16); self.midi_ch_sp.set_value(0)
        self.midi_ch_sp.set_tooltip_text("0 = wszystkie kanały")
        hb_ch.pack_start(self.midi_ch_sp, False, False, 0)
        hb_ch.pack_start(Gtk.Label(label="(0=all)"), False, False, 0)
        vb.pack_start(hb_ch, False, False, 0)

        # Mini podgląd bieżącej mieszanej fali
        vb.pack_start(Gtk.Label(label="─── Podgląd fali ───"), False, False, 2)
        self.mix_preview = Gtk.DrawingArea()
        self.mix_preview.set_size_request(-1, 60)
        self.mix_preview.connect("draw", self._draw_mix_preview)
        vb.pack_start(self.mix_preview, False, False, 0)

        return fr

    # ── ADSR panel ────────────────────────────────────────────────
    def _build_adsr_panel(self):
        fr = Gtk.Frame(label=" ADSR ")
        hb = Gtk.Box(spacing=5); hb.set_border_width(6); fr.add(hb)
        self.adsr_sc = []
        params = [("A", 0.001, 4., .01), ("D", 0.001, 4., .10),
                  ("S", 0.,    1., .80), ("R", 0.001, 4., .30)]
        for i, (lbl, lo, hi, val) in enumerate(params):
            vb2 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            vb2.pack_start(Gtk.Label(label=lbl), False, False, 0)
            sc = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
            sc.set_range(lo, hi); sc.set_value(val)
            sc.set_digits(3); sc.set_inverted(True)
            sc.set_draw_value(True); sc.set_size_request(52, 110)
            sc.connect("value-changed",
                       lambda w, j=i: self._adsr_changed(j, w.get_value()))
            vb2.pack_start(sc, True, True, 0)
            hb.pack_start(vb2, False, False, 0)
            self.adsr_sc.append(sc)
        return fr

    # ── Vector Envelope panel ─────────────────────────────────────
    def _build_vec_env_panel(self):
        fr = Gtk.Frame(label=" Vector Envelope — animowana ścieżka XY "
                       "[LPM=dodaj krok  PPM=usuń  Śr.=wyczyść] ")
        vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        vb.set_border_width(4); fr.add(vb)

        tb = Gtk.Box(spacing=5)
        vb.pack_start(tb, False, False, 0)
        self.venv_btn = Gtk.ToggleButton(label="▶ Env")
        self.venv_btn.connect("toggled", self._toggle_vec_env)
        tb.pack_start(self.venv_btn, False, False, 0)

        lp = Gtk.CheckButton(label="Loop")
        lp.set_active(True)
        lp.connect("toggled", lambda w: setattr(self.vec_env, 'loop', w.get_active()))
        tb.pack_start(lp, False, False, 0)

        tb.pack_start(Gtk.Label(label="Tempo:"), False, False, 0)
        self.venv_spd = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.venv_spd.set_range(.1, 8.); self.venv_spd.set_value(1.)
        self.venv_spd.set_digits(1); self.venv_spd.set_size_request(80, -1)
        self.venv_spd.set_draw_value(True)
        tb.pack_start(self.venv_spd, False, False, 0)
        b = Gtk.Button(label="🗑 Reset")
        b.connect("clicked", self._vec_env_reset)
        tb.pack_start(b, False, False, 0)
        tb.pack_start(Gtk.Label(label=" ─ kliknij w pole poniżej, aby dodać punkt ─"),
                      False, False, 4)

        self.venv_da = Gtk.DrawingArea()
        self.venv_da.set_size_request(-1, 90)
        self.venv_da.connect("draw", self._venv_draw)
        mask = (Gdk.EventMask.BUTTON_PRESS_MASK |
                Gdk.EventMask.BUTTON_RELEASE_MASK |
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.venv_da.set_events(mask)
        self.venv_da.connect("button-press-event",   self._venv_press)
        self.venv_da.connect("button-release-event", self._venv_release)
        self.venv_da.connect("motion-notify-event",  self._venv_motion)
        self._venv_drag = None   # indeks przeciąganego kroku
        vb.pack_start(self.venv_da, True, True, 0)
        return fr

    # ── MIDI status bar ───────────────────────────────────────────
    def _build_midi_bar(self, root):
        hb = Gtk.Box(spacing=8); hb.set_border_width(3)
        root.pack_start(hb, False, False, 0)

        def _mk(txt, wchars):
            lb = Gtk.Label(label=txt); lb.set_width_chars(wchars)
            hb.pack_start(lb, False, False, 0)
            return lb

        self.lbl_pb   = _mk("PB:  0.00",  12)
        self.lbl_mod  = _mk("MOD:0.00",   10)
        self.lbl_at   = _mk("AT: 0.00",   10)
        self.lbl_note = _mk("♪ ---",       10)
        self.lbl_cc   = _mk("CC: ---",     20)
        self.lbl_log  = Gtk.Label(label=""); self.lbl_log.set_xalign(0)
        hb.pack_start(self.lbl_log, True, True, 0)

    # ══════════════════════════════════════════════════════════════
    #  SEKWENCER — panel UI
    # ══════════════════════════════════════════════════════════════
    def _build_seq_panel(self):
        fr = Gtk.Frame(label=
            " ■ SEQUENCER  [Warstwa 1=PLAY: klawiatura gra synth] "
            "[Warstwa 2=PROG: klawiatura programuje zaznaczony krok] ")
        fr.set_border_width(2)
        vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        vb.set_border_width(4); fr.add(vb)

        # ── pasek 1: transport + BPM + parametry ──────────────────
        tb1 = Gtk.Box(spacing=5); vb.pack_start(tb1, False, False, 0)

        # Transport
        self.seq_play_btn = Gtk.ToggleButton(label="▶ PLAY")
        self.seq_play_btn.connect("toggled", self._seq_toggle_play)
        tb1.pack_start(self.seq_play_btn, False, False, 0)

        b_rst = Gtk.Button(label="↺ Reset")
        b_rst.connect("clicked", lambda w: self._seq_reset())
        tb1.pack_start(b_rst, False, False, 0)

        tb1.pack_start(_sep(), False, False, 4)

        # BPM
        tb1.pack_start(Gtk.Label(label="BPM:"), False, False, 0)
        self.bpm_sp = Gtk.SpinButton()
        self.bpm_sp.set_range(20, 300); self.bpm_sp.set_value(120)
        self.bpm_sp.set_increments(1, 10); self.bpm_sp.set_digits(0)
        self.bpm_sp.set_width_chars(5)
        self.bpm_sp.connect("value-changed",
                            lambda w: setattr(self.seq, 'bpm', w.get_value()))
        tb1.pack_start(self.bpm_sp, False, False, 0)

        # Tap BPM
        self._tap_times: list = []
        tap_b = Gtk.Button(label="TAP")
        tap_b.set_tooltip_text("Wciśnij rytmicznie aby ustawić BPM")
        tap_b.connect("clicked", self._seq_tap_bpm)
        tb1.pack_start(tap_b, False, False, 0)

        tb1.pack_start(_sep(), False, False, 4)

        # Długość sekwencji
        tb1.pack_start(Gtk.Label(label="Kroków:"), False, False, 0)
        self.steps_sp = Gtk.SpinButton()
        self.steps_sp.set_range(1, 16); self.steps_sp.set_value(16)
        self.steps_sp.set_increments(1, 4); self.steps_sp.set_width_chars(3)
        self.steps_sp.connect("value-changed",
                              lambda w: setattr(self.seq, 'active_len',
                                                int(w.get_value())))
        tb1.pack_start(self.steps_sp, False, False, 0)

        tb1.pack_start(_sep(), False, False, 4)

        # Swing
        tb1.pack_start(Gtk.Label(label="Swing:"), False, False, 0)
        self.swing_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.swing_sc.set_range(0, 1); self.swing_sc.set_value(0)
        self.swing_sc.set_digits(2); self.swing_sc.set_size_request(90, -1)
        self.swing_sc.set_draw_value(True)
        self.swing_sc.connect("value-changed",
                              lambda w: setattr(self.seq, 'swing', w.get_value()))
        tb1.pack_start(self.swing_sc, False, False, 0)

        tb1.pack_start(_sep(), False, False, 6)

        # ── TRYB MIDI (Warstwa 1 / Warstwa 2) ─────────────────────
        self.prog_btn = Gtk.ToggleButton(label="▶ W1: PLAY")
        self.prog_btn.set_tooltip_text(
            "Warstwa 1=PLAY: MIDI klawiatura gra synth\n"
            "Warstwa 2=PROG: MIDI klawiatura programuje zaznaczony krok")
        self.prog_btn.connect("toggled", self._seq_toggle_prog_mode)
        tb1.pack_start(self.prog_btn, False, False, 0)

        self.adv_chk = Gtk.CheckButton(label="Auto-adv")
        self.adv_chk.set_active(True)
        self.adv_chk.set_tooltip_text(
            "Po zaprogramowaniu kroku automatycznie przejdź do następnego")
        self.adv_chk.connect("toggled",
                             lambda w: setattr(self, '_seq_auto_adv', w.get_active()))
        tb1.pack_start(self.adv_chk, False, False, 0)

        tb1.pack_start(_sep(), False, False, 6)

        # Wzorce
        tb1.pack_start(Gtk.Label(label="Wzorzec:"), False, False, 0)
        self.pat_cb = Gtk.ComboBoxText()
        for nm in SEQ_PATTERNS:
            self.pat_cb.append_text(nm)
        self.pat_cb.append_text("Losowy")
        self.pat_cb.set_active(0)
        tb1.pack_start(self.pat_cb, False, False, 0)
        b_pat = Gtk.Button(label="Załaduj")
        b_pat.connect("clicked", self._seq_load_pattern)
        tb1.pack_start(b_pat, False, False, 0)

        # ── pasek 2: 16 kroków (SeqStepWidget) ────────────────────
        self.step_widget = SeqStepWidget(self.seq, on_select=self._seq_select_step)
        vb.pack_start(self.step_widget, False, False, 0)

        # ── pasek 3: edycja zaznaczonego kroku ────────────────────
        ef = Gtk.Frame(label=" Edycja kroku ")
        vb.pack_start(ef, False, False, 0)
        eb = Gtk.Box(spacing=6); eb.set_border_width(4); ef.add(eb)

        # Numer edytowanego kroku
        self.step_edit_lbl = Gtk.Label()
        self.step_edit_lbl.set_markup("<b>Krok 1</b>")
        self.step_edit_lbl.set_width_chars(8)
        eb.pack_start(self.step_edit_lbl, False, False, 0)

        eb.pack_start(Gtk.Label(label="Nuta:"), False, False, 0)
        self.step_note_cb = Gtk.ComboBoxText()
        # Dodaj wszystkie nuty MIDI (C-1 do G9)
        for n in range(128):
            self.step_note_cb.append_text(f"{midi_note_name(n)} ({n})")
        self.step_note_cb.set_active(60)
        self.step_note_cb.connect("changed", self._step_note_changed)
        eb.pack_start(self.step_note_cb, False, False, 0)

        eb.pack_start(Gtk.Label(label="Vel:"), False, False, 0)
        self.step_vel_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.step_vel_sc.set_range(1, 127); self.step_vel_sc.set_value(100)
        self.step_vel_sc.set_digits(0); self.step_vel_sc.set_size_request(90, -1)
        self.step_vel_sc.set_draw_value(True)
        self.step_vel_sc.connect("value-changed", self._step_vel_changed)
        eb.pack_start(self.step_vel_sc, False, False, 0)

        eb.pack_start(Gtk.Label(label="Gate:"), False, False, 0)
        self.step_gate_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.step_gate_sc.set_range(0.05, 1.0); self.step_gate_sc.set_value(0.8)
        self.step_gate_sc.set_digits(2); self.step_gate_sc.set_size_request(80, -1)
        self.step_gate_sc.set_draw_value(True)
        self.step_gate_sc.connect("value-changed", self._step_gate_changed)
        eb.pack_start(self.step_gate_sc, False, False, 0)

        eb.pack_start(Gtk.Label(label="Dł:"), False, False, 0)
        self.step_len_cb = Gtk.ComboBoxText()
        for lbl in ["1/16", "1/8", "1/4"]:
            self.step_len_cb.append_text(lbl)
        self.step_len_cb.set_active(0)
        self.step_len_cb.connect("changed", self._step_len_changed)
        eb.pack_start(self.step_len_cb, False, False, 0)

        eb.pack_start(_sep(), False, False, 4)

        # XY kroku
        eb.pack_start(Gtk.Label(label="X:"), False, False, 0)
        self.step_nx_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.step_nx_sc.set_range(-1, 1); self.step_nx_sc.set_value(0)
        self.step_nx_sc.set_digits(2); self.step_nx_sc.set_size_request(70, -1)
        self.step_nx_sc.set_draw_value(True)
        self.step_nx_sc.connect("value-changed", self._step_xy_changed)
        eb.pack_start(self.step_nx_sc, False, False, 0)

        eb.pack_start(Gtk.Label(label="Y:"), False, False, 0)
        self.step_ny_sc = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.step_ny_sc.set_range(-1, 1); self.step_ny_sc.set_value(0)
        self.step_ny_sc.set_digits(2); self.step_ny_sc.set_size_request(70, -1)
        self.step_ny_sc.set_draw_value(True)
        self.step_ny_sc.connect("value-changed", self._step_xy_changed)
        eb.pack_start(self.step_ny_sc, False, False, 0)

        # Kopiuj XY z pada
        b_cp = Gtk.Button(label="↓ Pad→Krok")
        b_cp.set_tooltip_text("Kopiuj aktualną pozycję joysticka do tego kroku")
        b_cp.connect("clicked", self._step_copy_pad_xy)
        eb.pack_start(b_cp, False, False, 0)

        eb.pack_start(_sep(), False, False, 4)

        # Active toggle
        self.step_active_btn = Gtk.ToggleButton(label="● Aktywny")
        self.step_active_btn.set_active(True)
        self.step_active_btn.connect("toggled", self._step_active_changed)
        eb.pack_start(self.step_active_btn, False, False, 0)

        # Kopiuj / Wklej krok
        b_copy = Gtk.Button(label="⎘ Kopiuj")
        b_copy.connect("clicked", self._step_copy)
        eb.pack_start(b_copy, False, False, 0)
        b_paste = Gtk.Button(label="⎗ Wklej")
        b_paste.connect("clicked", self._step_paste)
        eb.pack_start(b_paste, False, False, 0)

        # Wyczyść krok
        b_clr = Gtk.Button(label="✕ Wyczyść")
        b_clr.connect("clicked", self._step_clear)
        eb.pack_start(b_clr, False, False, 0)

        self._step_clipboard: dict = {}   # schowek dla kroku

        # Inicjalizuj UI edycji dla kroku 0
        self._seq_select_step(0)
        return fr

    # ══════════════════════════════════════════════════════════════
    #  SEKWENCER — metody silnika
    # ══════════════════════════════════════════════════════════════
    def _seq_toggle_play(self, btn):
        if btn.get_active():
            btn.set_label("⏹ STOP")
            self.seq.start()
        else:
            btn.set_label("▶ PLAY")
            self.seq.stop()

    def _seq_reset(self):
        """Zatrzymaj i wróć do kroku 1."""
        self.seq_play_btn.set_active(False)
        self.seq.current  = 0
        self.seq._elapsed = 0.0
        if hasattr(self, 'step_widget'):
            self.step_widget.queue_draw()

    def _seq_tap_bpm(self, w=None):
        """Wylicza BPM z rytmicznych wciśnięć."""
        import time as _time
        now = _time.time()
        self._tap_times.append(now)
        # Zostaw tylko ostatnie 4 uderzenia
        if len(self._tap_times) > 4:
            self._tap_times = self._tap_times[-4:]
        if len(self._tap_times) >= 2:
            diffs = [self._tap_times[i+1] - self._tap_times[i]
                     for i in range(len(self._tap_times)-1)]
            avg  = sum(diffs) / len(diffs)
            bpm  = 60.0 / avg
            bpm  = max(20, min(300, bpm))
            self.seq.bpm = bpm
            self.bpm_sp.set_value(int(bpm))

    def _seq_toggle_prog_mode(self, btn):
        self._seq_prog_mode = btn.get_active()
        if self._seq_prog_mode:
            btn.set_label("⌨ W2: PROG")
            btn.set_tooltip_text(
                "PROG aktywny — MIDI klawiatura programuje "
                f"krok {self._seq_edit_step + 1}. "
                "Kliknij krok w siatce, potem graj na klawiaturze.")
        else:
            btn.set_label("▶ W1: PLAY")
            btn.set_tooltip_text("PLAY — MIDI klawiatura gra synth normalnie")
        if hasattr(self, 'lbl_log'):
            self.lbl_log.set_text(
                f"Tryb: {'⌨ PROG — programujesz krok ' + str(self._seq_edit_step+1) if self._seq_prog_mode else '▶ PLAY — klawiatura gra normalnie'}")

    def _seq_load_pattern(self, w=None):
        nm = self.pat_cb.get_active_text()
        if nm == "Losowy":
            self.seq.randomize()
        elif nm:
            self.seq.load_pattern(nm)
        if hasattr(self, 'step_widget'):
            self.step_widget.queue_draw()
        self._seq_select_step(self._seq_edit_step)

    def _seq_select_step(self, idx: int):
        """Zaznacza krok do edycji — aktualizuje panel edit."""
        self._seq_edit_step = idx
        if hasattr(self, 'step_widget'):
            self.step_widget.selected = idx
            self.step_widget.queue_draw()
        if not hasattr(self, 'step_note_cb'):
            return
        step = self.seq.steps[idx]
        # Blokuj sygnały przy aktualizacji UI
        self._step_updating = True
        self.step_edit_lbl.set_markup(f"<b>Krok {idx+1}</b>")
        self.step_note_cb.set_active(step.note)
        self.step_vel_sc.set_value(step.velocity)
        self.step_gate_sc.set_value(step.gate)
        self.step_len_cb.set_active({1:0, 2:1, 4:2}.get(step.length, 0))
        self.step_nx_sc.set_value(step.nx)
        self.step_ny_sc.set_value(step.ny)
        self.step_active_btn.set_active(step.active)
        self.step_active_btn.set_label("● Aktywny" if step.active else "○ Nieaktywny")
        self._step_updating = False
        # Aktualizuj etykietę trybu PROG
        if self._seq_prog_mode and hasattr(self, 'prog_btn'):
            self.prog_btn.set_tooltip_text(
                f"PROG: klawiatura → krok {idx+1}")

    # ── edycja parametrów kroku ───────────────────────────────────
    def _step_note_changed(self, cb):
        if getattr(self, '_step_updating', False): return
        step = self.seq.steps[self._seq_edit_step]
        step.note = cb.get_active()
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_vel_changed(self, sc):
        if getattr(self, '_step_updating', False): return
        self.seq.steps[self._seq_edit_step].velocity = int(sc.get_value())
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_gate_changed(self, sc):
        if getattr(self, '_step_updating', False): return
        self.seq.steps[self._seq_edit_step].gate = sc.get_value()
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_len_changed(self, cb):
        if getattr(self, '_step_updating', False): return
        self.seq.steps[self._seq_edit_step].length = [1, 2, 4][cb.get_active()]
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_xy_changed(self, sc):
        if getattr(self, '_step_updating', False): return
        step = self.seq.steps[self._seq_edit_step]
        step.nx = self.step_nx_sc.get_value()
        step.ny = self.step_ny_sc.get_value()
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_active_changed(self, btn):
        if getattr(self, '_step_updating', False): return
        active = btn.get_active()
        self.seq.steps[self._seq_edit_step].active = active
        btn.set_label("● Aktywny" if active else "○ Nieaktywny")
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_copy_pad_xy(self, w=None):
        """Kopiuje aktualną pozycję joysticka do edytowanego kroku."""
        step = self.seq.steps[self._seq_edit_step]
        step.nx = self.pad.nx
        step.ny = self.pad.ny
        self._step_updating = True
        self.step_nx_sc.set_value(step.nx)
        self.step_ny_sc.set_value(step.ny)
        self._step_updating = False
        if hasattr(self, 'step_widget'): self.step_widget.queue_draw()

    def _step_copy(self, w=None):
        step = self.seq.steps[self._seq_edit_step]
        self._step_clipboard = {
            'note':     step.note,     'velocity': step.velocity,
            'gate':     step.gate,     'nx':       step.nx,
            'ny':       step.ny,       'length':   step.length,
            'active':   step.active,
        }

    def _step_paste(self, w=None):
        if not self._step_clipboard: return
        step = self.seq.steps[self._seq_edit_step]
        step.from_dict({
            'note': self._step_clipboard.get('note', 60),
            'vel':  self._step_clipboard.get('velocity', 100),
            'gate': self._step_clipboard.get('gate', .8),
            'nx':   self._step_clipboard.get('nx', 0.),
            'ny':   self._step_clipboard.get('ny', 0.),
            'len':  self._step_clipboard.get('length', 1),
            'active': self._step_clipboard.get('active', True),
        })
        self._seq_select_step(self._seq_edit_step)

    def _step_clear(self, w=None):
        step = self.seq.steps[self._seq_edit_step]
        step.active = False
        step.note = 60; step.velocity = 100
        step.gate = .8; step.nx = 0.; step.ny = 0.; step.length = 1
        self._seq_select_step(self._seq_edit_step)

    # ── granie kroku (callback z Sequencer.on_step_on) ────────────
    def _seq_step_on(self, step_idx: int, step: SeqStep):
        """Wywołany przez sekwencer gdy krok wystrzeliwuje — gra nutę."""
        # Przesuń joystick do pozycji XY tego kroku (wizualne)
        GLib.idle_add(self.pad.set_pos, step.nx, step.ny, False)
        # Podświetl krok
        GLib.idle_add(self.step_widget.queue_draw)
        # Zagraj nutę
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        v    = int(step.velocity * self._vol * self._expr)
        self.voices.note_on(
            step.note, v, oscs, step.nx, step.ny,
            tuple(self._adsr), self._pb,
            self._flt_cut, self._flt_res,
            self._reverb, self._lfo_dep, self._lfo_rate,
            self._at,
            duration=min(self._duration, 8.0))
        GLib.idle_add(self.lbl_note.set_text,
                      f"♪ {midi_note_name(step.note)} [SEQ {step_idx+1}]")

    def _seq_step_off(self, note: int):
        """Wywołany gdy gate kończy nutę sekwencera."""
        self.voices.note_off(note, 80)

    # ══════════════════════════════════════════════════════════════
    #  MIDI callbacks (z wątku rtmidi → GLib.idle_add dla UI)
    # ══════════════════════════════════════════════════════════════
    def _midi_on(self, note, vel):
        # ── WARSTWA 2: PROG — klawiatura programuje krok ──────────
        if self._seq_prog_mode:
            step = self.seq.steps[self._seq_edit_step]
            step.note     = note
            step.velocity = vel
            step.active   = True
            if self._seq_link_pad:
                step.nx = self.pad.nx
                step.ny = self.pad.ny
            # Krótko zagraj nutę, żeby słyszeć efekt
            oscs = {k: p.s for k, p in self.osc_panels.items()}
            self.voices.note_on(
                note, vel, oscs, step.nx, step.ny,
                (.005, .05, .0, .12), self._pb,
                self._flt_cut, self._flt_res, self._reverb,
                self._lfo_dep, self._lfo_rate, self._at, .5)
            # Auto-advance do następnego kroku
            next_idx = self._seq_edit_step
            if self._seq_auto_adv:
                next_idx = (self._seq_edit_step + 1) % self.seq.active_len
            GLib.idle_add(self._seq_select_step, next_idx)
            GLib.idle_add(self.step_widget.queue_draw)
            GLib.idle_add(self.lbl_log.set_text,
                          f"PROG krok {self._seq_edit_step+1} ← "
                          f"{midi_note_name(note)} v={vel}")
            return   # NIE graj przez normalny synth path

        # ── WARSTWA 1: PLAY — normalne granie ─────────────────────
        self._active_notes.add(note)
        v = int(vel * self._vol * self._expr)
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        self.voices.note_on(
            note, v, oscs, self.pad.nx, self.pad.ny,
            tuple(self._adsr), self._pb,
            self._flt_cut, self._flt_res,
            self._reverb, self._lfo_dep, self._lfo_rate,
            self._at, self._duration)
        GLib.idle_add(self._ui_note_on, note, vel)

    def _midi_off(self, note):
        self._active_notes.discard(note)
        if not self._sustain:
            self.voices.note_off(note)
        GLib.idle_add(self.lbl_note.set_text, "♪ ---")

    def _midi_pb(self, val):
        self._pb = val
        GLib.idle_add(self.lbl_pb.set_text, f"PB:{val:+.2f}")
        GLib.idle_add(self.pb_lbl.set_text,  f"Bend:{val:+.2f}")

    def _midi_at(self, val):
        self._at = val / 127.0
        GLib.idle_add(self.lbl_at.set_text, f"AT:{self._at:.2f}")

    def _midi_poly_at(self, note, val):
        # Per-note aftertouch — logujemy tylko
        GLib.idle_add(self.lbl_cc.set_text,
                      f"PolyAT n={note_name(note)} v={val}")

    def _midi_pc(self, prog):
        idx = prog % len(PRESETS)
        GLib.idle_add(self._apply_preset, PRESETS[idx])
        GLib.idle_add(self.lbl_log.set_text,
                      f"PC → {PRESETS[idx]['name']}")

    # Mapa CC → parametr wewnętrzny
    _CC_MAP = {
        1:  ('_mod',      lambda v: v/127.),
        7:  ('_vol',      lambda v: v/127.),
        11: ('_expr',     lambda v: v/127.),
        71: ('_flt_res',  lambda v: v/127.),
        74: ('_flt_cut',  lambda v: 200. + v/127.*19800.),
        76: ('_lfo_rate', lambda v: .2  + v/127.*19.8),
        77: ('_lfo_dep',  lambda v: v/127.),
        91: ('_reverb',   lambda v: v/127.*.9),
    }

    def _midi_cc(self, cc, val):
        if cc in self._CC_MAP:
            attr, conv = self._CC_MAP[cc]
            setattr(self, attr, conv(val))
        elif cc == 64:   # Sustain
            prev = self._sustain
            self._sustain = (val >= 64)
            if prev and not self._sustain:
                # Sustain zwolniony — stop trzymanych nut
                for n in list(self._active_notes):
                    self.voices.note_off(n)
        elif cc == 72:   # Release
            self._adsr[3] = val / 127. * 4.
        elif cc == 73:   # Attack
            self._adsr[0] = max(.001, val / 127. * 4.)
        elif cc == 75:   # Decay
            self._adsr[1] = val / 127. * 4.
        elif cc == 2:    # Breath → mały ruch pada ku centrum
            bv = val / 127.
            nx = self.pad.nx * (1. - bv * .15)
            ny = self.pad.ny * (1. - bv * .15)
            GLib.idle_add(self.pad.set_pos, nx, ny)

        GLib.idle_add(self.lbl_cc.set_text, f"CC{cc}={val}")
        GLib.idle_add(self._sync_global_ui)

    def _ui_note_on(self, note, vel):
        self.lbl_note.set_text(f"♪ {note_name(note)} v={vel}")
        self.lbl_log.set_text(f"Note On: {note_name(note)}  vel={vel}")
        self.mix_preview.queue_draw()

    def _sync_global_ui(self):
        """Aktualizuje widgety UI zgodnie ze stanem wewnętrznym (po CC)."""
        try:
            self.flt_cut_sc.set_value(self._flt_cut)
            self.flt_res_sc.set_value(self._flt_res)
            self.reverb_sc.set_value(self._reverb)
            self.lfo_rate_sc.set_value(self._lfo_rate)
            self.lfo_dep_sc.set_value(self._lfo_dep)
            for i, sc in enumerate(self.adsr_sc):
                sc.set_value(self._adsr[i])
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  Parametry zmiany (pad + OSC panels)
    # ══════════════════════════════════════════════════════════════
    def _on_pad_change(self, nx, ny):
        """Wywołany gdy joystick się porusza (mysz lub auto-ruch)."""
        self.mix_preview.queue_draw()
        self._send_wave_cb()

    def _on_param_change(self):
        """Wywołany gdy zmieni się ustawienie dowolnego OSC."""
        for p in self.osc_panels.values():
            p.preview.queue_draw()
        self.mix_preview.queue_draw()
        self._send_wave_cb()

    def _adsr_changed(self, idx, val):
        self._adsr[idx] = val

    # ══════════════════════════════════════════════════════════════
    #  Timer tick — auto-ruch + vector envelope
    # ══════════════════════════════════════════════════════════════
    def _tick(self):
        dt = self.TICK_MS / 1000.
        if self.auto_btn.get_active():
            x, y = self.auto_move.step()
            self.pad.set_pos(x, y)
        if self.venv_btn.get_active():
            spd    = self.venv_spd.get_value()
            result = self.vec_env.tick(dt * spd)
            if result:
                x, y = result
                GLib.idle_add(lambda: self.pad.set_pos(x, y))
        # ── Sekwencer ─────────────────────────────────────────────
        if self.seq.running:
            fired = self.seq.tick(dt)
            if fired:
                GLib.idle_add(self.step_widget.queue_draw)
        return True

    def _toggle_auto(self, btn):
        pass   # stan odczytywany w _tick przez btn.get_active()

    def _toggle_vec_env(self, btn):
        if btn.get_active():
            self.vec_env.start()
        else:
            self.vec_env.stop()

    def _vec_env_pos_cb(self, x, y):
        GLib.idle_add(self.pad.set_pos, x, y)

    # ══════════════════════════════════════════════════════════════
    #  Rysowanie Vector Envelope (mini-pad z krokami)
    # ══════════════════════════════════════════════════════════════
    def _venv_draw(self, widget, cr):
        W = widget.get_allocated_width()
        H = widget.get_allocated_height()
        pad = 8

        cr.set_source_rgb(.06, .06, .09); cr.paint()
        cr.set_source_rgba(.18, .18, .22, 1.)
        cr.rectangle(pad, pad, W - 2*pad, H - 2*pad); cr.fill()
        cr.set_source_rgba(.30, .30, .38, .7); cr.set_line_width(.8)
        cr.rectangle(pad, pad, W - 2*pad, H - 2*pad); cr.stroke()

        def s2px(nx, ny):
            x = pad + (nx + 1) / 2 * (W - 2*pad)
            y = pad + (1 - ny) / 2 * (H - 2*pad)
            return x, y

        steps = self.vec_env.steps
        if not steps:
            cr.set_source_rgba(.4, .4, .5, .6); cr.set_font_size(9)
            cr.move_to(W/2 - 80, H/2 + 4)
            cr.show_text("Kliknij aby dodać punkt envelope"); return

        # Rysuj linie między krokami
        if len(steps) > 1:
            cr.set_source_rgba(.5, .75, 1., .7); cr.set_line_width(1.5)
            for i in range(len(steps)):
                a = steps[i]; b = steps[(i+1) % len(steps)]
                ax, ay = s2px(a['x'], a['y'])
                bx, by = s2px(b['x'], b['y'])
                cr.move_to(ax, ay); cr.line_to(bx, by); cr.stroke()

        # Rysuj węzły
        for i, st in enumerate(steps):
            px2, py2 = s2px(st['x'], st['y'])
            cr.set_source_rgba(.3, .7, 1., .9)
            cr.arc(px2, py2, 6, 0, 6.28); cr.fill()
            cr.set_source_rgb(1, 1, 1); cr.set_line_width(1.2)
            cr.arc(px2, py2, 6, 0, 6.28); cr.stroke()
            cr.set_font_size(8); cr.set_source_rgba(1, 1, 1, .8)
            cr.move_to(px2 + 7, py2 - 3)
            cr.show_text(f"{i+1} {st['t']:.1f}s")

        # Aktualny wskaźnik joysticka na mini-padzie
        jx, jy = s2px(self.pad.nx, self.pad.ny)
        cr.set_source_rgba(1., .8, .2, .85)
        cr.arc(jx, jy, 4, 0, 6.28); cr.fill()

    def _venv_px2n(self, x, y):
        W = self.venv_da.get_allocated_width()
        H = self.venv_da.get_allocated_height()
        pad = 8
        nx = (x - pad) / (W - 2*pad) * 2 - 1
        ny = 1 - (y - pad) / (H - 2*pad) * 2
        return max(-1., min(1., nx)), max(-1., min(1., ny))

    def _venv_press(self, widget, event):
        x, y = event.x, event.y
        nx, ny = self._venv_px2n(x, y)
        W = self.venv_da.get_allocated_width()
        H = self.venv_da.get_allocated_height()
        pad = 8

        def s2px(s):
            sx = pad + (s['x'] + 1) / 2 * (W - 2*pad)
            sy = pad + (1 - s['y']) / 2 * (H - 2*pad)
            return sx, sy

        if event.button == 2:  # Środkowy — wyczyść
            self.vec_env.steps.clear(); self.venv_da.queue_draw(); return

        if event.button == 3:  # PPM — usuń najbliższy
            best = None; best_d = 15**2
            for i, st in enumerate(self.vec_env.steps):
                sx, sy = s2px(st)
                d = (sx - x)**2 + (sy - y)**2
                if d < best_d:
                    best_d = d; best = i
            if best is not None:
                self.vec_env.steps.pop(best)
            self.venv_da.queue_draw(); return

        # LPM — sprawdź czy drag istniejącego, inaczej dodaj
        self._venv_drag = None
        for i, st in enumerate(self.vec_env.steps):
            sx, sy = s2px(st)
            if (sx - x)**2 + (sy - y)**2 < 14**2:
                self._venv_drag = i; return

        # Dodaj nowy krok
        self.vec_env.steps.append({'x': nx, 'y': ny, 't': 1.0})
        self._venv_drag = len(self.vec_env.steps) - 1
        self.venv_da.queue_draw()

    def _venv_release(self, widget, event):
        self._venv_drag = None

    def _venv_motion(self, widget, event):
        if self._venv_drag is not None and self._venv_drag < len(self.vec_env.steps):
            nx, ny = self._venv_px2n(event.x, event.y)
            self.vec_env.steps[self._venv_drag]['x'] = nx
            self.vec_env.steps[self._venv_drag]['y'] = ny
            self.venv_da.queue_draw()

    def _vec_env_reset(self, w=None):
        self.vec_env.steps.clear()
        self.venv_btn.set_active(False)
        self.venv_da.queue_draw()

    # ══════════════════════════════════════════════════════════════
    #  Podgląd zmieszanej fali (mini DrawingArea w global panel)
    # ══════════════════════════════════════════════════════════════
    def _draw_mix_preview(self, widget, cr):
        W = widget.get_allocated_width()
        H = widget.get_allocated_height()
        cr.set_source_rgb(.06, .06, .09); cr.paint()
        if not any(p.s.level > .01 for p in self.osc_panels.values()):
            return
        n    = min(W, 256)
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        wA, wB, wC, wD = blend_weights(self.pad.nx, self.pad.ny)
        weights = {'A': wA, 'B': wB, 'C': wC, 'D': wD}
        wave = np.zeros(n)
        for oid, osc in oscs.items():
            w = weights[oid] * osc.level
            if w < 1e-5: continue
            raw = gen_wave(osc.wave, n, osc.extra)
            wave += raw * w
        mx = np.max(np.abs(wave))
        if mx > 1e-9: wave /= mx
        # Blended kolor
        bc = [sum(VectorPadWidget.OSC_COLORS[i][c] * ww
                  for i, ww in enumerate([wA, wB, wC, wD]))
              for c in range(3)]
        cr.set_source_rgba(bc[0], bc[1], bc[2], .9)
        cr.set_line_width(1.3)
        for i, v in enumerate(wave):
            x = i * W / n; y = H / 2 - v * (H / 2 - 3)
            if i == 0: cr.move_to(x, y)
            else:      cr.line_to(x, y)
        cr.stroke()
        cr.set_source_rgba(.25, .25, .32, .5); cr.set_line_width(.5)
        cr.move_to(0, H/2); cr.line_to(W, H/2); cr.stroke()

    # ══════════════════════════════════════════════════════════════
    #  Obsługa przycisków paska górnego
    # ══════════════════════════════════════════════════════════════
    def _play_test(self):
        if not PYGAME_OK: return
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        self.voices.note_on(69, 100, oscs, self.pad.nx, self.pad.ny,
                            tuple(self._adsr), self._pb,
                            self._flt_cut, self._flt_res,
                            self._reverb, self._lfo_dep, self._lfo_rate,
                            self._at, self._duration)

    def _stop_all(self):
        self.voices.all_off(160)

    def _all_off(self):
        """MIDI Panic."""
        self.seq.stop()
        if hasattr(self, 'seq_play_btn'):
            self.seq_play_btn.set_active(False)
        self.voices.all_off(20)
        self._active_notes.clear()
        if PYGAME_OK: pygame.mixer.stop()

    def _open_midi_dlg(self, w=None):
        ports = self.midi.list_ports()
        dlg   = MIDIPortDialog(self, ports)
        resp  = dlg.run()
        if resp == Gtk.ResponseType.OK and dlg.selected_idx >= 0:
            if self.midi.open(dlg.selected_idx):
                nm = self.midi.port_name[:22]
                self.midi_lbl.set_markup(
                    f'<span foreground="lime">⬤ {nm}</span>')
                self.lbl_log.set_text(f"MIDI: {self.midi.port_name}")
            else:
                self.midi_lbl.set_markup('<span foreground="red">⬤ BŁĄD</span>')
        dlg.destroy()

    # ══════════════════════════════════════════════════════════════
    #  Preset
    # ══════════════════════════════════════════════════════════════
    def _apply_preset(self, p: dict):
        for oid in ('A', 'B', 'C', 'D'):
            if oid in p: self.osc_panels[oid].set_from_dict(p[oid])
        adsr = p.get('adsr', [.01, .1, .8, .3])
        for i, sc in enumerate(self.adsr_sc):
            sc.set_value(adsr[i])
        self._adsr = list(adsr)
        flt = p.get('flt', [18000., .0])
        self.flt_cut_sc.set_value(flt[0]);  self._flt_cut = flt[0]
        self.flt_res_sc.set_value(flt[1]);  self._flt_res = flt[1]
        self.reverb_sc.set_value(p.get('rev', .0))
        self._reverb = p.get('rev', .0)
        lfo = p.get('lfo', [5., .0])
        self.lfo_rate_sc.set_value(lfo[0]);  self._lfo_rate = lfo[0]
        self.lfo_dep_sc.set_value(lfo[1]);   self._lfo_dep  = lfo[1]
        self.mix_preview.queue_draw()
        self._send_wave_cb()

    # ══════════════════════════════════════════════════════════════
    #  Callback do ED_Waves (bieżąca zmieszana fala jako próbka)
    # ══════════════════════════════════════════════════════════════
    def _send_wave_cb(self):
        if not self.wave_callback:
            return
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        wave = synthesize_voice(
            69, 100, oscs, self.pad.nx, self.pad.ny,
            (.005, .05, 1.0, .05),   # fast preview ADSR
            0., 20000., .0, .0, .0, 5., .0,
            duration=1.0)
        wA, wB, wC, wD = blend_weights(self.pad.nx, self.pad.ny)
        params = {
            'f0':     440.0,
            'nx':     self.pad.nx,
            'ny':     self.pad.ny,
            'blend':  [wA, wB, wC, wD],
            'waves':  {k: p.s.wave for k, p in self.osc_panels.items()},
            'source': 'vector_synth',
        }
        self.wave_callback(wave, params)

    # ══════════════════════════════════════════════════════════════
    #  Zapis WAV
    # ══════════════════════════════════════════════════════════════
    def _save_wav(self, w=None):
        dlg = Gtk.FileChooserDialog(
            title="Zapisz WAV", parent=self, action=Gtk.FileChooserAction.SAVE)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_SAVE,   Gtk.ResponseType.OK)
        ff = Gtk.FileFilter(); ff.set_name("WAV (*.wav)"); ff.add_pattern("*.wav")
        dlg.add_filter(ff)
        if dlg.run() == Gtk.ResponseType.OK:
            path = dlg.get_filename()
            if not path.endswith(".wav"): path += ".wav"
            oscs = {k: p.s for k, p in self.osc_panels.items()}
            wave = synthesize_voice(
                69, 100, oscs, self.pad.nx, self.pad.ny,
                tuple(self._adsr), self._pb,
                self._flt_cut, self._flt_res, self._reverb,
                self._lfo_dep, self._lfo_rate, self._at, self._duration)
            arr = (wave * 32767).astype(np.int16)
            if SCIPY_OK:
                _wavfile.write(path, SR, arr)
            else:
                import wave as _wave
                with _wave.open(path, 'w') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2)
                    wf.setframerate(SR); wf.writeframes(arr.tobytes())
            self.lbl_log.set_text(f"Zapisano: {path}")
        dlg.destroy()

    # ══════════════════════════════════════════════════════════════
    #  Publiczne API dla ED_Waves
    # ══════════════════════════════════════════════════════════════
    def get_current_wave(self) -> np.ndarray:
        oscs = {k: p.s for k, p in self.osc_panels.items()}
        return synthesize_voice(
            69, 100, oscs, self.pad.nx, self.pad.ny,
            tuple(self._adsr), 0., self._flt_cut, self._flt_res,
            self._reverb, self._lfo_dep, self._lfo_rate, 0.,
            duration=self._duration)

    def get_current_params(self) -> dict:
        wA, wB, wC, wD = blend_weights(self.pad.nx, self.pad.ny)
        return {
            'f0': 440.,
            'nx': self.pad.nx, 'ny': self.pad.ny,
            'blend': [wA, wB, wC, wD],
            'waves': {k: p.s.wave for k, p in self.osc_panels.items()},
            'adsr':  list(self._adsr),
        }

    def _on_destroy(self, widget):
        if self._timer_id:
            GLib.source_remove(self._timer_id)
        self.seq.stop()
        self.midi.close()
        self.voices.all_off(0)
        Gtk.main_quit()


# ═══════════════════════════════════════════════════════════════════
#  Standalone
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    def _demo_cb(wave, params):
        pass  # W standalone callback nic nie robi — wszystko obsługuje okno

    app = ChaosPadWindow(callback=_demo_cb)
    app.show_all()
    Gtk.main()
