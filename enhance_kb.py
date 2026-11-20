#!/usr/bin/env python3
"""
Enhance a recording with the fine-tuned DeepFilterNet3 keyboard-noise model (best_model.pt).

Command line:
    python enhance_kb.py a.wav --model best_model.pt
    python enhance_kb.py a.wav --model best_model.pt --out a_clean.wav --device cuda --atten-lim-db 30

Notebook / Colab cell: edit the settings below and run the whole file as one cell.

- Any sample rate (16 kHz input is upsampled to 48 kHz internally and converted back).
- Mono or multi-channel (each channel is processed separately).
- Needs torch + numpy. soundfile, loguru, appdirs and deepfilternet==0.5.6 (model code only)
  are installed automatically on the first run if missing.
- config.ini (model architecture) is looked up next to best_model.pt, in the DeepFilterNet3_kb/
  or pretrained/DeepFilterNet3/ folders made by the training notebook, or downloaded from the
  official DeepFilterNet repository.
"""
# ============================ settings ============================
INPUT_WAV = "a.wav"
MODEL_PT = "/content/drive/MyDrive/kb_denoise/runs/dfn3_kb_ft/best_model.pt"
CONFIG_INI = None       # None → search automatically (see above)
OUTPUT_WAV = None       # None → <input>_enhanced.wav
DEVICE = None           # None → "cuda" if available, else "cpu"
ATTEN_LIM_DB = None     # e.g. 30 → remove at most 30 dB (sounds more natural); None → no limit
# ==================================================================

import argparse, configparser, importlib, importlib.util, math, re, shutil, subprocess, sys, time, types, zipfile
import urllib.request
from pathlib import Path


def ensure_packages():
    need = [p for m, p in [("soundfile", "soundfile"), ("loguru", "loguru"), ("appdirs", "appdirs")]
            if importlib.util.find_spec(m) is None]
    if need:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
    if importlib.util.find_spec("df") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "deepfilternet==0.5.6"])
    importlib.invalidate_caches()


ensure_packages()
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

MEAN_NORM_INIT = (-60.0, -90.0)     # libDF constants
UNIT_NORM_INIT = (0.001, 0.0001)
DFN3_ZIP = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip"


# ------------------------- DeepFilterNet import shims -------------------------
def dfn_erb_widths(sr=48000, fft_size=960, nb_bands=32, min_nb_freqs=2):
    # port of libDF's erb_fb(): FFT bins per ERB band
    f32 = np.float32
    freq_width = f32(sr) / f32(fft_size)
    f2e = lambda f: f32(9.265) * np.log1p(f32(f) / (f32(24.7) * f32(9.265)))
    e2f = lambda e: f32(24.7) * f32(9.265) * (np.exp(f32(e) / f32(9.265)) - f32(1.0))
    lo, hi = f2e(f32(0.0)), f2e(f32(sr // 2))
    step = (hi - lo) / f32(nb_bands)
    widths, prev, over = np.zeros(nb_bands, dtype=np.int64), 0, 0
    for i in range(1, nb_bands + 1):
        fb = int(np.floor(e2f(lo + f32(i) * step) / freq_width + f32(0.5)))
        nb = fb - prev - over
        if nb < min_nb_freqs:
            over, nb = min_nb_freqs - nb, min_nb_freqs
        else:
            over = 0
        widths[i - 1], prev = nb, fb
    widths[-1] += 1
    extra = int(widths.sum()) - (fft_size // 2 + 1)
    if extra > 0:
        widths[-1] -= extra
    return widths


class _DFState:
    # stand-in for libdf.DF (Rust); the model code only needs erb_widths()
    def __init__(self, sr=48000, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2, *args, **kwargs):
        self._sr, self._fft, self._hop, self._nb, self._min = int(sr), int(fft_size), int(hop_size), int(nb_bands), int(min_nb_erb_freqs)
    def erb_widths(self):
        return dfn_erb_widths(self._sr, self._fft, self._nb, self._min)
    def sr(self): return self._sr
    def fft_size(self): return self._fft
    def hop_size(self): return self._hop
    def nb_erb(self): return self._nb


def install_shims():
    try:
        import torchaudio  # noqa: F401
    except Exception:
        ta = types.ModuleType("torchaudio")
        def _ta_getattr(name):
            raise AttributeError(name)
        ta.__getattr__ = _ta_getattr
        sys.modules["torchaudio"] = ta
    lib = types.ModuleType("libdf")
    lib.DF = _DFState
    lib.unit_norm_init = lambda n: np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], int(n)).astype(np.float32)
    lib.erb_norm_init = lambda n: np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], int(n)).astype(np.float32)
    lib.mean_norm_init = lib.erb_norm_init
    def _lib_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        def _unavailable(*a, **k):
            raise RuntimeError(f"libdf.{name}() is not used (PyTorch DSP is used instead)")
        return _unavailable
    lib.__getattr__ = _lib_getattr
    sys.modules["libdf"] = lib


def patch_df_sources():
    # fix imports removed in newer torch / torchaudio (torchaudio.backend.*, torch._six)
    pkg = Path(list(importlib.util.find_spec("df").submodule_search_locations)[0])
    def repl(m):
        ind, names = m.group(1), [n.strip() for n in m.group(2).split(",") if n.strip()]
        return (f"{ind}try:  # kb-patch\n{ind}    from torchaudio import {', '.join(names)}\n"
                f"{ind}except Exception:  # kb-patch\n{ind}    {' = '.join(names)} = object")
    for f in pkg.rglob("*.py"):
        try:
            src = f.read_text()
            new = re.sub(r"^([ \t]*)from torchaudio\.backend[\w.]* import ([\w, ]+?)[ \t]*$", repl, src, flags=re.M)
            new = new.replace("from torch._six import string_classes", "string_classes = (str, bytes)  # kb-patch")
            if new != src:
                f.write_text(new)
        except OSError as e:
            print(f"warning: could not patch {f}: {e}")
    return pkg


def import_df_config(pkg):
    try:
        import df  # noqa: F401
        from df.config import config
        return config
    except Exception as e:
        print(f"`import df` failed ({type(e).__name__}) → using a minimal df/__init__.py")
        init = pkg / "__init__.py"
        if not (pkg / "__init__.py.orig").exists():
            shutil.copy(init, pkg / "__init__.py.orig")
        init.write_text("from .config import config  # kb-patch: minimal init\n")
        for m in [m for m in sys.modules if m == "df" or m.startswith("df.")]:
            del sys.modules[m]
        importlib.invalidate_caches()
        from df.config import config
        return config


# ------------------------- PyTorch DSP -------------------------
_RS_KERNELS = {}

def _sinc_kernel(orig, new, device, lowpass_filter_width=16, rolloff=0.945):
    key = (orig, new, str(device))
    if key not in _RS_KERNELS:
        base = min(orig, new) * rolloff
        width = math.ceil(lowpass_filter_width * orig / base)
        idx = torch.arange(-width, width + orig, dtype=torch.float64)[None, None] / orig
        t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx
        t = (t * base).clamp(-lowpass_filter_width, lowpass_filter_width)
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        t = t * math.pi
        kernel = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t) * window * (base / orig)
        _RS_KERNELS[key] = (kernel.to(device=device, dtype=torch.float32), width)
    return _RS_KERNELS[key]


def resample(x, orig_sr, new_sr):
    # band-limited windowed-sinc resampling, x: [..., L], zero delay
    orig_sr, new_sr = int(orig_sr), int(new_sr)
    if orig_sr == new_sr:
        return x
    g = math.gcd(orig_sr, new_sr)
    orig, new = orig_sr // g, new_sr // g
    kernel, width = _sinc_kernel(orig, new, x.device)
    shape, L = x.shape[:-1], x.shape[-1]
    y = F.pad(x.reshape(-1, 1, L).float(), (width, width + orig))
    y = F.conv1d(y, kernel, stride=orig)
    y = y.transpose(1, 2).reshape(y.shape[0], -1)
    n_out = math.ceil(new * L / orig)
    return y[:, :n_out].reshape(*shape, n_out)


def fix_length(x, n):
    L = x.shape[-1]
    return x[..., :n] if L >= n else F.pad(x, (0, n - L))


def read_dfn_params(cfg_path):
    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    gi = lambda k, d: cp.getint("df", k, fallback=d)
    return dict(sr=gi("sr", 48000), fft_size=gi("fft_size", 960), hop_size=gi("hop_size", 480),
                nb_erb=gi("nb_erb", 32), nb_df=gi("nb_df", 96), min_nb_erb_freqs=gi("min_nb_erb_freqs", 2),
                norm_tau=cp.getfloat("df", "norm_tau", fallback=1.0))


def dfn_norm_alpha(sr, hop, tau):
    a_, prec, a = math.exp(-(hop / sr) / tau), 3, 1.0
    while a >= 1.0:
        a = round(a_, prec)
        prec += 1
    return a


class DFNDSP:
    # PyTorch port of libDF: Vorbis-window STFT, ERB features, unit-normalised complex bins
    def __init__(self, erb_widths, device, sr=48000, fft_size=960, hop_size=480, nb_df=96, alpha=0.99):
        self.sr, self.n_fft, self.hop, self.nb_df, self.alpha = sr, fft_size, hop_size, nb_df, float(alpha)
        self.erb_widths = np.asarray(erb_widths, dtype=np.int64)
        n = torch.arange(fft_size, dtype=torch.float64)
        self.win = torch.sin(0.5 * math.pi * torch.sin(0.5 * math.pi * (n + 0.5) / (fft_size // 2)) ** 2).float().to(device)
        self.wnorm = 1.0 / (fft_size ** 2 / (2 * hop_size))
        n_freqs, n_bands = fft_size // 2 + 1, len(self.erb_widths)
        assert int(self.erb_widths.sum()) == n_freqs
        M = torch.zeros(n_freqs, n_bands)
        start = 0
        for b, w in enumerate(self.erb_widths.tolist()):
            M[start:start + w, b] = 1.0 / w
            start += w
        self.erb_mat = M.to(device)
        self.erb_init = torch.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], n_bands, device=device)
        self.unit_init = torch.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], nb_df, device=device)

    def stft(self, x):
        frames = F.pad(x, (self.hop, 0)).unfold(-1, self.n_fft, self.hop)
        return torch.fft.rfft(frames * self.win, dim=-1) * self.wnorm

    def istft(self, X):
        frames = torch.fft.irfft(X, n=self.n_fft, dim=-1) * self.n_fft * self.win
        B = frames.shape[0]
        a, b = frames[..., :self.hop], frames[..., self.hop:]
        y = a + F.pad(b[:, :-1], (0, 0, 1, 0))
        y = torch.cat([y.reshape(B, -1), b[:, -1]], dim=-1)
        return y[:, self.hop:]

    def _ema(self, x, init, chunk=128):
        a = self.alpha
        B, T, K = x.shape
        s = init.double().reshape(1, K).expand(B, K)
        outs = []
        for t0 in range(0, T, chunk):
            xc = x[:, t0:t0 + chunk].double()
            k = torch.arange(xc.shape[1], device=x.device, dtype=torch.float64)
            acc = torch.cumsum(xc * (a ** (-k))[None, :, None], dim=1)
            sc = (a ** k)[None, :, None] * ((1 - a) * acc + a * s[:, None, :])
            outs.append(sc)
            s = sc[:, -1]
        return torch.cat(outs, dim=1).float()

    @torch.no_grad()
    def features(self, x):
        X = self.stft(x)
        P = X.real.square() + X.imag.square()
        erb = 10.0 * torch.log10(P @ self.erb_mat + 1e-10)
        erb = (erb - self._ema(erb, self.erb_init)) / 40.0
        Xd = X[..., :self.nb_df]
        s = self._ema(Xd.abs(), self.unit_init).clamp_min(1e-12)
        Xd = Xd / s.sqrt()
        return torch.view_as_real(X).unsqueeze(1), erb.unsqueeze(1), torch.view_as_real(Xd).unsqueeze(1)


# ------------------------- model -------------------------
def find_config(model_pt, config_ini=None):
    if config_ini:
        return Path(config_ini)
    mdir = Path(model_pt).resolve().parent
    candidates = [mdir / "config.ini", mdir / "DeepFilterNet3_kb" / "config.ini", mdir.parent / "config.ini"]
    if len(mdir.parents) > 1:
        candidates.append(mdir.parents[1] / "pretrained" / "DeepFilterNet3" / "config.ini")
    for c in candidates:
        if c.exists():
            return c
    cache = Path.home() / ".cache" / "kb_dfn3"
    cfg = cache / "config.ini"
    if not cfg.exists():
        print("config.ini not found locally → downloading the official DeepFilterNet3 config")
        cache.mkdir(parents=True, exist_ok=True)
        zp = cache / "DeepFilterNet3.zip"
        urllib.request.urlretrieve(DFN3_ZIP, zp)
        with zipfile.ZipFile(zp) as z:
            name = next(n for n in z.namelist() if n.endswith("config.ini") and "__MACOSX" not in n)
            cfg.write_bytes(z.read(name))
    return cfg


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(model_pt, cfg_path, device):
    install_shims()
    pkg = patch_df_sources()
    df_config = import_df_config(pkg)
    p = read_dfn_params(cfg_path)
    try:
        df_config.load(str(cfg_path), config_must_exist=True, allow_defaults=True, allow_reload=True)
    except TypeError:
        df_config.load(str(cfg_path))
    from df.deepfilternet3 import init_model
    model = init_model(df_state=_DFState(p["sr"], p["fft_size"], p["hop_size"], p["nb_erb"], p["min_nb_erb_freqs"]))
    sd = torch_load(model_pt)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    sd = {k.replace("clc", "df"): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    params = {n for n, _ in model.named_parameters()}
    missing = [k for k in res.missing_keys if k in params]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters missing in {model_pt}: {missing[:5]}")
    model = model.to(device).eval()

    widths = dfn_erb_widths(p["sr"], p["fft_size"], p["nb_erb"], p["min_nb_erb_freqs"])
    fb = getattr(model, "erb_fb", None)
    if isinstance(fb, torch.Tensor) and fb.dim() == 2:
        w = (fb.detach().cpu() > 0).sum(0).numpy().astype(np.int64)
        if int(w.sum()) == fb.shape[0]:
            widths = w
    dsp = DFNDSP(widths, device, p["sr"], p["fft_size"], p["hop_size"], p["nb_df"],
                 dfn_norm_alpha(p["sr"], p["hop_size"], p["norm_tau"]))
    return model, dsp


# ------------------------- enhancement -------------------------
@torch.no_grad()
def enhance_48k(model, dsp, x):
    # x: [B, L] at 48 kHz → [B, L], time-aligned
    L = x.shape[-1]
    xp = F.pad(x, (0, (-L) % dsp.hop + dsp.n_fft))
    spec, erb_feat, spec_feat = dsp.features(xp)
    out = model(spec.clone(), erb_feat, spec_feat)[0]
    return dsp.istft(torch.view_as_complex(out.float().squeeze(1).contiguous()))[:, :L]


@torch.no_grad()
def enhance_48k_long(model, dsp, x, chunk_s=60.0, overlap_s=3.0):
    # long files: overlapping chunks with a linear cross-fade
    L, C, O = x.shape[-1], int(chunk_s * dsp.sr), int(overlap_s * dsp.sr)
    if L <= C + O:
        return enhance_48k(model, dsp, x)
    out, wsum, start = torch.zeros_like(x), torch.zeros(L, device=x.device), 0
    while True:
        end = min(L, start + C + O)
        w = torch.ones(end - start, device=x.device)
        if start > 0:
            w[:O] = torch.linspace(0, 1, O, device=x.device)
        if end < L:
            w[-O:] = torch.linspace(1, 0, O, device=x.device)
        out[..., start:end] += enhance_48k(model, dsp, x[..., start:end]) * w
        wsum[start:end] += w
        if end == L:
            break
        start = end - O
    return out / wsum.clamp_min(1e-8)


@torch.no_grad()
def enhance_audio(model, dsp, x, sr, atten_lim_db=None):
    # x: [C, L] tensor at any sample rate → enhanced [C, L] at the same rate
    y = fix_length(resample(enhance_48k_long(model, dsp, resample(x, sr, dsp.sr)), dsp.sr, sr), x.shape[-1])
    if atten_lim_db is not None and abs(atten_lim_db) > 0:
        lim = 10 ** (-abs(atten_lim_db) / 20)
        y = lim * x + (1 - lim) * y
    return y


def enhance_file(input_wav, model_pt, output_wav=None, config_ini=None, device=None, atten_lim_db=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = find_config(model_pt, config_ini)
    model, dsp = load_model(model_pt, cfg, device)
    x, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)       # [L, C]
    xt = torch.from_numpy(np.ascontiguousarray(x.T)).to(device)            # [C, L]
    t0 = time.time()
    y = enhance_audio(model, dsp, xt, sr, atten_lim_db)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    out = Path(output_wav) if output_wav else Path(input_wav).with_name(Path(input_wav).stem + "_enhanced.wav")
    y = y.clamp(-1.0, 1.0).cpu().numpy().T
    sf.write(str(out), y[:, 0] if y.shape[1] == 1 else y, sr)
    print(f"{input_wav} → {out}  |  {x.shape[0] / sr:.1f} s, {sr} Hz, {x.shape[1]} ch  |  "
          f"{dt:.2f} s on {device.type} ({x.shape[0] / sr / max(dt, 1e-9):.0f}x real time)  |  config: {cfg}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Keyboard-noise suppression with fine-tuned DeepFilterNet3")
    ap.add_argument("input", nargs="?", default=INPUT_WAV, help="input audio file (any sample rate)")
    ap.add_argument("--model", default=MODEL_PT, help="best_model.pt from the training notebook")
    ap.add_argument("--config", default=CONFIG_INI, help="DeepFilterNet3 config.ini (optional)")
    ap.add_argument("--out", default=OUTPUT_WAV, help="output file (default: <input>_enhanced.wav)")
    ap.add_argument("--device", default=DEVICE, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--atten-lim-db", type=float, default=ATTEN_LIM_DB, help="max attenuation in dB (optional)")
    args = ap.parse_args([] if "ipykernel" in sys.modules else None)   # notebook → use the settings above
    enhance_file(args.input, args.model, args.out, args.config, args.device, args.atten_lim_db)


if __name__ == "__main__":
    main()
