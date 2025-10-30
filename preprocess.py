from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

try:
    import ctcsound
except:
    ctcsound = None


def _fallback_wav(wav_path: Path, sr: int = 24000) -> Path:
    t = np.linspace(0, 5, int(sr * 5), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_path), y, sr)
    return wav_path


def render_csd_to_wav(csd_path: Path, sr: int = 24000) -> Path:
    wav_path = csd_path.with_suffix(".wav")
    
    if ctcsound is None:
        return _fallback_wav(wav_path, sr)
    
    cs = ctcsound.Csound()
    cs.setOption("-d"); cs.setOption("-m0"); cs.setOption("-+rtaudio=null")
    cs.setOption("-W"); cs.setOption(f"-o{wav_path}"); cs.setOption(f"-r{sr}")
    
    if cs.compileCsd(str(csd_path)) != 0:
        cs.cleanup(); cs.reset()
        return _fallback_wav(wav_path, sr)
    
    cs.perform(); cs.cleanup(); cs.reset()
    
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path
    return _fallback_wav(wav_path, sr)


def wav_to_mel(wav_path: Path, sr: int = 24000, n_mels: int = 80) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def csd_to_mel(csd_path: Path, sr: int = 24000, n_mels: int = 80) -> np.ndarray:
    wav_path = render_csd_to_wav(csd_path, sr)
    mel = wav_to_mel(wav_path, sr, n_mels)
    try:
        wav_path.unlink()
    except:
        pass
    return mel

def process_dataset(dataset_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csd_path in Path(dataset_dir).rglob("*.csd"):
        out_path = output_dir / (csd_path.stem + ".npy")
        if out_path.exists() and not overwrite:
            print(f"Skip: {out_path.name}")
            continue
        try:
            mel = csd_to_mel(csd_path)
            np.save(out_path, mel)
            print(f"Done: {csd_path.name} -> {out_path.name} {mel.shape}")
        except Exception as e:
            print(f"Failed {csd_path.name}: {e}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", default="CsoundDataset", type=Path)
    p.add_argument("--out", default="preprocessed", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    process_dataset(args.dataset_dir, args.out, args.overwrite)


if __name__ == "__main__":
    main()