import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import argparse
import subprocess
import shutil
from datetime import datetime


def get_unique_filename(base_name: str, extension: str, output_dir: Path) -> Path:
    """Generate a unique filename by adding timestamp if file exists"""
    filename = f"{base_name}{extension}"
    file_path = output_dir / filename
    
    if not file_path.exists():
        return file_path
    
    # Add timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{base_name}_{timestamp}{extension}"
    return output_dir / unique_filename


def load_trained_model(models_dir: Path = Path("models")):
    """Load the trained model from the models folder"""
    checkpoint_path = models_dir / "trained_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['model_config']
    
    # Create model architecture
    model = nn.ModuleList([
        nn.RNN(input_size=config['Hin'], 
               hidden_size=config['Hout'], 
               batch_first=True),
        nn.Linear(in_features=config['Hout'], 
                  out_features=config['output_dim'])
    ])
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Final training loss: {checkpoint['loss']:.4f}")
    print(f"Model config: {config}")
    
    return model, config


def generate_snap_parameters(model, mel_spectrogram, config, variation: float = 0.0, temperature: float = 1.0):
    """Generate snap parameters from a mel-spectrogram with optional variation"""
    with torch.no_grad():
        # Ensure input is the right shape: (1, time_frames, mel_bins)
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Add batch dimension
        
        # Handle dimension mismatch - if model expects different input size
        expected_mel_bins = config['Hin']
        actual_mel_bins = mel_spectrogram.size(-1)
        
        if actual_mel_bins != expected_mel_bins:
            print(f"Warning: Model expects {expected_mel_bins} mel bins, got {actual_mel_bins}")
            print("This suggests the model was trained with dummy data.")
            print("Reshaping input to match model expectations...")
            
            # Option 1: Resize to match model expectations
            if actual_mel_bins > expected_mel_bins:
                # Take first N bins
                mel_spectrogram = mel_spectrogram[:, :, :expected_mel_bins]
            else:
                # Pad with zeros
                padding = expected_mel_bins - actual_mel_bins
                mel_spectrogram = torch.cat([
                    mel_spectrogram, 
                    torch.zeros(mel_spectrogram.size(0), mel_spectrogram.size(1), padding)
                ], dim=-1)
        
        # Add noise to input for variation
        if variation > 0:
            noise = torch.randn_like(mel_spectrogram) * variation
            mel_spectrogram = mel_spectrogram + noise
        
        # Initialize hidden state
        ht_1 = torch.zeros(1, mel_spectrogram.size(0), config['Hout'])
        
        # Forward pass
        output, ht = model[0](mel_spectrogram, ht_1)
        
        # Generate prediction
        snap_params = model[1](ht.squeeze(0))
        
        # Apply temperature scaling for more variation
        if temperature != 1.0:
            snap_params = snap_params / temperature
        
        # Add random variation to output parameters
        if variation > 0:
            param_noise = torch.randn_like(snap_params) * variation * 0.8
            snap_params = snap_params + param_noise
            
            # Add some extreme randomization for crazy sounds
            if variation > 1.0:
                # Random parameter swapping and extreme values
                extreme_mask = torch.rand_like(snap_params) < 0.3
                snap_params[extreme_mask] = torch.randn_like(snap_params[extreme_mask]) * 2.0
                
                # Random sign flipping for some parameters
                sign_flip_mask = torch.rand_like(snap_params) < 0.2
                snap_params[sign_flip_mask] = -snap_params[sign_flip_mask]
        
        return snap_params.squeeze(0).numpy()  # Remove batch dimension and convert to numpy


def create_csd_template(snap_params, output_path: Path, template_name: str = "generated"):
    """Create a richer .csd file using generated snap parameters."""

    def p(n, default):
        try:
            v = float(snap_params[n])
        except Exception:
            v = default
        return max(min(v, 1.0), -1.0)

    p0 = p(0, 0.0)
    p1 = p(1, 0.0)
    p2 = p(2, 0.0)
    p3 = p(3, 0.0)
    p4 = p(4, 0.0)
    p5 = p(5, 0.0)
    p6 = p(6, 0.0)
    p7 = p(7, 0.0)

    csd_content = f"""<CsoundSynthesizer>
<CsOptions>
-o {template_name}.wav -W
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

; -----------------------------
; Generated global parameters
; -----------------------------
gk_p0 init {p0:.6f}
gk_p1 init {p1:.6f}
gk_p2 init {p2:.6f}
gk_p3 init {p3:.6f}
gk_p4 init {p4:.6f}
gk_p5 init {p5:.6f}
gk_p6 init {p6:.6f}
gk_p7 init {p7:.6f}

; -----------------------------
; Tables
; -----------------------------
giSine ftgen 1, 0, 4096, 10, 1
giTri  ftgen 2, 0, 4096, 10, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1

; -----------------------------
; Synth voice
; -----------------------------
instr 1
  kBaseHz   = 220 + gk_p0 * 220
  kDetune   = 0.5 + (gk_p1 * 0.5)
  kCutoff   = 800 + abs(gk_p2) * 5000
  kRes      = 0.1 + abs(gk_p6) * 0.85
  kNoiseAmt = gk_p5
  if (kNoiseAmt < 0) then
    kNoiseAmt = 0
  endif
  kNoiseAmt = kNoiseAmt * 0.4
  kChDepth  = gk_p7
  if (kChDepth < 0) then
    kChDepth = 0
  endif
  kChDepth = kChDepth * 0.008
  kWidth    = 0.2 + (abs(gk_p4) * 0.8)

  iAtk = 0.005 + abs(i(gk_p3)) * 0.150
  iDec = 0.050 + abs(i(gk_p3)) * 0.300
  iSus = 0.4  + abs(i(gk_p3)) * 0.5
  iRel = 0.200 + abs(i(gk_p3)) * 0.500
  kEnv linsegr 0, iAtk, 1, iDec, iSus, iRel, 0

  aOsc1 vco2 0.35, kBaseHz * (1 + kDetune*0.01)
  aOsc2 vco2 0.35, kBaseHz * (1 - kDetune*0.01)
  aSub  vco2 0.25, kBaseHz * 0.5

  aNoise rand 1
  aNoise = aNoise * kNoiseAmt

  aMix = aOsc1 + aOsc2 + aSub + aNoise

  aFilt moogladder aMix, kCutoff, kRes

  ; Simple stereo output with basic chorus effect
  aL = aFilt * kEnv * 0.7
  aR = aFilt * kEnv * 0.7
  
  ; Add simple stereo widening
  aOutL = aL * (1 + kChDepth)
  aOutR = aR * (1 - kChDepth)

  outs aOutL, aOutR
endin

</CsInstruments>
<CsScore>
i1 0 5
e
</CsScore>
</CsoundSynthesizer>"""

    # Ensure we only write the Csound content, not any extra data
    clean_content = csd_content.strip()
    if clean_content.endswith('</CsoundSynthesizer>'):
        output_path.write_text(clean_content)
        print(f"Generated .csd file: {output_path}")
    else:
        print(f"Warning: Generated .csd content appears malformed, skipping write to {output_path}")


def create_snap_file(snap_params, output_path: Path):
    """Create a .snaps file with the generated parameters"""
    snap_data = {}
    
    # Create parameter names
    for i, param in enumerate(snap_params):
        snap_data[f"param_{i}"] = float(param)
    
    # Write JSON snap file
    with open(output_path, 'w') as f:
        json.dump(snap_data, f, indent=2)
    
    print(f"Generated .snaps file: {output_path}")


def render_csd(csd_path: Path, template_name: str, timeout_sec: int = 120, csound_bin: str = "csound") -> Path | None:
    """Render a .csd file to audio using Csound with a timeout.

    Returns the path to the generated .wav if successful, else None.
    """
    if shutil.which(csound_bin) is None:
        print(f"Csound binary '{csound_bin}' not found in PATH. Skipping render.")
        return None

    try:
        print(f"Rendering {csd_path.name} with {csound_bin} (timeout={timeout_sec}s)...")
        result = subprocess.run(
            [csound_bin, csd_path.name],
            cwd=str(csd_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            print("Csound render failed:")
            print(result.stderr.strip())
            return None
        # Look for the generated .wav file (Csound creates it based on -o option)
        # The template uses the template_name as the base, not the csd filename
        wav_path = csd_path.parent / (template_name + ".wav")
        if wav_path.exists():
            print(f"Rendered audio: {wav_path}")
            return wav_path
        
        # If not found, look for any .wav files created recently
        wav_files = list(csd_path.parent.glob("*.wav"))
        if wav_files:
            # Get the most recently created .wav file
            latest_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
            print(f"Rendered audio: {latest_wav}")
            return latest_wav
            
        print("Render completed but expected .wav not found.")
        return None
    except subprocess.TimeoutExpired:
        print(f"Csound render timed out after {timeout_sec}s for {csd_path.name}.")
        return None
    except Exception as e:
        print(f"Error while rendering {csd_path.name}: {e}")
        return None


def generate_from_mel(mel_file: Path, models_dir: Path, output_dir: Path, *,
                      render: bool = False, timeout_sec: int = 120, csound_bin: str = "csound",
                      override_base_name: str | None = None, variation: float = 0.0, temperature: float = 1.0):
    """Generate new .csd and .snaps files from a mel-spectrogram"""
    # Load model
    model, config = load_trained_model(models_dir)
    
    # Load mel-spectrogram
    mel_data = np.load(mel_file)
    mel_tensor = torch.tensor(mel_data, dtype=torch.float32)
    
    # Transpose to match training format: (time_frames, mel_bins)
    if len(mel_tensor.shape) == 2:
        mel_tensor = mel_tensor.T
    
    print(f"Loaded mel-spectrogram: {mel_tensor.shape}")
    
    # Generate snap parameters with variation
    snap_params = generate_snap_parameters(model, mel_tensor, config, variation=variation, temperature=temperature)
    print(f"Generated snap parameters: {snap_params}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate files with unique names
    base_name = override_base_name if override_base_name else mel_file.stem
    csd_path = get_unique_filename(f"{base_name}_generated", ".csd", output_dir)
    snaps_path = get_unique_filename(f"{base_name}_generated", ".snaps", output_dir)
    
    create_csd_template(snap_params, csd_path, base_name)
    create_snap_file(snap_params, snaps_path)
    if render:
        render_csd(csd_path, base_name, timeout_sec=timeout_sec, csound_bin=csound_bin)
    
    return snap_params


def generate_random(model, config, output_dir: Path, num_samples: int = 5, *,
                    render: bool = False, timeout_sec: int = 120, csound_bin: str = "csound",
                    base_prefix: str | None = None, start_index: int = 1, variation: float = 0.0, temperature: float = 1.0):
    """Generate random mel-spectrograms and create corresponding .csd files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Generate random mel-spectrogram
        random_mel = torch.randn(config['L'], config['Hin'])
        
        # Generate snap parameters with variation
        snap_params = generate_snap_parameters(model, random_mel, config, variation=variation, temperature=temperature)
        
        # Create files with unique names
        if base_prefix:
            base_name = f"{base_prefix}{start_index + i}"
        else:
            base_name = f"random_sample_{i+1}"
        csd_path = get_unique_filename(base_name, ".csd", output_dir)
        snaps_path = get_unique_filename(base_name, ".snaps", output_dir)
        
        create_csd_template(snap_params, csd_path, base_name)
        create_snap_file(snap_params, snaps_path)
        if render:
            render_csd(csd_path, base_name, timeout_sec=timeout_sec, csound_bin=csound_bin)
        
        print(f"Generated random sample {i+1}: {snap_params[:3]}...")


def retrain_with_correct_dimensions():
    """Retrain the model with correct mel-spectrogram dimensions"""
    print("Retraining model with correct dimensions...")
    
    # Load actual mel-spectrograms
    preprocessed_dir = Path("preprocessed")
    mel_files = list(preprocessed_dir.glob("*.npy"))
    
    if not mel_files:
        print("No mel-spectrogram files found!")
        return
    
    # Get actual dimensions from first file
    sample_mel = np.load(mel_files[0])
    actual_mel_bins = sample_mel.shape[0]  # mel bins
    actual_time_frames = sample_mel.shape[1]  # time frames
    
    print(f"Actual mel dimensions: {actual_mel_bins} bins x {actual_time_frames} frames")
    
    # Create dummy snap parameters (since we don't have real ones)
    dummy_snap_params = torch.randn(len(mel_files), 10)  # 10 parameters per sample
    
    # Load and reshape mel data
    X_data = []
    for mel_file in mel_files:
        mel = np.load(mel_file)
        X_data.append(torch.tensor(mel.T, dtype=torch.float32))  # Transpose to (time, mel_bins)
    
    X = torch.stack(X_data)
    y = dummy_snap_params
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Create new model with correct dimensions
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    Hin = actual_mel_bins
    Hout = 64
    output_dim = y.shape[1]
    
    model = nn.ModuleList([
        nn.RNN(input_size=Hin, hidden_size=Hout, batch_first=True),
        nn.Linear(in_features=Hout, out_features=output_dim)
    ])
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print("Training new model...")
    for epoch in range(3):
        total_loss = 0
        for xb, yb in train_loader:
            ht_1 = torch.zeros(1, xb.size(0), Hout)
            output, ht = model[0](xb, ht_1)
            y_pred = model[1](ht.squeeze(0))
            loss = loss_fn(y_pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the retrained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss / len(train_loader),
        'epoch': 3,
        'model_config': {
            'Hin': Hin,
            'Hout': Hout,
            'output_dim': output_dim,
            'L': X.shape[1]
        }
    }, models_dir / "trained_model_correct.pt")
    
    torch.save(model, models_dir / "model_complete_correct.pt")
    print("Retrained model saved!")


def main():
    parser = argparse.ArgumentParser(description="Generate new .csd files using trained model")
    parser.add_argument("--models_dir", default="models", type=Path)
    parser.add_argument("--output_dir", default="generated", type=Path)
    parser.add_argument("--mel_file", type=Path, help="Specific mel-spectrogram file to use")
    parser.add_argument("--random", type=int, default=0, help="Generate N random samples")
    parser.add_argument("--render", action="store_true", help="Render generated .csd with Csound")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout seconds for Csound render")
    parser.add_argument("--csound", type=str, default="csound", help="Csound binary name/path")
    parser.add_argument("--name", type=str, default=None, help="Override base output name (e.g., gen1)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for sequential names (e.g., gen)")
    parser.add_argument("--start", type=int, default=1, help="Starting index for sequential names")
    parser.add_argument("--variation", type=float, default=0.6, help="Amount of variation/noise (0.0-2.0, default 0.6)")
    parser.add_argument("--temperature", type=float, default=2.5, help="Temperature scaling for more diversity (default 2.5)")
    parser.add_argument("--crazy", action="store_true", help="Enable extreme variation mode (variation=1.5, temperature=3.0)")
    parser.add_argument("--insane", action="store_true", help="Enable maximum chaos mode (variation=2.0, temperature=4.0)")
    parser.add_argument("--retrain", action="store_true", help="Retrain model with correct dimensions")
    
    args = parser.parse_args()
    
    # Handle extreme mode flags
    if args.insane:
        args.variation = 2.0
        args.temperature = 4.0
        print("🔥 INSANE MODE: Maximum chaos activated!")
    elif args.crazy:
        args.variation = 1.5
        args.temperature = 3.0
        print("🎵 CRAZY MODE: Extreme variation activated!")
    
    if args.retrain:
        retrain_with_correct_dimensions()
        return
    
    if args.mel_file:
        # Generate from specific mel file
        generate_from_mel(
            args.mel_file,
            args.models_dir,
            args.output_dir,
            render=args.render,
            timeout_sec=args.timeout,
            csound_bin=args.csound,
            override_base_name=args.name,
            variation=args.variation,
            temperature=args.temperature,
        )
    elif args.random > 0:
        # Generate random samples
        model, config = load_trained_model(args.models_dir)
        generate_random(
            model,
            config,
            args.output_dir,
            args.random,
            render=args.render,
            timeout_sec=args.timeout,
            csound_bin=args.csound,
            base_prefix=args.prefix or args.name,
            start_index=args.start,
            variation=args.variation,
            temperature=args.temperature,
        )
    else:
        # Generate from all available mel files
        preprocessed_dir = Path("preprocessed")
        mel_files = list(preprocessed_dir.glob("*.npy"))
        
        if not mel_files:
            print("No mel-spectrogram files found in preprocessed/")
            return
        
        print(f"Found {len(mel_files)} mel-spectrogram files")
        
        for idx, mel_file in enumerate(mel_files[:3]):  # Process first 3 files
            print(f"\nProcessing {mel_file.name}...")
            # Compute base name if prefix/name provided; otherwise derive from file
            override_name = None
            if args.prefix or args.name:
                base = args.prefix or args.name
                override_name = f"{base}{args.start + idx}"
            generate_from_mel(
                mel_file,
                args.models_dir,
                args.output_dir,
                render=args.render,
                timeout_sec=args.timeout,
                csound_bin=args.csound,
                override_base_name=override_name,
                variation=args.variation,
                temperature=args.temperature,
            )

if __name__ == "__main__":
    main()
