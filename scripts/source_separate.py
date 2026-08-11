from src.models.source_separation import source_separation, Predictor
from src.preprocessing.standardization import standardization

from src.utils import write_mp3
import json # Import json module
from tqdm import tqdm
import torch
import os
import argparse
import librosa


def read_manifest(manifest_path):
    """Reads a JSONL manifest file line by line into a list of dicts."""
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line: {e}")
    return data

def main(args):
    """
    Main function to process and segment audio files.
    """
    # --- 1. Setup Models ---
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    # Source separation model config
    ss_cfg = {
        "model_path": "pretrained_models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise": True,
        "margin": 44100,
        "chunks": 15,
        "n_fft": 6144,
        "dim_t": 8,
        "dim_f": 3072
    }
    
    separate_predictor = Predictor(args=ss_cfg, device=device_name)

    # --- 2. Read Manifest ---
    print(f"Reading manifest from: {args.manifest}")
    manifest_data = read_manifest(args.manifest)
    print(f"Found {len(manifest_data)} entries.")

    # --- 3. Process Audio Files ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")
    total_duration = sum(item.get("duration", 0.0) for item in manifest_data)


    # Iterate through manifest data with a progress bar
    with tqdm(
        total=total_duration,
        unit="sec",
        unit_scale=True,
        desc="Processing audio"
    ) as pbar:
        for item in manifest_data:
            duration = item.get("duration", 0.0)

            audio_filepath = item.get("audio_filepath")
            audio_filepath = os.path.join(args.root, audio_filepath)
            
            if not audio_filepath or not os.path.exists(audio_filepath):
                print(f"⚠️ Skipping item, audio_filepath not found or invalid: {audio_filepath}")
                continue

            try:
                # Get the original base name and construct the output path
                base_name = os.path.basename(audio_filepath).split('.')[0]
                channel_id = item.get("channel_id")
                output_channel_dir = os.path.join(output_dir, channel_id)
                os.makedirs(output_channel_dir, exist_ok=True)
                output_filename = f"{base_name}.mp3"
                output_path = os.path.join(output_channel_dir, output_filename)
                if os.path.exists(output_path):
                    continue
                audio = standardization(audio_filepath)  # Standardize the audio file

                # Run source separation
                separated_audio = source_separation(separate_predictor, audio)
                
                # The separated_audio dictionary now contains the separated vocals in 'waveform'
                # and the original sample rate in 'sample_rate' (if loaded correctly inside source_separation)
                vocals_waveform = separated_audio["waveform"]
                sample_rate = separated_audio["sample_rate"] # Assuming source_separation stores original rate

                # The ConvTDFNet.istft seems to return float, so we rely on write_mp3 to convert.
                write_mp3(output_path, vocals_waveform, sample_rate)

            except Exception as e:
                print(f"\n❌ Error processing {audio_filepath}: {e}")
            finally:
                pbar.update(duration)

    print("\n--- Processing complete! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio Source Separation Script")
    parser.add_argument(
        "--manifest", 
        type=str, 
        required=True, 
        help="Path to the input JSONL manifest file."
    )
    parser.add_argument(
        "--root", 
        type=str, 
        required=True, 
        help="Path to root of audio files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./separated_output", 
        help="Directory to save separated audio files."
    )
    args = parser.parse_args()
    main(args)
