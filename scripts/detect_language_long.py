import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from torchaudio.transforms import Resample
import os
from tqdm import tqdm
import json
import argparse
import random
from pydub import AudioSegment
import numpy as np
import subprocess # Added for ffmpeg
import math # Added for splitting duration calculation

# Custom exception for large audio files
class AudioFileTooLargeError(Exception):
    """Custom exception for audio files exceeding processing limits."""
    pass

# ----------------------------- #
#         Load Components         #
# ----------------------------- #

def load_model_and_tokenizer(model_name=None):
    """Loads the Whisper model, processor, and tokenizer from a local path or Hugging Face."""
    fallback_model_name = "openai/whisper-large-v3"
    resolved_model_name = model_name or fallback_model_name

    if model_name:
        if os.path.exists(model_name):
            print(f"Loading Whisper model from local path: {model_name}")
        else:
            print(f"Whisper model path '{model_name}' was not found. Falling back to Hugging Face: {fallback_model_name}")
            resolved_model_name = fallback_model_name

    try:
        processor = WhisperProcessor.from_pretrained(resolved_model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        tokenizer = WhisperTokenizer.from_pretrained(resolved_model_name)
        return processor, model, tokenizer
    except Exception as e:
        if resolved_model_name != fallback_model_name:
            print(f"Failed to load Whisper model from local path '{resolved_model_name}': {e}")
            print(f"Falling back to Hugging Face: {fallback_model_name}")
            return load_model_and_tokenizer(fallback_model_name)
        raise

# ----------------------------- #
#      Load and Preprocess       #
# ----------------------------- #

def calculate_rms(waveform):
    """Calculates the Root Mean Square (RMS) energy of a waveform."""
    return torch.sqrt(torch.mean(waveform**2))

def load_and_preprocess_audio_segments(audio_path, target_sample_rate=16000, num_segments=5, min_segment_len=5, max_segment_len=20, rms_threshold=0.01):
    """
    Loads a single audio file, extracts random non-overlapping segments, resamples them,
    filters by RMS, and returns a list of waveforms for the valid segments.
    """
    segments = []
    try:
        audio = AudioSegment.from_file(audio_path)
        sr = audio.frame_rate
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # Convert to numpy array and then to torch tensor
        waveform = torch.from_numpy(np.array(audio.get_array_of_samples())).float() / (1 << (audio.sample_width * 8 - 1))
        waveform = waveform.unsqueeze(0) # Add channel dimension

        if sr != target_sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform[0] # mono
        audio_len_seconds = waveform.shape[0] / target_sample_rate

        if audio_len_seconds < min_segment_len:
            print(f"Skipping {audio_path}: audio too short ({audio_len_seconds:.2f}s) for min segment length of {min_segment_len}s.")
            return None

        if audio_len_seconds <= max_segment_len:
            # If audio is within or shorter than max_segment_len, take the whole audio
            if calculate_rms(waveform) > rms_threshold:
                segments.append(waveform)
            else:
                print(f"Skipping {audio_path}: whole audio RMS below threshold.")
        else:
            # Generate potential start points for non-overlapping segments
            available_start_samples = list(range(0, waveform.shape[0] - int(min_segment_len * target_sample_rate) + 1))
            
            # Refactored non-overlapping segment selection
            segments = []
            selected_segments_info = [] # Stores (start_sample, end_sample) for selected segments

            # Generate a pool of potential start times
            potential_start_times = []
            max_possible_start = waveform.shape[0] - int(min_segment_len * target_sample_rate)
            if max_possible_start > 0:
                # Generate more potential start points than needed to increase chances of finding non-overlapping, valid segments
                for _ in range(num_segments * 5): 
                    potential_start_times.append(random.randint(0, max_possible_start))
            potential_start_times = sorted(list(set(potential_start_times))) # Remove duplicates and sort

            for start_sample in potential_start_times:
                if len(segments) >= num_segments:
                    break

                # Check for overlap with already selected segments
                is_overlapping = False
                for prev_start, prev_end in selected_segments_info:
                    # Check if the current potential segment (even a minimal one) would overlap with a selected segment
                    if not (start_sample >= prev_end or start_sample + int(min_segment_len * target_sample_rate) <= prev_start):
                        is_overlapping = True
                        break
                if is_overlapping:
                    continue

                # Determine segment duration
                segment_duration = random.uniform(min_segment_len, max_segment_len)
                segment_length_samples = int(segment_duration * target_sample_rate)

                # Adjust segment length to not exceed audio bounds
                end_sample = start_sample + segment_length_samples
                if end_sample > waveform.shape[0]:
                    end_sample = waveform.shape[0]
                    segment_length_samples = end_sample - start_sample
                
                # Ensure segment length is at least min_segment_len after adjustments
                if segment_length_samples < int(min_segment_len * target_sample_rate):
                    continue

                current_segment = waveform[start_sample:end_sample]

                # RMS-based filtering
                if calculate_rms(current_segment) > rms_threshold:
                    segments.append(current_segment)
                    selected_segments_info.append((start_sample, end_sample))
                    selected_segments_info.sort() # Keep sorted for efficient overlap checking
                # else:
                    # print(f"Segment from {audio_path} (start={start_sample/target_sample_rate:.2f}s, end={end_sample/target_sample_rate:.2f}s) RMS below threshold.")

    except Exception as e:
        error_message = str(e)
        if "Unable to process >4GB files" in error_message:
            raise AudioFileTooLargeError(f"File {audio_path} is too large: {error_message}")
        print(f"Failed to load or process {audio_path}: {e}")
        return None
    
    if not segments:
        return None
    return pad_waveforms(segments)

def pad_waveforms(waveforms):
    """Pads a list of waveforms to the same length and stacks them into a tensor."""
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = [
        torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms
    ]
    return torch.stack(padded_waveforms)

# ----------------------------- #
#      Language Detection       #
# ----------------------------- #

def detect_language(model, tokenizer, input_features, possible_languages=None):
    """Detects the language of the input audio features."""
    # Get all language tokens (e.g., "<|en|>", "<|fa|>")
    all_language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]

    # Filter language tokens if specific possible_languages are provided
    if possible_languages is not None:
        language_tokens_to_consider = [t for t in all_language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens_to_consider) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')
    else:
        language_tokens_to_consider = all_language_tokens

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens_to_consider)
    decoder_input_ids = torch.tensor([[50258]] * input_features.shape[0]).to(input_features.device) # 50258 is the <|startoflm|> token
    logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

    # Mask out non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).cpu()

    results = []
    for i in range(logits.shape[0]):
        # Map token IDs back to language codes (e.g., "en", "fa")
        lang_probs = {
            lang[2:-2]: output_probs[i, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens_to_consider)
        }
        # Find the language with the highest probability
        detected_lang = max(lang_probs, key=lang_probs.get)
        detected_lang_prob = lang_probs[detected_lang]
        
        results.append({
            "detected_lang": detected_lang,
            "detected_lang_prob": detected_lang_prob,
            "all_lang_probs": lang_probs # Optionally include all probabilities
        })
    return results

# ----------------------------- #
#        Helper Functions       #
# ----------------------------- #

def append_to_manifest(filepath, records):
    """Appends a list of records to a JSON lines file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def get_audio_duration_ffmpeg(audio_path):
    """Gets audio duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Could not get duration for {audio_path} using ffprobe: {e}")
        return None

def split_audio_file(original_audio_path):
    """
    Splits an audio file into two parts using ffmpeg.
    Returns a list of paths to the two new split files.
    """
    print(f"Splitting large audio file: {original_audio_path}")
    
    # Get duration of the original file
    duration = get_audio_duration_ffmpeg(original_audio_path)
    if duration is None:
        print(f"Cannot split {original_audio_path}: could not determine duration.")
        return []

    midpoint = duration / 2
    
    base, ext = os.path.splitext(original_audio_path)
    part1_path = f"{base}_0{ext}"
    part2_path = f"{base}_1{ext}"

    # Command to split the audio into two parts
    # Part 1: from start to midpoint
    # Part 2: from midpoint to end
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(part1_path), exist_ok=True)

        # Split command for part 1
        cmd1 = [
            'ffmpeg', '-i', original_audio_path,
            '-t', str(midpoint), # duration for the first part
            '-c', 'copy', # copy stream without re-encoding
            part1_path
        ]
        subprocess.run(cmd1, check=True, capture_output=True)
        print(f"Created split part 1: {part1_path}")

        # Split command for part 2
        cmd2 = [
            'ffmpeg', '-i', original_audio_path,
            '-ss', str(midpoint), # start from midpoint
            '-c', 'copy', # copy stream without re-encoding
            part2_path
        ]
        subprocess.run(cmd2, check=True, capture_output=True)
        print(f"Created split part 2: {part2_path}")

        return [part1_path, part2_path]

    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio file {original_audio_path} with ffmpeg: {e.stderr.decode()}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during audio splitting: {e}")
        return []

# ----------------------------- #
#            Main               #
# ----------------------------- #

def main(input_manifest_path, output_manifest_path, batch_size, save_iterations, num_segments, min_segment_len, max_segment_len, rms_threshold, whisper_model_path=None):
    """Main function to run the language detection pipeline for long audio files."""
    # Load manifest data from JSON lines file
    manifest_data = []
    print(f"Reading manifest from: {input_manifest_path}")
    try:
        with open(input_manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'audio_filepath' not in record:
                        print(f"Skipping line, missing 'audio_filepath': {line.strip()}")
                        continue
                    manifest_data.append(record)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Input manifest file not found at {input_manifest_path}")
        return

    if not manifest_data:
        print("No valid audio filepaths found in the manifest.")
        return

    audio_paths = [rec['audio_filepath'] for rec in manifest_data]
    total_audio_files = len(audio_paths)

    # Load model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor, model, tokenizer = load_model_and_tokenizer(whisper_model_path)
    model = model.to(device)
    model.eval()

    # Read existing output manifest to identify already processed files
    processed_audio_filepaths = set()
    if os.path.exists(output_manifest_path):
        print(f"Reading existing output manifest from: {output_manifest_path}")
        with open(output_manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'audio_filepath' in record:
                        processed_audio_filepaths.add(record['audio_filepath'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in output manifest: {line.strip()}")
    
    # Filter out already processed files from the input manifest
    audio_paths_to_process = [path for path in audio_paths if path not in processed_audio_filepaths]
    total_to_process = len(audio_paths_to_process)

    if total_to_process == 0:
        print("All audio files in the input manifest have already been processed and are present in the output manifest. Exiting.")
        return

    # Open output file in append mode if it exists, otherwise create a new one
    file_mode = 'a' if os.path.exists(output_manifest_path) and processed_audio_filepaths else 'w'
    with open(output_manifest_path, file_mode, encoding='utf-8') as f:
        if file_mode == 'w':
            pass # File is already cleared if opened in 'w' mode

    records_to_write = []
    print(f"Processing {total_to_process} new audio files...")
    pbar = tqdm(audio_paths_to_process, desc="Audio Files")

    for audio_file_path in pbar:
        current_audio_paths_to_process = [audio_file_path]
        original_record = next((rec for rec in manifest_data if rec['audio_filepath'] == audio_file_path), None)
        
        all_detection_results_for_original_file = []
        processed_successfully = False

        # Use a while loop to process current_audio_paths_to_process,
        # allowing new split paths to be added and processed within the same outer loop iteration.
        idx = 0
        while idx < len(current_audio_paths_to_process):
            current_path = current_audio_paths_to_process[idx]
            idx += 1 # Move to the next path

            try:
                segments_waveform = load_and_preprocess_audio_segments(
                    current_path,
                    target_sample_rate=16000,
                    num_segments=num_segments,
                    min_segment_len=min_segment_len,
                    max_segment_len=max_segment_len,
                    rms_threshold=rms_threshold
                )

                if segments_waveform is None:
                    continue

                # Process segments in batches
                file_detection_results = []
                with torch.inference_mode():
                    for i in range(0, segments_waveform.shape[0], batch_size):
                        batch_segments = segments_waveform[i:i + batch_size]
                        
                        inputs = processor(batch_segments.numpy(), sampling_rate=16000, return_tensors="pt")
                        input_features = inputs.input_features.to(device, dtype=torch.float16)
                        
                        detection_results = detect_language(model, tokenizer, input_features)
                        file_detection_results.extend(detection_results)
                
                if file_detection_results:
                    all_detection_results_for_original_file.extend(file_detection_results)
                    processed_successfully = True

            except AudioFileTooLargeError:
                print(f"Caught AudioFileTooLargeError for {current_path}. Attempting to split and re-process.")
                split_paths = split_audio_file(current_path)
                if split_paths:
                    # Add split paths to be processed in the current iteration
                    current_audio_paths_to_process.extend(split_paths)
                else:
                    print(f"Failed to split {current_path}. Skipping.")
                # Mark the original large file as processed to avoid re-attempting it directly
                # This is important if the original audio_file_path was the one that failed
                processed_audio_filepaths.add(audio_file_path)
                continue # Continue to the next path in current_audio_paths_to_process

            except Exception as e:
                print(f"An unexpected error occurred while processing {current_path}: {e}")
                continue

        if processed_successfully and all_detection_results_for_original_file:
            # Aggregate results for the original audio file (or its split parts)
            aggregated_lang_probs = {}
            detected_langs_count = {}

            for result in all_detection_results_for_original_file:
                for lang, prob in result["all_lang_probs"].items():
                    aggregated_lang_probs[lang] = aggregated_lang_probs.get(lang, 0.0) + prob
                
                detected_lang = result["detected_lang"]
                detected_langs_count[detected_lang] = detected_langs_count.get(detected_lang, 0) + 1

            num_processed_segments = len(all_detection_results_for_original_file)
            if num_processed_segments > 0:
                for lang in aggregated_lang_probs:
                    aggregated_lang_probs[lang] /= num_processed_segments

            final_detected_lang = max(detected_langs_count, key=detected_langs_count.get) if detected_langs_count else None
            final_detected_lang_prob = aggregated_lang_probs.get(final_detected_lang, 0.0) if final_detected_lang else 0.0

            if original_record:
                updated_record = original_record.copy()
                updated_record.update({
                    "lang": final_detected_lang,
                    "lang_prob": final_detected_lang_prob,
                    "all_lang_probs": aggregated_lang_probs
                })
                # Add duration if available in the original record
                if 'duration' in original_record:
                    updated_record['duration'] = original_record['duration']
                records_to_write.append(updated_record)
                processed_audio_filepaths.add(audio_file_path) # Mark original file as processed

        # Check if it's time to save progress
        if save_iterations > 0 and len(records_to_write) >= save_iterations:
            pbar.set_postfix_str(f"Saving progress... ({len(records_to_write)} records)")
            append_to_manifest(output_manifest_path, records_to_write)
            records_to_write = [] # Reset buffer

    # Final save for any remaining records
    if records_to_write:
        append_to_manifest(output_manifest_path, records_to_write)

    print(f"Language detection for long audios completed. Results saved to {output_manifest_path}")

# ----------------------------- #
#        Run the Script         #
# ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform language detection on long audio files by sampling segments.")
    parser.add_argument('--input_manifest', type=str, required=True, help='Path to the input manifest file (JSON lines).')
    parser.add_argument('--output_manifest', type=str, required=True, help='Path to save the output manifest file with language detection results.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing segments.')
    parser.add_argument('--save_iterations', type=int, default=64, help='Save progress every N records. If 0, saves only at the end. (default: 0)')
    parser.add_argument('--num_segments', type=int, default=10, help='Number of random segments to extract from each long audio.')
    parser.add_argument('--min_segment_len', type=int, default=10, help='Minimum length of an audio segment in seconds.')
    parser.add_argument('--max_segment_len', type=int, default=30, help='Maximum length of an audio segment in seconds.')
    parser.add_argument('--rms_threshold', type=float, default=0.01, help='RMS energy threshold for filtering speech segments. Segments with RMS below this value will be discarded.')
    parser.add_argument('--whisper_model_path', type=str, default=None, help='Optional local path to a Whisper model directory. If not provided or not found, the script falls back to the Hugging Face model openai/whisper-large-v3.')

    args = parser.parse_args()

    main(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        batch_size=args.batch_size,
        save_iterations=args.save_iterations,
        num_segments=args.num_segments,
        min_segment_len=args.min_segment_len,
        max_segment_len=args.max_segment_len,
        rms_threshold=args.rms_threshold,
        whisper_model_path=args.whisper_model_path
    )
