#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from pydub.utils import mediainfo
import subprocess


AUDIO_EXTENSIONS = {
    ".webm",
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".flac",
}


def get_duration(filepath):
    """
    Get duration using mediainfo.
    Returns None if unavailable.
    """

    try:
        info = mediainfo(str(filepath))

        duration = info.get("duration")

        if not duration:
            return None

        return float(duration)

    except Exception as e:
        print(
            f"Failed reading duration {filepath}: {e}"
        )
        return None



def process_file(
    filepath,
    dataset_dir,
    max_duration,
    output_audio_dir,
):
    """
    Process one audio file.
    Returns manifest entries.
    """

    filepath = Path(filepath)

    video_id = filepath.stem

    if len(video_id) != 11:
        print(
            f"Skipping invalid video id: {filepath}"
        )
        return []


    duration = get_duration(filepath)

    if duration is None:
        print(
            f"Skipping invalid duration: {filepath}"
        )
        return []


    channel_id = filepath.parent.name


    # No split required
    if duration <= max_duration:

        return [
            {
                "audio_filepath": str(
                    filepath.relative_to(dataset_dir)
                ),
                "duration": duration,
                "channel_id": channel_id,
                "video_id": video_id,
                "start": 0.0,
                "end": duration,
            }
        ]


    if output_audio_dir:
        output_dir = Path(output_audio_dir)
    else:
        output_dir = filepath.parent


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    records = []

    chunk_idx = 1
    start_sec = 0.0

    # Cut chunks directly with ffmpeg instead of decoding the whole
    # file into memory. Avoids OOM kills and the pydub >4GB wave limit.
    while start_sec < duration:

        end_sec = min(start_sec + max_duration, duration)


        output_path = (
            output_dir
            / f"{video_id}_{chunk_idx}.mp3"
        )

        tmp_path = output_path.with_suffix(
            ".tmp.mp3"
        )


        try:

            subprocess.run(
                [
                    "ffmpeg", "-y", "-nostdin",
                    "-ss", str(start_sec),
                    "-i", str(filepath),
                    "-t", str(end_sec - start_sec),
                    "-vn", "-acodec", "libmp3lame", "-b:a", "192k",
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Replace existing file atomically
            tmp_path.replace(output_path)


        except Exception as e:

            print(
                f"Failed exporting {output_path}: {e}"
            )

            if tmp_path.exists():
                tmp_path.unlink()

            return records


        records.append(
            {
                "audio_filepath": str(
                    output_path.relative_to(dataset_dir)
                ),
                "duration": end_sec - start_sec,
                "channel_id": channel_id,
                "video_id": video_id,
                "start": start_sec,
                "end": end_sec,
            }
        )


        start_sec = end_sec
        chunk_idx += 1


    return records



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=str,
    )

    parser.add_argument(
        "--output",
        default="manifest.jsonl",
    )

    parser.add_argument(
        "--max_duration",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--output_audio_dir",
        default=None,
    )

    args = parser.parse_args()


    dataset_dir = Path(
        args.dataset_dir
    ).resolve()


    # Group files by channel
    channels = defaultdict(list)


    for filepath in dataset_dir.rglob("*"):

        if filepath.suffix.lower() in AUDIO_EXTENSIONS:

            channel = filepath.parent.name

            channels[channel].append(filepath)


    # Sort everything
    for channel in channels:
        channels[channel].sort()


    sorted_channels = sorted(
        channels.keys()
    )


    total_files = sum(
        len(files)
        for files in channels.values()
    )


    print(
        f"Found {total_files} audio files "
        f"in {len(sorted_channels)} channels"
    )


    # Resume support: skip files already present in an existing manifest
    processed = set()

    if Path(args.output).exists():

        with open(args.output, "r", encoding="utf-8") as fin:

            for line in fin:
                try:
                    rec = json.loads(line)
                    processed.add((rec["channel_id"], rec["video_id"]))
                except Exception:
                    continue

        print(f"Resuming: {len(processed)} files already in manifest")


    try:

        with open(
            args.output,
            "a",
            encoding="utf-8",
        ) as fout:


            for channel in tqdm(
                sorted_channels,
                desc="Channels",
                position=0,
            ):

                files = channels[channel]


                for filepath in tqdm(
                    files,
                    desc=f"{channel}",
                    position=1,
                    leave=False,
                ):

                    if (channel, filepath.stem) in processed:
                        continue

                    records = process_file(
                        filepath,
                        dataset_dir,
                        args.max_duration,
                        args.output_audio_dir,
                    )


                    for record in records:

                        fout.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )


    except KeyboardInterrupt:

        print(
            "\nInterrupted by user. "
            "Manifest closed safely."
        )


    except Exception as e:

        print(
            f"\nUnexpected error: {e}"
        )

        raise



if __name__ == "__main__":
    main()