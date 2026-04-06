# Command: python whisperX.py <input_dir> <output_dir>
import argparse
from pathlib import Path

import torch
import whisperx

ASR_MODEL = "large"
LANGUAGE = "fi"  # Set to "en" or "fi" to force; None to auto-detect
TASK = "transcribe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32"
ASR_OPTIONS = {
    "beam_size": 15,
    "patience": 2.0,
    "condition_on_previous_text": True,
    "initial_prompt": None,
}
VAD_OPTIONS = {
    "vad_onset": 0.40,
    "vad_offset": 0.40
}


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe .mp3 audio files with WhisperX"
    )
    parser.add_argument("input_dir", help="Folder containing .mp3 files")
    parser.add_argument("output_dir", help="Folder where transcript .txt files are saved")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    print(f"using device: {DEVICE}")
    model = whisperx.load_model(
        ASR_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
        task=TASK,
        asr_options=ASR_OPTIONS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"
    ]
    if not audio_paths:
        raise FileNotFoundError(f"No .mp3 files found in: {input_dir}")

    for audio_path in audio_paths:
        print(f"Transcribing: {audio_path.name}")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio)
        output_path = output_dir / f"{audio_path.stem}.txt"
        text = result.get("text", "")
        if not text and result.get("segments"):
            text = " ".join(seg.get("text", "").strip() for seg in result["segments"]).strip()
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
