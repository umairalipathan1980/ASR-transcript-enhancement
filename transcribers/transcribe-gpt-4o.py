# Command: python transcribe-gpt-4o.py <input_dir> <output_dir> [--language <code>]
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

class Transcriber:
    def __init__(self, api_key=None, prompt=None):
        load_dotenv()
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.client = OpenAI(api_key=api_key)
        self.prompt = prompt
        self.last_transcript_obj = None

    def transcribe_file(self, audio_path, language="fi"):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path.name}")
        with open(audio_path, "rb") as audio_file:
            request = {
                "model": "gpt-4o-transcribe",
                "file": audio_file,
                "language": language,
                "temperature": 0,
                "response_format": "json",
                "include": ["logprobs"],
            }

            transcript_obj = self.client.audio.transcriptions.create(**request)

        self.last_transcript_obj = transcript_obj

        return transcript_obj.text if hasattr(transcript_obj, "text") else transcript_obj


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe .m4a audio files with GPT-4o Transcribe"
    )
    parser.add_argument("input_dir", help="Folder containing .m4a files")
    parser.add_argument("output_dir", help="Folder where transcript .txt files are saved")
    parser.add_argument(
        "--language",
        default="fi",
        help="Language code to pass to the transcription API, default is fi",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.glob("*.m4a"))
    if not audio_files:
        print(f"No .m4a files found in: {input_dir}")
        return

    transcriber = Transcriber()

    print(f"Found {len(audio_files)} .m4a file(s).")
    print(f"Output folder: {output_dir}")

    success = 0
    failed = 0

    for audio_path in audio_files:
        output_path = output_dir / f"{audio_path.stem}.txt"
        try:
            transcript_text = transcriber.transcribe_file(
                audio_path, language=args.language
            )
            output_path.write_text(str(transcript_text), encoding="utf-8")
            print(f"Saved: {output_path.name}")
            success += 1
        except Exception as exc:
            print(f"Failed: {audio_path.name} -> {exc}")
            failed += 1

    print(f"Done. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
