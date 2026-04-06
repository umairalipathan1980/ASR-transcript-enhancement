"""Example for the TranscriptEnhancer software component."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from gaik.software_components.enhance_transcript import TranscriptEnhancer, get_openai_config


def main() -> None:
    config = get_openai_config(use_azure=True)
    enhancer = TranscriptEnhancer(api_config=config)
    transcript_file = Path("Sample9.txt")
    if transcript_file.exists():
        file_result = enhancer.enhance_file(
            transcript_file,
            generate_summary=True,
            diff_chunks=True,
        )
        print("Enhanced transcript from file:")
        print(file_result.enhanced_text)
        print()
        print("JSON result for file input:")
        print(file_result.model_dump_json(indent=2))
    else:
        print(f"Skipping file example because {transcript_file} was not found.")


if __name__ == "__main__":
    main()