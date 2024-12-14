import argparse
from pathlib import Path

import webvtt
from ffmpeg import FFmpeg
from ffmpeg_normalize import FFmpegNormalize
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pywhispercpp.model import Model, Segment
from pywhispercpp.utils import to_timestamp

initial_prompt_template = """
Please extract the following information from the episode transcript and format it as shownotes for a podcast description:

Transcript:
{transcript}

Please ensure to only output valid Markdown content.
"""


def output_vtt(
    segments: list[Segment],
    output_file_path: Path,
) -> None:
    """
    Creates a vtt file from a list of segments

    :param segments: list of segments
    :param output_file_path:  path of the file
    :return: path of the file

    :return: Absolute path of the file
    """

    with open(output_file_path, "w") as file:
        file.write("WEBVTT\n\n")
        for seg in segments:
            file.write(
                f"{to_timestamp(seg.t0, separator='.')} --> {to_timestamp(seg.t1, separator='.')}\n"
            )
            file.write(f"{seg.text}\n\n")


# Define a function to normalize audio to -16 LUFS
def normalize_audio(input_file: Path, output_file: Path) -> None:
    normalizer = FFmpegNormalize(
        target_level=-16.0,
        true_peak=-1.0,
        loudness_range_target=3.5,
    )
    normalizer.add_media_file(str(input_file), str(output_file))
    normalizer.run_normalization()


# Define a function to generate subtitles using Whisper
def generate_subtitles(
    mp3_file: Path,
    vtt_file: Path,
    model_name: str = "medium",
    language: str = "en",
) -> None:
    model = Model(
        f"{model_name}.{language}",
        print_progress=True,
        print_realtime=True,
        n_threads=6,
    )

    def on_segment(segment: Segment):
        logger.info(f"Segment: {segment}")

    segments = model.transcribe(str(mp3_file), new_segment_callback=on_segment)

    # Create VTT file
    output_vtt(
        segments,
        vtt_file,
    )


def escape_curly_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def read_vtt_file(vtt_file: Path) -> str:
    logger.info("Reading transcript from file:", vtt_file)
    vtt = webvtt.read(str(vtt_file))
    transcript = " ".join([caption.text for caption in vtt])
    logger.info("Transcript length:", len(transcript))
    return transcript


# Define a function to create show notes based on Whisper subtitles
def create_show_notes(vtt_file: Path, show_notes_file: Path) -> None:
    transcript = read_vtt_file(vtt_file)

    # Create promt using PromptTemplate by combining the transcript and above defined prompt template
    prompt = PromptTemplate(
        initial_prompt_template,
        input_variables=["transcript"],
    )

    query = prompt.format(transcript=transcript)
    logger.info("Prompt query:", query)

    # Initialize the LLM model
    local_llm = "phi3"

    logger.info(f"Initializing LLM for extraction: {local_llm}")
    model = ChatOllama(model=local_llm, temperature=0.1)

    # Invoke the LLM to get the response
    response = model.invoke(query)
    raw_content = response.content
    logger.info("Initial Response:", raw_content)

    # Write the show notes to the file
    with open(show_notes_file, "w") as file:
        file.write(raw_content)


def create_mp3(input_file: Path, output_file: Path) -> Path:
    mp3_output = output_file.with_suffix(".mp3")
    logger.info(f"Converting audio to MP3: {mp3_output}")

    # Convert the audio to MP3
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(input_file))
        .output(str(mp3_output), {"codec:a": "libmp3lame"})
    )
    ffmpeg.execute()

    return mp3_output


def convert_vtt_to_ass(vtt_file: Path, ass_file: Path) -> None:
    ffmpeg = FFmpeg().input(str(vtt_file)).output(str(ass_file))
    ffmpeg.execute()


def overlay_subtitles(video_file: Path, ass_file: Path, output_file: Path) -> None:
    ffmpeg = (
        FFmpeg().input(str(video_file)).output(str(output_file), vf=f"ass={ass_file}")
    )
    ffmpeg.execute()


# Define the main function to process the input file
def main(
    input_file: Path,
    output_file: Path,
    vtt_file: Path,
    show_notes_file: Path,
    use_whisper: bool,
    model_name: str = None,
    language: str = None,
) -> None:
    # Normalize the audio
    logger.info("Normalizing audio...")
    normalize_audio(input_file, output_file)

    # Generate subtitles if the flag is set
    if use_whisper:
        logger.info("Generating subtitles using Whisper...")
        logger.info("Generating mp3 file...")
        mp3 = create_mp3(output_file, output_file)
        logger.info("Mp3 file generated...")
        logger.info("Generating subtitles...")
        generate_subtitles(mp3, vtt_file, model_name, language)
        logger.info("Subtitles generated...")
        logger.info("Add subtitles to video file...")
        ass_file = output_file.with_suffix(".ass")
        convert_vtt_to_ass(vtt_file, ass_file)
        overlay_subtitles(output_file, ass_file, output_file)
        logger.info("Added subtitles to video file...")
        logger.info("Creating show notes...")
        create_show_notes(vtt_file, show_notes_file)


# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Matrix Live videos.")
    parser.add_argument("input_file", type=Path, help="Path to the input file")
    parser.add_argument("output_file", type=Path, help="Path to the output file")
    parser.add_argument(
        "--use-whisper", action="store_true", help="Enable Whisper subtitle generation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="medium",
        help="Whisper model name",
        choices=["small", "medium", "large"],
        action="store",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the video. Default is English. Use ISO 639-1 code.",
        action="store",
    )
    # Force flag for overwriting the output file
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite the output files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if the output file already exists
    if not args.force and (
        args.output_file.exists()
        or args.output_file.with_suffix(".vtt").exists()
        or args.output_file.with_suffix(".md").exists()
        or args.output_file.with_suffix(".mp3").exists()
    ):
        logger.error("Output file already exists. Use --force to overwrite.")
        exit(1)

    # Check if the input file exists
    if not args.input_file.exists():
        logger.error("Input file does not exist.")
        exit(1)

    main(
        args.input_file,
        args.output_file,
        args.output_file.with_suffix(".vtt"),
        args.output_file.with_suffix(".md"),
        args.use_whisper,
        args.model_name,
        args.language,
    )
