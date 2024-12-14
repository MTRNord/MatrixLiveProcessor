# Matrix Live Processor

Small script which ensures a somewhat consistent matrix live experience.

## Features

- Audio normalization to -16 LUFS
- Optional Whisper subtitle generation
- Whisper subtitle based show notes

## Requirements

- Python 3.12+
- ffmpeg
- ffmpeg-normalize
- whisper

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python matrix_live_processor.py ./input.mp4 ./output.mp4 --use-whisper --model-name medium --language en
```

### Arguments

- `input_file`: The input file to process
- `output_file`: The output file to write to (video filename)
- `--use-whisper`: Enable whisper subtitle generation
- `--model-name`: The whisper model to use (default: `medium`). Check whispercpp for available models.
- `--language`: The language to use for whisper subtitle generation (default: `en`). Check whispercpp for available
  languages.
- `--speaker`: The speaker to use for whisper subtitle generation. Should be the name of the speaker in the video.
- `--force`: Force overwrite of the output file if it already exists.

## Note

For best performance you should build the whisper lib manually. See https://github.com/abdeladim-s/pywhispercpp for
details.

## License

Apache 2.0