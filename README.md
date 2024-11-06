# subsplit

**subsplit** is a library for making transcriptions more suitable for display as
subtitles.

## How to use

The package accepts Whisper-like segments, in the form:

```json
{
    "segments": [
        {
            "words": [
                {"word": "...", "start": 0.0, "end": 1.0},
                // ...
            ],
            // ...
        },
        // ...
    ]
}
```

and re-splits the segments to be better suited for subtitle display. Output
defaults to SRT format, but can also be JSON.


## How to run

```bash
$ git clone https://github.com/kwikwag/subsplit.git
$ pip install -e subsplit
$ python -m subsplit whisper-output.json --out subtitles.srt
```
