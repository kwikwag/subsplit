"""Test module for subsplit"""

from subsplit.subsplit import get_words_total_len, split_segments_for_subtitles


def test_get_words_total_len():
    """Test function for `get_words_total_len`"""
    for input_args, expected_output in (
        (
            (
                (
                    {
                        "word": "Hello, ",
                    },
                    {
                        "word": "world. ",
                    },
                    {
                        "word": "How ",
                    },
                    {
                        "word": "you ",
                    },
                    {
                        "word": "doin'?",
                    },
                ),
            ),
            28,
        ),
    ):
        actual_output = get_words_total_len(*input_args)
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_split_segments_for_subtitles():
    """Test function for `split_segments_for_subtitles`"""
    force_break_duration = 0.5
    allow_break_duration = 0.2
    lipsum = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec non mi "
        "eget risus luctus pulvinar ac sed dui. Donec feugiat interdum auctor. "
        "Vestibulum sit amet massa eget justo sagittis vehicula. Donec "
        "hendrerit, neque vel iaculis auctor, libero quam suscipit mauris, non "
        "varius mauris dui nec neque. In vulputate sollicitudin ex, nec "
        "aliquam metus. Nam eu tortor semper, pulvinar justo sit amet, rhoncus "
        "nisi. Pellentesque varius dapibus quam vitae volutpat. Fusce sit amet "
        "nisl in sem hendrerit maximus at elementum arcu. Aenean at sem vel "
        "erat aliquet pellentesque eget ut nunc. Quisque ultrices sem nibh, "
        "quis lobortis nibh vestibulum id. Vivamus placerat mi eget mattis "
        "porta. Aenean nisi elit, laoreet in velit in, laoreet sollicitudin "
        "quam. Mauris luctus quam eu leo auctor, in feugiat justo aliquam. "
        "Nulla facilisi."
    )
    expected_output = (
        # <--               50 characters              -->
        ##################################################
        "Lorem ipsum dolor sit amet, ",
        "consectetur adipiscing elit. ",
        "Donec non mi eget risus luctus pulvinar ac sed dui. ",
        "Donec feugiat interdum auctor. ",
        "Vestibulum sit amet massa eget justo sagittis vehicula. ",
        "Donec hendrerit, neque vel iaculis auctor, ",
        "libero quam suscipit mauris, ",
        "non varius mauris dui nec neque. ",
        "In vulputate sollicitudin ex, nec aliquam metus. ",
        "Nam eu tortor semper, pulvinar justo sit amet, ",
        "rhoncus nisi. ",
        "Pellentesque varius dapibus quam vitae volutpat. ",
        "Fusce sit amet nisl in sem hendrerit maximus at elementum arcu. ",
        "Aenean at sem vel erat aliquet pellentesque eget ut nunc. ",
        "Quisque ultrices sem nibh, ",
        "quis lobortis nibh vestibulum id. ",
        "Vivamus placerat mi eget mattis porta. ",
        "Aenean nisi elit, laoreet in velit in, ",
        "laoreet sollicitudin quam. ",
        "Mauris luctus quam eu leo auctor, ",
        "in feugiat justo aliquam. ",
        "Nulla facilisi.",
    )
    eps = 1e-7
    words = lipsum.split(" ")

    def word_duration(word, idx):
        # each word duration is slightly smaller than the previous
        # this makes all gaps uneven, such that shorter gaps appear earlier,
        # so that the algorithm becomes predictable for this test
        duration = 1 - idx * eps
        if "." in word:
            # leave large gap after
            duration -= force_break_duration
        elif "," in word:
            # leave short gap after
            duration -= allow_break_duration
        return duration

    words = [
        {
            "word": word + " ",
            "start": idx,
            "end": idx + word_duration(word, idx),
        }
        for idx, word in enumerate(words)
    ]
    # remove trailing space for last word
    words[-1]["word"] = words[-1]["word"][:-1]
    segments = [{"words": words, "start": words[0]["start"], "end": words[-1]["end"]}]
    kwargs = dict(
        segments=segments,
        force_break_duration=force_break_duration,
        allow_break_duration=allow_break_duration,
        extra_time=0.1,
        max_len=50,
    )
    segments = split_segments_for_subtitles(**kwargs)
    actual_output = tuple("".join(word["word"] for word in segment["words"]) for segment in segments)
    assert actual_output == expected_output, dict(input=kwargs, actual=actual_output, expected=expected_output)
