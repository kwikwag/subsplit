"""Test module for subsplit"""
from subsplit.subsplit import (
    Segment,
    add_silences,
    cum_sum,
    diff,
    find_sufficient_silence,
    get_words_total_len,
    max_threshold_min_split,
    optimal_k_partition,
)


def test_diff():
    """Test function for `diff`"""
    for input_args, expected_output in (
        (((0,),), ()),
        (((0, 0),), (0,)),
        (((0, 1, 3),), (1, 2)),
        (((0, 1, -1),), (1, -2)),
    ):
        actual_output = tuple(diff(*input_args))
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_add_silences(eps=1e-7):
    """Test function for `add_silences`"""
    for words, expected_output in (
        (
            (
                {
                    "word": "Hello,",
                    "start": 0.1,
                    "end": 1.1,
                },
                {
                    "word": "world.",
                    "start": 1.2,
                    "end": 2.2,
                },
                {
                    "word": "How",
                    "start": 3.7,
                    "end": 4.7,
                },
                {
                    "word": "you",
                    "start": 4.8,
                    "end": 5.8,
                },
                {
                    "word": "doin'?",
                    "start": 5.9,
                    "end": 6.9,
                },
            ),
            (
                {"word": "Hello,", "start": 0.1, "end": 1.1, "preceding_silence": 0.1},
                {"word": "world.", "start": 1.2, "end": 2.2, "preceding_silence": 0.1},
                {"word": "How", "start": 3.7, "end": 4.7, "preceding_silence": 1.5},
                {"word": "you", "start": 4.8, "end": 5.8, "preceding_silence": 0.1},
                {"word": "doin'?", "start": 5.9, "end": 6.9, "preceding_silence": 0.1},
            ),
        ),
    ):
        words_copy = tuple(dict(word) for word in words)
        segments = [Segment(words=words)]
        add_silences(segments)
        words = segments[0]["words"]
        assert all(
            (
                abs(actual_word["preceding_silence"] - expected_word["preceding_silence"]) < eps
                and actual_word["start"] == expected_word["start"]
                and actual_word["end"] == expected_word["end"]
                and actual_word["word"] == expected_word["word"]
            )
            for actual_word, expected_word in zip(words, expected_output)
        ), dict(input=words_copy, actual=words, expected=expected_output)


def test_get_words_total_len():
    """Test function for `get_words_total_len`"""
    for input_args, expected_output in (
        (
            (
                (
                    {"word": "Hello, ", "preceding_silence": 0.1},
                    {"word": "world. ", "preceding_silence": 0.1},
                    {"word": "How ", "preceding_silence": 1.5},
                    {"word": "you ", "preceding_silence": 0.1},
                    {"word": "doin'?", "preceding_silence": 0.1},
                ),
            ),
            28,
        ),
    ):
        actual_output = get_words_total_len(*input_args)
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_optimal_k_partition():
    """Test function for `optimal_k_partition`"""
    for input_args, expected_output in (
        (
            ((1, 1, 1), 3),
            (1, 2),
        ),
        (
            ((1, 1, 2), 2),
            (2,),
        ),
        (
            ((1, 2, 3, 4, 5, 8, 7, 6, 8, 1), 3),
            (5, 7),
        ),
    ):
        actual_output = optimal_k_partition(*input_args)
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_max_threshold_min_split():
    """Test function for `max_threshold_min_split`"""
    for input_args, expected_output in (
        (((), (), 1), None),
        (((1,), (0,), 1), 1),
        (((1,), (1,), 1), 1),
        (((1,), (2,), 1), None),
        (((-1,), (1,), 1), -1),
        (((0, 1, 0, 1, 0), (1, 2, 1, 2, 1), 2), 0),
    ):
        actual_output = max_threshold_min_split(*input_args)
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_find_sufficient_silence():
    """Test function for `find_sufficient_silence`"""
    for input_args, expected_output in (
        (
            (
                (
                    {"word": "Hello, ", "preceding_silence": 0.1},
                    {"word": "world. ", "preceding_silence": 0.1},
                    {"word": "How ", "preceding_silence": 1.5},
                    {"word": "you ", "preceding_silence": 0.1},
                    {"word": "doin'?", "preceding_silence": 0.1},
                ),
                0.2,
                14,
                False,
            ),
            1.5,
        ),
        (
            (
                (
                    {"word": "Hello, ", "preceding_silence": 0},
                    {"word": "world. ", "preceding_silence": 0.1},
                    {"word": "How ", "preceding_silence": 1.5},
                    {"word": "you ", "preceding_silence": 0.1},
                    {"word": "doin'?", "preceding_silence": 0.1},
                ),
                0.2,
                13,
                False,
            ),
            None,
        ),
        (
            (
                (
                    {"word": "Hello, ", "preceding_silence": 0},
                    {"word": "world. ", "preceding_silence": 0.1},
                    {"word": "How ", "preceding_silence": 1.5},
                    {"word": "you ", "preceding_silence": 0.1},
                    {"word": "doin'?", "preceding_silence": 0.1},
                ),
                0.2,
                13,
                True,
            ),
            1.5,
        ),
    ):
        actual_output = find_sufficient_silence(*input_args)
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)


def test_cum_sum():
    """Test function for `cum_sum`"""
    for input_args, expected_output in (
        (((),), ()),
        (((0,),), (0,)),
        (((1,),), (1,)),
        (((1, 2),), (1, 3)),
        (((1, -1),), (1, 0)),
        (((-1, 1),), (-1, 0)),
        (((1, 2, 3, 4),), (1, 3, 6, 10)),
    ):
        actual_output = tuple(cum_sum(*input_args))
        assert actual_output == expected_output, dict(input=input_args, actual=actual_output, expected=expected_output)
