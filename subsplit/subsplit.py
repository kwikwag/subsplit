"""Split transcription segments in a wise way for display as subtitles."""

import re
import sys
import typing
from collections import defaultdict
from functools import partial


class Word(typing.TypedDict):
    """Represents a word dict"""

    word: str
    start: float
    end: float


Words = typing.Iterable[Word]


class Segment(typing.TypedDict):
    """Represents a segment dict"""

    words: typing.Iterable[Word]
    start: typing.Optional[int]
    end: typing.Optional[int]


Segments = typing.Iterable[Segment]


# see: https://www.clevercast.com/bbc-subtitling-guidelines
# at least 1.5sec between subtitles -> anything longer we split
# no longer than 2 lines -> use estimate for number of chars per line, and split according to that
# min_duration_per_word
def split_segments_for_subtitles(
    segments,
    force_break_duration: float = 1.5,
    allow_break_duration: float = 0.25,
    max_len: int = 50,
    target_spw: float = 0.3,
    extra_time: float = 0.5,
):
    """Split segments for display as subtitles.

    First, it plits the segments into chunks that were separated by large
    silences (`force_break_duration`). Then, it finds segments that are long
    (longer than `max_len`) and tries to break them up at the longest duration
    possible (see `split_segment_max_len`). Finally, it merges any resulting
    small segments that would be good to merge (see `merge_short_segments`).

    It also takes care of inter-segment display, adding some extra time for each
    subtitle to appear before/after the respective utterance.

    :param segments: a sequence of segments
    :param force_break_duration: any silence above this threshold will cause the
        subtitle to be split into two segments.
    :param allow_break_duration: segments will never be split at silence
        durations below this threshold.
    :param max_len: recommended length for subtitle text. Some subtitle text
        might end up being longer than this if there is no other way to split
        the line that respects the `allow_break_duration` threshold.
    :param target_spw: target seconds-per-word reading rate. Currently just
        displays a warning for violating segments.
    :param extra_time: extra time, in seconds, to add to the subtitle display
        before and after the utterance.
    :return: segments, (hopefully) ready to be shown as subtitles
    """
    # first, hard-split where silences >= allow_break_duration
    segments = resegment(segments, partial(split_segment_greedy, threshold=allow_break_duration))
    # then, merge everything we can as long we don't merge silences >= force_break_duration
    segments = merge_short_segments(segments, force_break_duration, max_len)

    add_extra_time(segments, extra_time)
    warn_spw(segments, target_spw)

    return segments


def resegment(
    segments: Segments,
    segment_to_segments_func: typing.Callable[[Words], Segments],
    longer_than: typing.Optional[int] = None,
):
    """Apply a function the splits each segments into possible sub-segments,
    and return the new list of segments resulting from this process.

    :param segments: a list of segments
    :param segment_to_segments_func: a function that receives a list of words
        and returns a list of one or more segments.
    :param longer_than: only process segments with total text length larger than
        this.
    :return: a list of segments after (possible) splitting.
    """
    new_segments = []
    for segment in segments:
        words = segment["words"]
        if longer_than is not None and get_words_total_len(words) < longer_than:
            new_segments.append(segment)
            continue
        new_segments.extend(segment_to_segments_func(words))
    return new_segments


def get_words_total_len(words: Words):
    """Total length of the string represented by the list of word dicts

    :param words: sequnece of word dicts
    :return: total length of all text, including delimiters between
        words.
    """
    return sum(len(word["word"]) for word in words)


def split_segment_greedy(words: Words, threshold: float):
    """Split a single segment word sequence into multiple segments, wherever
    a silence duration of at least `threshold` is encountered.

    :param words: sequence of words
    :param threshold: silence duration threshold to split at
    :return: a list of segments
    """
    segments = []
    segment_words = []

    def push():
        nonlocal segment_words
        segment = Segment(
            words=segment_words,
            start=segment_words[0]["start"],
            end=segment_words[-1]["end"],
        )
        segments.append(segment)
        segment_words = []

    prev_word: typing.Optional[Word] = None
    for word in words:
        silence = 0
        if prev_word is not None:
            silence = word["start"] - prev_word["end"]
        if silence >= threshold and segment_words:
            push()
        segment_words.append(word)
        prev_word = word

    if segment_words:
        push()

    return segments


def merge_short_segments(segments: Segments, max_silence: float, max_len: int):
    """Merge adjacent segments separated by silences shorter than `max_silence`
    as long as the combined segment text does not exceed `max_len`

    :param segments: a sequence of segments
    :param max_silence: maximum silence between segments to allow merge
    :param max_len: maximum length of merged segment length to allow merge
    :return: a sequence of (possibly merged) segments
    """
    # start from shortest silence
    # idx==0 is for the silence between word at index 0 and word at index 1
    sorted_silences = sorted(
        ((s2["start"] - s1["end"], idx) for idx, (s1, s2) in enumerate(zip(segments[:-1], segments[1:]))),
    )
    segment_lens = [get_words_total_len(segment["words"]) for segment in segments]
    merge_idxs = set()
    last_idx = len(segments) - 1
    for silence, idx in sorted_silences:
        if silence > max_silence:
            break
        if idx == last_idx:
            continue
        s = segment_lens[idx] + segment_lens[idx + 1]
        if s < max_len:
            # short silence, we can merge without violating max_len
            # mark segments as merged by setting both lengths to merged length
            segment_lens[idx] = segment_lens[idx + 1] = s
            # we merge backwards; so this merges idx with idx + 1
            merge_idxs.add(idx + 1)
    # do actual merge
    new_segments = []
    for idx, segment in enumerate(segments):
        if idx in merge_idxs:
            new_segments[-1] = Segment(
                start=new_segments[-1]["start"],
                end=segment["end"],
                words=new_segments[-1]["words"] + segment["words"],
            )
        else:
            new_segments.append(segment)
    return new_segments


def add_extra_time(segments: Segments, extra_time: float):
    """Pads segments with some extra time so that they are displayed
    a bit before/after actual speech.

    :param segments: _description_
    :param extra_time: _description_
    """
    segments[0]["start"] -= extra_time
    for s1, s2 in zip(segments[:-1], segments[1:]):
        if s2["start"] - extra_time > s1["end"] + extra_time:
            s1["end"] += extra_time
            s2["start"] -= extra_time
        else:
            s1["end"] = s2["start"] = (s1["end"] + s2["start"]) / 2
    segments[-1]["end"] += extra_time


def warn_spw(segments: Segments, min_spw: float):
    """Display warnings for segments that don't allow enough seconds per word
    according to the recommended reading rate.

    TODO : try to automatically fix SPW by combining adjacent segments, where
        possible.

    :param segments: sequence of segments
    :param min_spw: minimum recommended seconds-per-word reading rate
    """
    problems = []
    for segment in segments:
        num_words = len(segment["words"])
        duration = segment["end"] - segment["start"]
        spw = duration / num_words
        if spw < min_spw:
            problems.append(f"at time {format_srt_time(segment['start'])} spw={spw:.2f}")
    if problems:
        print(f"Warning: {len(problems)} segments have high word rate (seconds per word < {min_spw})", file=sys.stderr)
        for problem in problems:
            print("  - " + problem, file=sys.stderr)


def format_srt_time(seconds: float):
    """Formats the timestamp (in seconds) in an SRT timestamp format.

    :param seconds: timestamp in seconds
    :return: timestamp formatted in SRT format
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"


def fix_start_end(segments: Segments):
    """Ensures that all words in all segments have "start" and "end" times,
    guessing to the best of our ability.

    :param segments: a sequence of segment dicts
    :return: the same segments, modified so that all words have both
        "start" and "end" times.
    """
    known_durations = defaultdict(list)
    for segment in segments:
        words = segment["words"]
        for word in words:
            if not ("end" in word and "start" in word):
                continue
            s = word["word"]
            duration = word["end"] - word["start"]
            known_durations[s] = duration
            known_durations["by_class", get_char_classes(s)] = duration
            known_durations["by_length", len(s)] = duration

    def best_guess_duration(s):
        for k in (s, get_char_classes(s), len(s)):
            if k not in known_durations:
                continue
            ds = known_durations[k]
            return sum(ds) / len(ds)
        return mean_duration

    last_end = 0.0
    for segment in segments:
        words = segment["words"]
        silences = [w2["start"] - w1["end"] for w1, w2 in zip(words[:-1], words[1:]) if "start" in w2 and "end" in w1]
        mean_silence = sum(silences) / len(silences)
        durs = [w["end"] - w["start"] for w in words if "start" in w and "end" in w]
        mean_duration = sum(durs) / len(durs)
        segment_start = segment["start"] if "start" in segment else None

        changes = True
        while changes:
            changes = False
            if "start" not in words[0]:
                words[0]["start"] = last_end + mean_silence
                changes = True
                dur = best_guess_duration(words[0]["word"])

            for w1, w2 in zip(words[:-1], words[1:]):
                if "end" not in w1 and "start" in w2:
                    w1["end"] = w2["start"]
                if "start" in w2:
                    if w2["start"] < w1["end"]:
                        w2["start"] = w1["end"]
                        changes = True
                    if w2["end"] < w2["start"]:
                        w2["end"] = w2["start"]
                        changes = True
                    if w1["end"] < w1["start"]:
                        w1["end"] = w1["start"]
                        changes = True
                    continue
                w2["start"] = w1["end"] + mean_silence
                dur = best_guess_duration(w2["word"])
                w2["end"] = w2["start"] + dur
                changes = True

            if "end" not in words[-1]:
                dur = best_guess_duration(words[-1]["word"])
                words[-1]["end"] = words[-1]["start"] + dur
                changes = True

        last_end = words[-1]["end"]
        if "end" not in segment or segment["end"] < last_end:
            segment["end"] = last_end

        if segment_start is None:
            segment["start"] = segment["words"][0]["start"]

    for segment in segments:
        prev_end = None
        for word in segment["words"]:
            assert "start" in word, (segment, word)
            assert "end" in word, (segment, word)
            assert word["end"] >= word["start"], (segment, word)
            if prev_end is not None:
                assert word["start"] >= prev_end, (segment, word, prev_end)
            prev_end = word["end"]
        assert "start" in segment, segment
        assert "end" in segment, segment
    return segments


def get_char_classes(
    s: str,
    rs: tuple[str] = ("א-ת", "a-zA-Z", "0-9"),
    no_class: str = "\0",
):
    """Returns a string that represents the character classes for each character
    in the string `s`. Each character class range in `rs` is represented by the
    first letter in that range. A character matching no class is represented
    by `no_class`.

    >>> get_char_classes("שלום world")
    "אאאא\x00aaaaa"

    :param s: a string
    :param rs: a tuple of character class ranges (regular expression style)
    :param no_class: the character to use for characters not in any range in `rs`
    :return: the character classes appearing in that string, represented by one
        letter each, in order of appearence.
    """

    for r in rs:
        s = re.sub(f"[{r}]", r[0], s)
    s = re.sub(f'[^{"".join(rs)}]', no_class, s)
    return s


def dump_srt(segments: Segments, fp: typing.TextIO):
    """Dump the given transcription segments as a subtitle file in SRT format.

    :param segments: transcription segments
    :param fp: output file-like object
    """
    segment_id = 0
    for segment in segments:
        # print(segment)
        if not segment["words"]:
            continue
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = "".join(word["word"] for word in segment["words"])
        segment_id += 1
        line = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
        fp.write(line)
