"""Split transcription segments in a wise way for display as subtitles."""
import argparse
import contextlib
import inspect
import json
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
    preceding_silence: typing.Optional[float] = None


Words = typing.Iterable[Word]


class Segment(typing.TypedDict):
    """Represents a segment dict"""
    words: typing.Iterable[Word]
    start: typing.Optional[int]
    end: typing.Optional[int]


Segments = typing.Iterable[Segment]


Number = typing.Union[int, float]


def cum_sum(a: typing.Sequence[Number]) -> typing.Iterable[Number]:
    """Generate cumulative sums of the numeric sequence `a`

    >>> tuple(cum_sum(range(4))
    (1, 3, 6, 10)

    :param a: a sequence of numbers
    :yield: cumulative sums
    """
    if not a:
        return
    s = 0
    for el in a:
        s += el
        yield s


def max_threshold_min_split(split_vals: typing.Sequence[float], sum_vals: typing.Sequence[float], s: float) -> float:
    """Finds the maximum threshold value `t` such that, when the corresponding
    sequences `split_vals` and `sum_vals` are split into segments at all
    positions where the `split_vals` value is greater than `t`, the sum of the
    largest segment values from `sum_vals` does not exceed the limit `s`.

    :param split_vals: a list of numbers to compare to the split thresholds
    :param sum_vals: a list of numbers to compare to the sum threshold `s`
    :param s: maximum allowed segment sum
    :return: thresold that would split the given sequences into segments,
        each of which don't exceed `s` in sum of values from `sum_vals`.
        If no such splitting exists, returns None.
    """
    unique_values = sorted(set(split_vals))
    a_sums = list(cum_sum(sum_vals))
    n = len(sum_vals)
    if n != len(split_vals):
        raise ValueError("`split_vals` should match `sum_vals`")

    low, high = 0, len(unique_values) - 1
    result = None

    while low <= high:
        mid = (low + high) // 2
        t = unique_values[mid]
        split_idxs = [None] + [i for i in range(n) if split_vals[i] >= t] + [None]
        max_s = 0
        for i1, i2 in zip(split_idxs[:-1], split_idxs[1:]):
            s1 = 0 if i1 is None else a_sums[i1]
            s2 = a_sums[-1] if i2 is None else a_sums[i2]
            max_s = max(max_s, s2 - s1)

        if max_s <= s:
            result = t
            low = mid + 1
        else:
            high = mid - 1

    return result


def diff(a: typing.Iterable[Number]) -> typing.Iterable[Number]:
    """Calculate sequential differences within a numeric sequence.

    The length of the resulting iterable is one less than the input iterable.

    >>> tuple(diff((4, 2, 5)))
    (-2, 3)

    :param a: a sequence of numbers
    :yield: differences between adjacent elements
    """
    prev = None
    for el in a:
        if prev is not None:
            yield el - prev
        prev = el


def find_sufficient_silence(
    words: Words,
    min_silence: float,
    required_text_len: int,
    allow_longer_text: bool = True,
):
    """Find the maximal silence duration that would ensure that the list of
    words could be split to segments, such that the text length of each segment
    does not exceed `max_text_length`.

    If `allow_longer_text` is True (default), then if there is no threshold
    that respects `min_silence` that allows splitting such that no segment
    exceeds `required_text_len` in length, the next best length that does allow
    such a split is chosen. This is done by means of a binary search.

    :param words: a list of word dicts
    :param min_silence: minimum silence to allows splits
    :param required_text_len: the maximal text length allows for a segment.
    :param allow_longer_text: allow stretching the limit of `required_text_len`
        to always reach some kind of split.
    """
    split_idxs, candidate_silences = zip(
        *(
            (idx, word["preceding_silence"])
            for idx, word in enumerate(words)
            if word["preceding_silence"] >= min_silence
        )
    )

    # splitting at split_idxs results in between len(split_idxs) - 1 and len(split_idxs) + 1
    # segments, depending on whether the starrt and end indices are included.
    # here we force-include them to calculate corresponding segment lengths
    split_idxs = (0,) + split_idxs + (len(words),)

    # prepend zero so that the length indicate the sum up to the word index,
    # exclusive
    cum_lens = [0] + list(get_cum_words_lens(words))
    split_cum_lens = tuple(cum_lens[idx] for idx in split_idxs)
    seg_lens = list(diff(split_cum_lens))

    # each candidate silence corresponds to the corresponding seg_lens index,
    # except for the last, which represent the segment length after the final
    # silence and the end.
    # we add an artificial silence candiatate at the end so that each segment
    # has the corresponding silence attached to it (and we can't split at the end)
    candidate_silences = candidate_silences + (min_silence,)

    i = 0
    threshold = None
    while threshold is None:
        threshold = max_threshold_min_split(candidate_silences, seg_lens, required_text_len * (2**i))
        i += 1
        if not allow_longer_text:
            break

    if i > 1:
        low = required_text_len * (2 ** (i - 2))
        high = required_text_len * (2 ** (i - 1))
        best = None
        while low <= high:
            mod_required_text_len = (low + high) // 2
            threshold = max_threshold_min_split(candidate_silences, seg_lens, mod_required_text_len)
            if threshold is None:
                # too short
                low = mod_required_text_len + 1
            else:
                # long enough, try shorter
                best = threshold
                high = mod_required_text_len - 1
        threshold = best
        assert threshold is not None

    return threshold


def add_silences(segments: Segments):
    """Add the 'preceding_silence' key to each word dict

    :param segments: list of segment dicts
    :return: segments, where each word dict also contains the calculated
        'preceding_silence' key
    """
    last_end = 0.0
    for segment in segments:
        for word in segment["words"]:
            try:
                silence = word["start"] - last_end
                last_end = word["end"]
            except KeyError:
                continue
            word["preceding_silence"] = silence
    return segments


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


def optimal_k_partition(seq: typing.Sequence[int], k) -> list[int]:
    """Partitions a sequence `seq` of ints to `k` segments, minimizing the
    differences of sums between segments (i.e. the sum of difference between
    each segment and the mean segment sum).

    Implemented using dynamic programming.

    :param seq: a list of ints
    :param k: number of required partitions
    :return: list of indices where `seq` should be split to obtain the partition
        that minimizes the difference between sums of segments.
    """
    # thanks to ChatGPT
    n = len(seq)
    if k == 1:
        return []  # No partitions needed if only one segment

    # Precompute cumulative sums for faster range sum queries
    cum_sums = [0] * (n + 1)
    for i in range(n):
        cum_sums[i + 1] = cum_sums[i] + seq[i]

    def range_sum(i, j):
        """Return sum(seq[i:j]), using cumulative sum."""
        return cum_sums[j] - cum_sums[i]

    # Initialize DP table and partition points table
    dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    partition_points = [[-1] * (k + 1) for _ in range(n + 1)]

    # Base case: for 1 partition, the sum is just the sum of the entire subarray
    for i in range(n + 1):
        dp[i][1] = abs(range_sum(0, i) - cum_sums[n] / k)

    # Fill the DP table
    for j in range(2, k + 1):  # Number of partitions
        for i in range(n + 1):  # End index for the current partition
            for m in range(j - 1, i):  # Partition point (m..i)
                current_sum = range_sum(m, i)
                imbalance = dp[m][j - 1] + abs(current_sum - cum_sums[n] / k)
                if imbalance < dp[i][j]:
                    dp[i][j] = imbalance
                    partition_points[i][j] = m

    # Backtrack to find the partition points
    result = []
    idx = n
    for j in range(k, 1, -1):
        idx = partition_points[idx][j]
        result.append(idx)

    return tuple(sorted(result))


def get_words_total_len(words: Words):
    """Total length of the string represented by the list of word dicts

    :param words: sequnece of word dicts
    :return: total length of all text, including delimiters between
        words.
    """
    return sum(len(word["word"]) for word in words)


def get_cum_words_lens(words: Words):
    """Generates a list of ints, counting the total length of text up to the
    respective word index, inclusive.

    :param words: a list of word dicts
    :yield: a generator producing cumulative lengths respective of each word
        position.
    """
    return cum_sum(len(word["word"]) for word in words)


# see: https://www.clevercast.com/bbc-subtitling-guidelines
#   - at least 1.5sec between subtitles -> anything longer we split
#   - no longer than 2 lines -> use estimate for number of chars per line,
#     and split according to that
#   - min_duration_per_word
def calculate_penalty(
    text_len: int,
    num_words: int,
    start_time,
    end_time,
    prev_end_time,
    max_len: int = 50,
    spw: float = 0.3,
    gap: float = 1.5,
    extra_time: float = 0.5,
    len_penalty: float = 1.0,
    spw_penalty: float = 1.0,
    gap_penalty: float = 1.0,
    eps: float = 1e-7,
):
    """Calculate penalty for a given segment.
    :param text_len: text length of the segment
    :param num_words: number of words in segment
    :param start_time: voice start time
    :param end_time: voice end time
    :param prev_end_time: previous segment voice end time
    :param max_len: max allowed text length
    :param spw: target seconds per word reading rate
    :param gap: required gap in seconds
    :param extra_time: extra time each segment is expected to be displayed on screen, before and after voice start/end
    :param len_penalty: penalty weight for going over max allowed text length.
        When the length is twice as long as the max allowed length, the incurred
        penalty is 2 * len_penalty (it is linear with the violation of the max text length)
    :param spw_penalty: penalty weight for going over max allowed seconds-per-word reading rate.
        When duration is 1/2 * spw * num_words, the incurred penalty is 2 * spw_penalty
        (it is in inverse ratio with the violation of the seconds-per-word reading rate limit)
    :param gap_penalty: penalty weight for having too little gap between segments.
        When the gap is 1/2 * gap, the incurred penalty is 2 * gap_penalty (it is
        in inverse ratio with the violation of the gap limit)
    :param eps: a small value, provided to avoid zero-division errors
    :return: the calculated penalty
    """

    duration = (extra_time * 2) + (end_time - start_time)
    penalty = 0

    # Penalty for exceeding character limit
    if text_len > max_len:
        # when text_len = 2 * max_len, we get 2 * len_penalty
        penalty += (text_len / max_len) * len_penalty

    # Penalty for not meeting the 0.3s per word requirement
    if duration < spw * num_words:
        # when duration is 0.5 * spw * num_words, we get 2 * spw_penalty
        penalty += ((spw * num_words) / (duration + eps)) * spw_penalty

    # Penalty for insufficient gap between segments
    if start_time - prev_end_time < gap:
        # when existing gap is 0.5 * gap, we get 2 * gap_penalty
        penalty += gap / (start_time - prev_end_time + eps) * gap_penalty

    return penalty


# pylint: disable=redefined-outer-name
def split_segment_dp_penalty(words: Words, calculate_penalty=calculate_penalty):
    """Calculates the best way to split `words` into segments of words
    assuming a penalty calculated by the function `calculate_penalty`.
    By default, this balances three factors: total length, silence
    between segments and word reading rate.

    :param words: a sequence of word dicts.
    :param calculate_penalty: a loss function; should accept the arguments
        (text_len, num_words, start_time, end_time, prev_end_time) representing
        the suggested segment total text length, number of words, start time,
        end time, and the previous segment end time, and should return a penalty
        value.
    :return: The splitting of words into segment that minimizes the penalty.
    """

    n = len(words)

    # dp[i] will store the minimum penalty for splitting words[0...i-1]
    dp = [float("inf")] * (n + 1)
    dp[0] = 0  # No words, no penalty.

    # To store the split points to reconstruct the solution
    split_points = [-1] * (n + 1)

    # add 0 at the start so that cum_sums[i] - cum_sums[j] is the length of words
    # between words[j] and words[i], including j but excluding i
    cum_sums = [0] + list(get_cum_words_lens(words))

    # Iterate through each word position
    for i in range(1, n + 1):
        for j in range(i):
            # Calculate the penalty for the segment words[j:i]
            segment_start = words[j]["start"]
            segment_end = words[i - 1]["end"]
            segment_text_len = cum_sums[i] - cum_sums[j]
            segment_num_words = i - j  # Number of words in the segment

            # Handle the previous end time (if j > 0)
            prev_end_time = words[j - 1]["end"] if j > 0 else 0

            # Calculate the penalty for this segment
            penalty = calculate_penalty(segment_text_len, segment_num_words, segment_start, segment_end, prev_end_time)

            # Update dp[i] if a better (lower penalty) split is found
            if dp[j] + penalty < dp[i]:
                dp[i] = dp[j] + penalty
                split_points[i] = j

    # Reconstruct the solution (the actual splits)
    segments = []
    i = n
    while i > 0:
        j = split_points[i]
        # Add the segment to the list with the correct start and end times
        segments.append(Segment(words=words[j:i], start=words[j]["start"], end=words[i - 1]["end"]))
        i = j

    segments.reverse()
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


def format_srt_time(seconds: float):
    """Formats the timestamp (in seconds) in an SRT timestamp format.

    :param seconds: timestamp in seconds
    :return: timestamp formatted in SRT format
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"


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


def split_segment_max_len(words: Words, min_silence: float, max_len: int):
    """
    Given a sequence `seq` of positive integers, find the optimal partition,
    defined as the list of indices that divide `seq` into sequential segments,
    such that the number of segments is kept minimal while each segment does not exceed
    `max_sum`.

    :param seq: List of positive integers
    :param max_sum: Maximum sum allowed for any partition
    :return: List of indices indicating where partitions occur
    """
    threshold = find_sufficient_silence(words, min_silence=min_silence, required_text_len=max_len)
    if threshold is None:
        segment = Segment(words=words, start=words[0]["start"], end=words[-1]["end"])
        return [segment]

    # split at this threshold
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

    for word in words:
        if word["preceding_silence"] >= threshold and segment_words:
            push()
        segment_words.append(word)
    if segment_words:
        push()

    # merge together sections greedily as long as we don't exceed max_len
    return segments


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


def merge_short_segments(segments: Segments, max_silence: float, max_len: int):
    """Merge adjacent segments separated by silences shorter than `max_silence`
    as long as the combined segment text does not exceed `max_len`

    :param segments: a sequence of segments
    :param max_silence: maximum silence between segments to allow merge
    :param max_len: maximum length of merged segment length to allow merge
    :return: a sequence of (possibly merged) segments
    """
    # start from shortest silence
    sorted_silences = sorted(
        ((s2["start"] - s1["end"], idx) for idx, (s1, s2) in enumerate(zip(segments[:-1], segments[1:]))),
    )
    segment_lens = [get_words_total_len(segment["words"]) for segment in segments]
    merge_idxs = set()
    for silence, idx in sorted_silences:
        if silence > max_silence:
            break
        if idx == 0:
            continue
        s = segment_lens[idx - 1] + segment_lens[idx]
        if s < max_len:
            # short silence, we can merge without violating max_len
            # mark segments as merged by setting both lengths to merged length
            segment_lens[idx - 1] = segment_lens[idx] = s
            merge_idxs.add(idx)
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


# see: https://www.clevercast.com/bbc-subtitling-guidelines
# at least 1.5sec between subtitles -> anything longer we split
# no longer than 2 lines -> use estimate for number of chars per line, and split according to that
# min_duration_per_word
def split_segments_for_subtitles1(
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
    # first, hard-split where silences >= force_break_duration
    segments = resegment(segments, partial(split_segment_greedy, threshold=force_break_duration))

    segments = resegment(
        segments,
        partial(split_segment_max_len, min_silence=allow_break_duration, max_len=max_len),
        longer_than=max_len,
    )

    segments = merge_short_segments(segments, force_break_duration, max_len)

    add_extra_time(segments, extra_time)
    warn_spw(segments, target_spw)

    return segments


def split_segments_for_subtitles2(segments: Segments, gap: float, max_len: float, calculate_penalty=calculate_penalty):
    """Split segments for display as subtitles.

    This approach uses dynamic programming with a loss function for splits, that
    is supposed to balance between the reading rate, silences between subtitles
    and subtitle lengths. However, this isn't quite baked yet, and perhaps will
    never be.

    TODO : consider improving this.

    :param segments: a sequence of segments
    :param gap: ideal silence between utterances for splitting subtitle segments
    :param max_len: recommended length for subtitle text
    :param calculate_penalty: a loss function (see `calculate_penalty`)
    :return: segments, (hopefully) ready to be shown as subtitles
    """
    segments = split_segment_greedy([word for segment in segments for word in segment["words"]], threshold=gap)
    new_segments = []
    for segment in segments:
        words = segment["words"]
        if sum(len(word["word"]) for word in words) <= max_len:
            new_segments.append(segment)
            continue
        new_segments.extend(split_segment_dp_penalty(words, calculate_penalty=calculate_penalty))
    segments = new_segments
    return segments


def add_defaults_as_args(f: typing.Callable, parser: argparse.ArgumentParser):
    """Add the default keyword arguments of the callable `f` as arguments to
     `parser`, to allow specifying them in a CLI.

    :param f: a callable with some default named arguments
    :param parser: CLI argparse parser to add arguments to
    """
    sig = inspect.signature(f)
    defaults = {name: param.default for name, param in sig.parameters.items() if param.default is not param.empty}
    for name, default in defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)


def bind_default_args(f: typing.Callable, args: argparse.Namespace):
    """Take arguments specified as CLI arguments in `args` and return a partial
    function `f` with any matching default arguments set to those specified in
    the CLI.

    :param f: a callable with some default named arguments
    :param args: CLI arguments
    :return: a partial function `f` with the CLI arguments bound to named
        arguments
    """
    signature = inspect.signature(f)
    kwargs = {}
    for name, param in signature.parameters.items():
        if param.default is param.empty:
            continue
        kwargs[name] = getattr(args, name)
    return partial(f, **kwargs)


def main():
    """Subsplit CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", default="-")
    parser.add_argument("--out", default="-")
    parser.add_argument("--format", default=None, choices=["json", "srt"])
    parser.add_argument("--add-spaces", action="store_true")

    # add_defaults_as_args(calculate_penalty, parser)
    add_defaults_as_args(split_segments_for_subtitles1, parser)

    args = parser.parse_args()

    if args.format is None:
        if args.out == "-":
            args.format = "srt"
        else:
            args.format = args.out.rsplit(".", 1)[-1]
    if args.format not in {"srt", "json"}:
        parser.error(f"--format should be 'srt' or 'json', not {args.format!r}")

    # bind_default_args(calculate_penalty_partial, args)
    bind_default_args(split_segments_for_subtitles1, args)

    with contextlib.ExitStack() as stack:
        if args.file == "-":
            fp = sys.stdin
        else:
            fp = stack.push(open(args.file, "r", encoding="utf-8"))
        segments = json.load(fp)["segments"]

    if args.add_spaces:
        for segment in segments:
            for word in segment["words"][:-1]:
                if word["word"].endswith(" "):
                    continue
                word["word"] += " "

    fix_start_end(segments)
    add_silences(segments)

    # segments = split_segments_for_subtitles2(segments, args.gap, args.max_len, calculate_penalty)
    segments = split_segments_for_subtitles1(segments)

    with contextlib.ExitStack() as stack:
        if args.out == "-":
            fp = sys.stdout
        else:
            fp = stack.push(open(args.out, "w", encoding="utf-8"))
        if args.format == "json":
            json.dump(segments, fp, indent=1, ensure_ascii=False)
        elif args.format == "srt":
            dump_srt(segments, fp)
        else:
            raise ValueError(args.format)


if __name__ == "__main__":
    main()
