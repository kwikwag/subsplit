"""Main CLI for subsplit"""
import argparse
import contextlib
import inspect
import json
import sys
import typing
from functools import partial

from .subsplit import (
    dump_srt,
    fix_start_end,
    split_segments_for_subtitles,
)


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
    add_defaults_as_args(split_segments_for_subtitles, parser)

    args = parser.parse_args()

    if args.format is None:
        if args.out == "-":
            args.format = "srt"
        else:
            args.format = args.out.rsplit(".", 1)[-1]
    if args.format not in {"srt", "json"}:
        parser.error(f"--format should be 'srt' or 'json', not {args.format!r}")

    # bind_default_args(calculate_penalty_partial, args)
    bind_default_args(split_segments_for_subtitles, args)

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

    # segments = split_segments_for_subtitles2(segments, args.gap, args.max_len, calculate_penalty)
    segments = split_segments_for_subtitles(segments)

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
