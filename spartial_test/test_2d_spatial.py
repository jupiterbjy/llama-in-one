"""
Tests 2D spatial capability of give models.

Script is designed to run on CPU because I'm VRAM poor :(
```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

:Author: jupiterbjy@gmail.com
"""

import itertools
import re
import time
import pathlib
from ast import literal_eval
from argparse import ArgumentParser
from collections.abc import Generator
from contextlib import contextmanager
from typing import Sequence, ContextManager, Tuple, List, Iterable

import psutil
import tqdm
from llama_cpp import Llama, LlamaCache


# --- Config ---

ANSWER_PATTERN = re.compile(r"```\s*([^`]*)\s*```")

REPORT_PATH = pathlib.Path(__file__).parent / "results"

PROMPT = """
Imagine 10x10 grid, where top-left corner is (1,1) and bottom-right corner is (10,10).

Thus, moving upward decreases y coordinate and downward increases it. 

For input with format of `(start_x,start_y)->(end_x,end_y)` - output only the list of movements
required to move from start to end, each separated by newline and wrapped in markdown code block.

Each line should follow format `<direction>->(x_after_move,y_after_move)`,
where `<direction>` is either one of `up, down, left, right`.

Starting from below:
---

"""

TEST_INPUTS: Sequence[
    str
] = """
(3, 7) -> (9, 2)
(8, 1) -> (5, 6)
(1, 5) -> (7, 9)
(6, 3) -> (2, 8)
(2, 2) -> (10, 6)
(9, 9) -> (4, 4)
(5, 10) -> (8, 3)
(4, 6) -> (1, 2)
(7, 8) -> (3, 5)
(10, 4) -> (6, 1)
(1, 1) -> (1, 1)
(8, 7) -> (5, 3)
(1, 1) -> (10, 10)
(6, 6) -> (3, 3)
(9, 2) -> (4, 9)
(3, 4) -> (8, 10)
(5, 7) -> (2, 3)
(10, 9) -> (7, 4)
(4, 2) -> (1, 8)
(7, 3) -> (10, 1)
(2, 6) -> (9, 8)
(8, 5) -> (3, 1)
(1, 3) -> (6, 7)
(6, 9) -> (2, 5)
(9, 6) -> (5, 2)
(3, 1) -> (8, 8)
(1, 9) -> (7, 5)
(5, 4) -> (2, 1)
(8, 3) -> (4, 10)
(2, 4) -> (9, 1)
(6, 2) -> (3, 9)
(7, 5) -> (4, 8)
""".strip().split(
    "\n"
)

SEED = 0

CONTEXT_LEN = 2048

THREADS = psutil.cpu_count(logical=False) - 1

LLAMA_VERBOSE = False

FLASH_ATTENTION = True


# --- Utilities ---


def fetch_gguf_gen(path: pathlib.Path) -> Generator[pathlib.Path]:
    """Generator that fetches GGUF from given path recursively."""

    # ignore underscore prefix
    if path.name.startswith("_"):
        return

    # if dir yield from it
    if path.is_dir():
        for sub_path in path.iterdir():
            yield from fetch_gguf_gen(sub_path)

    # if file check for suffix and yield it
    if path.suffix.lower() == ".gguf":
        print("Found", path.name)
        yield path


@contextmanager
def load_gguf(gguf_path: pathlib.Path, *args, **kwargs) -> ContextManager[Llama]:
    """Context Manager handling init and shutdown of llm model."""

    print(f"[{gguf_path.name}] Loading")

    cache = LlamaCache()
    llm = Llama(
        gguf_path.as_posix(),
        seed=SEED,
        n_ctx=CONTEXT_LEN,
        # wish there was library for accounting P core only
        n_threads=THREADS,
        verbose=LLAMA_VERBOSE,
        flash_attn=FLASH_ATTENTION,
        *args,
        **kwargs,
    )
    llm.set_cache(cache)

    start_time = time.time()
    try:
        print(f"[{gguf_path.name}] Started")
        yield llm

    finally:
        print(f"[{gguf_path.name}] Completed in {time.time() - start_time:.2f}")
        llm.close()


# --- Logic ---


# Screen coordination
DIRECTION_MAP = {
    "right": (1, 0),
    "left": (-1, 0),
    "up": (0, -1),
    "down": (0, 1),
}


def validate_answer(input_: str, output: str) -> bool:
    """Validates answers llm generated

    Args:
        input_: String with format of `(x_start, y_start) -> (x_end, y_end)`.
        output: Sequences of string with format of `direction -> (x_after_move, y_after_move)`, sep by newline.

    Returns:
        True if given destination is reached regardless of minimum or not
    """

    current: Tuple[int, int]
    destination: Tuple[int, int]
    current, destination = map(literal_eval, input_.split("->"))

    output = output.strip()

    if output:
        for action in output.split("\n"):
            split = action.split("->")

            movement = DIRECTION_MAP[split[0].strip(" ").lower()]
            post_movement: Tuple[int, int] = literal_eval(split[-1])

            current = current[0] + movement[0], current[1] + movement[1]

            if current != post_movement:
                return False

    return current == destination


def test_single_model(gguf_path: pathlib.Path) -> Sequence[str]:
    """Load single model and run test"""

    # noinspection PyTypeChecker
    outputs: List[str] = []

    with load_gguf(gguf_path) as llm:
        for test in tqdm.tqdm(TEST_INPUTS):

            # noinspection PyTypeChecker
            generated = llm.create_chat_completion(
                [{"role": "user", "content": PROMPT + test}],
                seed=0,
            )["choices"][0]["message"]["content"]

            outputs.append(ANSWER_PATTERN.search(generated).group(1))

    # extract answer
    return outputs


def main(gguf_paths: Iterable[pathlib.Path]):
    """Loads each model and runs test"""

    for gguf_path in gguf_paths:
        # prep output
        report_file_path = REPORT_PATH / (gguf_path.stem + ".txt")
        report_file_path.touch(exist_ok=True)

        write_pending: List[str] = [f"\n\nMODEL: {gguf_path.name}"]

        # generate output
        outputs = test_single_model(gguf_path)

        # validate score
        corrects = 0

        for input_, output in zip(TEST_INPUTS, outputs):
            corrects += validate_answer(input_, output)

        write_pending.append(f"SCORE: {corrects} / {len(TEST_INPUTS)}")

        # write output
        for input_, output in zip(TEST_INPUTS, outputs):
            write_pending.append(f"INPUT: {input_}\nOUTPUT: {output}\n")

        print(f"[{gguf_path.name}] Scored {corrects} / {len(TEST_INPUTS)}")

        # write to file
        report_file_path.write_text("\n".join(write_pending), encoding="utf8")


if __name__ == "__main__":
    _parser = ArgumentParser()
    _parser.add_argument(
        "gguf_paths",
        type=pathlib.Path,
        nargs="+",
        help="GGUF files or directories containing it.",
    )

    REPORT_PATH.mkdir(exist_ok=True)

    _args, _llama_args = _parser.parse_known_args()
    # TODO: support some llama args in future

    main(itertools.chain(*(fetch_gguf_gen(_p) for _p in _args.gguf_paths)))
