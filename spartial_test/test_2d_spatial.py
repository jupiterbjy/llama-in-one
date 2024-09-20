"""
Tests 2D spatial capability of give models.

Script is designed to run on CPU because I'm VRAM poor :(
```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

:Author: jupiterbjy@gmail.com
"""

import re
import time
import pathlib
from ast import literal_eval
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Sequence, ContextManager, Tuple, List

import psutil
import tqdm
from llama_cpp import Llama, LlamaCache


ANSWER_PATTERN = re.compile(r"```\s*([^`]*)\s*```")

REPORT_FILE_PATH = (
    pathlib.Path(__file__).parent / f"test_2d_spatial_{int(time.time())}.txt"
)

PROMPT = """
Imagine 10x10 grid, where top-left corner is (1, 1) and bottom-right corner is (10, 10).

For input with format of `(start_x, start_y) -> (end_x, end_y)`:
output list of movements required from start to end separated by newline & wrapped inside markdown code block,
in format of `direction -> (x_after_move, y_after_move)` where `direction` is one of `(up, down, left, right)`

"""

TEST_INPUTS: Sequence[
    str
] = """
(3, 7) -> (9, 2)
(8, 1) -> (5, 6)
""".strip().split(
    "\n"
)

SEED = 0

CONTEXT_LEN = 1024

THREADS = psutil.cpu_count(logical=False)

LLAMA_VERBOSE = False

FLASH_ATTENTION = True

# --- Utilities ---


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

    for action in output.strip().split("\n"):
        split = action.split("->")

        movement = DIRECTION_MAP[split[0].strip(" ").lower()]
        post_movement: Tuple[int, int] = literal_eval(split[-1])

        current = current[0] + movement[0], current[1] + movement[1]

        if current != post_movement:
            return False

    return current == destination


def test_single_model(gguf_path: pathlib.Path) -> Sequence[str]:
    """Load single model and run test"""

    messages = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": ""},
    ]

    # noinspection PyTypeChecker
    outputs: List[str] = []

    with load_gguf(gguf_path) as llm:
        for test in tqdm.tqdm(TEST_INPUTS):

            # noinspection PyTypeChecker
            generated = llm.create_chat_completion(
                messages + [{"role": "user", "content": test}], seed=0
            )["choices"][0]["message"]["content"]

            outputs.append(ANSWER_PATTERN.search(generated).group(1))

    # extract answer
    return outputs


def main(gguf_paths: Sequence[pathlib.Path]):
    """Loads each model and runs test"""

    write_pending: List[str] = []

    for gguf_path in reversed(gguf_paths):
        corrects = 0

        write_pending.append(f"\n\nMODEL: {gguf_path.name}")

        outputs = test_single_model(gguf_path)

        for input_, output in zip(TEST_INPUTS, outputs):
            corrects += validate_answer(input_, output)

        write_pending.append(f"SCORE: {corrects} / {len(TEST_INPUTS)}")

        for input_, output in zip(TEST_INPUTS, outputs):
            write_pending.append(f"INPUT: {input_}\nOUTPUT: {output}\n")

        print(f"[{gguf_path.name}] Scored {corrects} / {len(TEST_INPUTS)}")

    REPORT_FILE_PATH.write_text("\n".join(write_pending), encoding="utf8")


if __name__ == "__main__":
    _parser = ArgumentParser()
    _parser.add_argument(
        "gguf_paths",
        type=pathlib.Path,
        nargs="+",
        help="GGUF files or directories containing it.",
    )

    _args, _llama_args = _parser.parse_known_args()
    # TODO: support some llama args in future

    _paths: List[pathlib.Path] = []

    # fetch gguf only, but don't go deep
    for _path in _args.gguf_paths:
        _path: pathlib.Path

        if _path.is_dir():
            _paths.extend(_p for _p in _path.iterdir() if _p.suffix.lower() == ".gguf")

        elif _path.suffix.lower() == ".gguf":
            _paths.append(_path)

    main(_paths)
