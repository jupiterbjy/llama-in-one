"""
Python Rich Live Console Demo for Dummy LLM stream output.

Since it's just whitespace & newline cut it's not accurate at all
but should be enough to test and fix long output Live data with various MD features.

:Author: jupiterbjy@gmail.com
"""
import itertools
import time
from sys import stdout
from typing import Iterable

from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console


NOTICE = """
Press ctrl+c to exit, enter to replay sample dummy text.

NOTE: FOLLOWING SAMPLE TEXT IS GENERATED FROM LLAMA3 8B.

---
"""

SAMPLE = """
To achieve this in Python, you'll need to use a library that provides low-level keyboard input handling. One such library is `pynput`. Here's an example code snippet that listens for keypresses and echoes them back, including the Shift modifier key:
```python
import pynput
from pynput.keyboard import Key, Listener

def on_key_press(key):
    try:
        print(f"Pressed: {key.char} ({'Shift' if key.shift else ''})")
    except AttributeError:
        print(f"Special key pressed: {key}")

with Listener(on_press=on_key_press) as listener:
    listener.join()
```
Here's how the code works:

1. We import `pynput` and `Key` from its `keyboard` module.
2. We define a callback function `on_key_press` that will be called whenever a key is pressed. This function takes a `key` object as an argument, which contains information about the pressed key (e.g., character, modifier keys).
3. In `on_key_press`, we check if the key has a `char` attribute (i.e., it's a printable character). If it does, we print the character and indicate whether the Shift key was pressed (`key.shift` is `True` if Shift was held down while pressing the key). If the key doesn't have a `char` attribute (e.g., it's a special key like Enter or Backspace), we print a message indicating that a special key was pressed.
4. We create an instance of `Listener`, passing our callback function `on_key_press`. The `with` statement ensures that the listener is properly cleaned up when we're done with it.
5. Finally, we call `listener.join()` to start listening for keyboard events.

To test Shift+Enter alongside normal inputs:

1. Run the code and observe the console output as you press keys on your keyboard.
2. Try pressing Shift+Enter (hold down Shift while pressing Enter). You should see the output indicate "Pressed: \n (Shift)" or similar, depending on your system's locale settings.

Note that `pynput` may not work correctly on all platforms or under certain circumstances (e.g., when running in a virtualized environment or with certain keyboard layouts). If you encounter issues, feel free to ask for help troubleshooting!
"""


def iter_with_separator(iterable, sep=" "):
    """Iterate given iterable with a separator between each item."""

    iterator = iterable.__iter__()

    try:
        yield next(iterator)
    except StopIteration:
        return

    for item in iterator:
        yield sep
        yield item


# would prefer for loop but not to dirty place, but for speed
SAMPLE_SPLIT = [
    *itertools.chain(
        *(
            iter_with_separator(line.split(" "))
            for line in [*iter_with_separator(SAMPLE.split("\n"), "\n")]
        )
    )
]


def delayed_iterator(iterable, delay=0.01):
    """Delays item yield"""

    for item in iterable:
        time.sleep(delay)
        yield item


def live_print_jolting(content_iterable: Iterable[str]):
    """First attempt of live print that jolts the console up & down when
    actual line count (counting text wrap) exceed console height."""

    accumulated_line = ""

    with Live(vertical_overflow="visible") as live:
        for chunk in content_iterable:
            accumulated_line += chunk
            live.update(Markdown(accumulated_line), refresh=True)


def live_print(content_iterable: Iterable[str]):
    """Print content in live."""

    # We need to cut at specific line to set new live instance.
    # otherwise we get screen jolting due to Console Rewriting from
    # non-visible line to current line, making screen go back and forth.
    # though regex should be avoided due to cost.

    iterator = iter(content_iterable)

    in_code_block = False
    code_block_lang = ""

    is_eof = False

    while not is_eof:
        accumulated_line = ""

        # set vertical_overflow just in case
        with Live(vertical_overflow="visible") as live:
            while True:
                try:
                    token = next(iterator)
                except StopIteration:
                    is_eof = True
                    break

                # if tail is newline break after print, we'd need new console
                if token == "\n":
                    stdout.write("\x1B[1A\n")
                    break

                # TODO: fix this crap with states

                # is this code block at start of line?
                # (since llm don't quote normally, won't care about quoted codeblocks)
                if token.startswith("```"):
                    if in_code_block:
                        in_code_block = False

                    else:
                        # this must be the new codeblock.
                        in_code_block = True

                        # not sure if token is ```language or just ```.
                        if token == "```":
                            try:
                                next_token = next(iterator)
                            except StopIteration:
                                is_eof = True
                                break

                            if not next_token == "\n":
                                # if next token isn't, or doesn't start with newline, must be language name.
                                code_block_lang = token
                            else:
                                # set empty type instead
                                code_block_lang = ""

                            # now change with token so it's added back later
                            accumulated_line += token
                            token = next_token

                        else:
                            # must have token with it.
                            code_block_lang = token[3:]

                # if in codeblock but accumulated_line is empty, should add new codeblock as
                # this Live console doesn't know about it
                elif in_code_block and not accumulated_line:
                    # print("\r")
                    accumulated_line = f"```{code_block_lang}\n"

                accumulated_line += token
                live.update(Markdown(accumulated_line), refresh=True)


class State:
    NORMAL = 0
    IN_CODE_BLOCK_OPENING = 1
    IN_CODE_BLOCK = 2
    IN_CODE_BLOCK_CLOSING = 3


def line_by_line_gen(content_iterable: Iterable[str]):
    line = ""

    for chunk in content_iterable:
        line += chunk
        if "\n" in line:
            line_front, line_back = line.split("\n")

            yield line_front
            line = line_back


def console_print(content_iterable: Iterable[str]):
    state = State.NORMAL
    code_block_prefix = "```\n"

    console = Console()

    for line in line_by_line_gen(content_iterable):
        match state:

            case State.NORMAL:
                if "```" in line:
                    state = State.IN_CODE_BLOCK
                    code_block_prefix = f"```{line.split('```')[-1]}\n"
                else:
                    console.print(Markdown(line))
                    stdout.write("\33[A")

            case State.IN_CODE_BLOCK:
                if "```" in line:
                    # console.print(Markdown(code_block_prefix))
                    # stdout.write("\x1B[2A\n")
                    stdout.write("\33[2B\33[1000C\n")
                    state = State.NORMAL
                    continue

                console.print(Markdown(code_block_prefix + line))
                stdout.write("\33[2A\33[1000C")

        # stdout.write(f"\x1B")


def dummy_exchange_turn():
    """Exchange turn with dummy reply text."""

    while True:
        # live_print_jolting(delayed_iterator(SAMPLE_SPLIT))
        # live_print(delayed_iterator(SAMPLE_SPLIT))
        console_print(delayed_iterator(SAMPLE_SPLIT))

        input("Press enter >> ")


if __name__ == "__main__":
    print(NOTICE)
    dummy_exchange_turn()
