"""
Python Rich Live Console Demo for Dummy LLM stream output

:Author: jupiterbjy@gmail.com
"""

import time

from rich.live import Live
from rich.markdown import Markdown


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


SAMPLE_SPLIT = [
    [*iter_with_separator(line.split(" "))]
    for line in [*iter_with_separator(SAMPLE.split("\n"), "\n")]
]


def dummy_exchange_turn():
    """Exchange turn with dummy reply text."""

    # TODO: find way to prevent jolting when vertical space is exceeded
    # Maybe split in every empty newline?
    # but then how to determine if it's middle of a code block or not?
    # might need to keep state if it's in code block or not

    in_code_block = False
    iterator = (chunk for chunk in (chunks for chunks in SAMPLE_SPLIT))

    while True:
        input("Press enter >> ")

        accumulated_line = ""

        with Live(refresh_per_second=32, vertical_overflow="visible") as live:
            for chunks in SAMPLE_SPLIT:
                for chunk in chunks:
                    if chunk == "```":
                        in_code_block = not in_code_block

                    accumulated_line += chunk
                    live.update(Markdown(accumulated_line))
                    time.sleep(0.01)

                    # if it had two newlines, then split live once
                    if accumulated_line.endswith("\n\n"):



if __name__ == "__main__":
    print(NOTICE)
    dummy_exchange_turn()
