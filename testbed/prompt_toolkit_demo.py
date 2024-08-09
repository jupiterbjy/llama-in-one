"""
FOR ANYONE TRYING TO STUDY THE PROMPT_TOOLKIT:
READ PDF RATHER THAN READTHEDOCS, IT'S MORE READABLE AND EASIER TO SEARCH.

https://readthedocs.org/projects/python-prompt-toolkit/downloads/pdf/stable/

---

This is in total mess as I am right now using as scratch pad for learning prompt_toolkit.
"""

import asyncio
import itertools
import time
from collections.abc import Iterable

from prompt_toolkit import prompt, print_formatted_text, HTML, ANSI, PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import prompt

import pygments
from prompt_toolkit.styles import style_from_pygments_cls
from pygments.lexers import MarkdownLexer
from pygments.formatters import (
    HtmlFormatter,
    TerminalTrueColorFormatter,
    RawTokenFormatter,
)
from pygments.styles import get_style_by_name
from pygments.token import Token


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


tokens = [
    *pygments.lex(SAMPLE, lexer=MarkdownLexer(style=get_style_by_name("monokai")))
]

STYLE = style_from_pygments_cls(get_style_by_name("monokai"))


from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit import Application, HTML
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.application import create_app_session

from markdown_it import MarkdownIt


# TODO: refer this https://github.com/prompt-toolkit/python-prompt-toolkit/blob/master/examples/prompts/clock-input.py
def bottom_toolbar():
    # TODO: find out how to do the god darn buttons here
    return HTML("Press <b>Alt+Enter</b> to send.")


display_buffer = Buffer()
input_buffer = Buffer()
# history = FileHistory("history.txt")
history = InMemoryHistory()
command_completer = WordCompleter(["exit", "quit"])


async def interactive_shell():

    session = PromptSession(
        history=history,
        lexer=PygmentsLexer(MarkdownLexer),
        style=STYLE,
        color_depth=ColorDepth.TRUE_COLOR,
        completer=command_completer,
        multiline=True,
        bottom_toolbar=bottom_toolbar,
        enable_history_search=True,
    )

    input_pipe = create_pipe_input()

    while True:
        try:
            text = await session.prompt_async(
                ">> ",
            )

            if text in {"exit", "quit"}:
                break
        except (EOFError, KeyboardInterrupt):
            return

        with create_app_session(input=input_pipe):
            with create_pipe_input() as pipe:

                for token in delayed_iterator(tokens):
                    pipe.send_text(token[1])


if __name__ == "__main__":
    try:
        asyncio.run(interactive_shell())
    except Exception:
        import traceback

        traceback.print_exc()
        input()
        raise
