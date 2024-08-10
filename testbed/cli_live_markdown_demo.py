"""
Python Rich Live Console Demo for Dummy LLM stream output.

Since it's just whitespace & newline cut it's not accurate at all
but should be enough to test and fix long output Live data with various MD features.

:Author: jupiterbjy@gmail.com
"""

import asyncio
import itertools
import time
from datetime import datetime

from prompt_toolkit import Application, HTML
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import merge_key_bindings, KeyBindings
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import ScrollablePane, Layout, BufferControl
from prompt_toolkit.layout.containers import HSplit, FloatContainer, Window
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import style_from_pygments_cls, Style
from prompt_toolkit.widgets import Frame, TextArea, Label
from pygments.lexers import MarkdownLexer
from pygments.styles import get_style_by_name
from prompt_toolkit.key_binding.bindings.page_navigation import (
    scroll_page_up,
    scroll_page_down,
)


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
            for line in [*iter_with_separator(SAMPLE.strip().split("\n"), "\n")]
        )
    )
]


async def delayed_iterator(iterable, delay=0.03):
    """Delays item yield"""

    for item in iterable:
        await asyncio.sleep(delay)
        yield item


STYLE = Style(
    style_from_pygments_cls(get_style_by_name("monokai")).style_rules
    + Style.from_dict(
        {
            "chat_user": "#49D8FF",
            "chat_assistant": "#FC7BFF",
        }
    ).style_rules
)


def scroll_one_line_down() -> None:
    """
    scroll_offset += 1
    """
    w = get_app().layout.current_window
    b = get_app().current_buffer

    if w:
        # When the cursor is at the top, move to the next line. (Otherwise, only scroll.)
        if w.render_info:
            info = w.render_info

            if w.vertical_scroll < info.content_height - info.window_height:
                if info.cursor_position.y <= info.configured_scroll_offsets.top:
                    b.cursor_position += b.document.get_cursor_down_position()

                w.vertical_scroll += 1


class App(Application):
    kb = KeyBindings()
    kb.add("tab")(focus_next)
    kb.add("s-tab")(focus_previous)

    def __init__(self):
        self.buffer = Buffer()
        self.history = InMemoryHistory()
        self.completer = WordCompleter(
            [":exit", ":save", ":quit", ":load", ":temp", ":clear"]
        )

        self.current_area: TextArea | Label = Label(text="Model loaded")

        self.split = HSplit([self.current_area], padding=2, padding_char=" ")

        self.container = ScrollablePane(self.split)
        self.window = Window()

        super().__init__(
            layout=Layout(container=FloatContainer(self.container, [])),
            style=STYLE,
            key_bindings=merge_key_bindings(
                [
                    load_key_bindings(),
                    App.kb,
                ]
            ),
            mouse_support=True,
            full_screen=True,
            color_depth=ColorDepth.TRUE_COLOR,
        )

    def startup(self):
        """Extra startup setup"""

        self.send_message("", "user", "cyan")

    async def user_complete_message(self):

        await self.bot_complete_message()

        self.send_message("", "user", "cyan")

    async def bot_complete_message(self):
        self.send_message("", "assistant", "pink")

        async for token in delayed_iterator(SAMPLE_SPLIT):
            self.current_area.text += token
            self.container.keep_cursor_visible()
            # self.container.scroll_offsets()
            # self.container.vertical_scroll += 1
            # scroll_one_line_down()
            # self.current_area.window.allow_scroll_beyond_bottom()
            self.current_area.buffer.cursor_down()
            self.current_area.control.move_cursor_down()
            self.current_area.buffer.cursor_right(10000)

            self.invalidate()

    def send_message(self, msg: str, role: str = "", color=""):
        """Adds message to split"""

        self.current_area = TextArea(
            text=f"{msg}",
            lexer=PygmentsLexer(MarkdownLexer),
            history=self.history,
            dont_extend_height=True,
            scrollbar=True,
        )
        # self.current_area.control.preferred_height(5, 10, True, None)
        # self.current_area.window.preferred_height(5, 10)

        formatted = f"<{color}>{role}</{color}>" if color else role

        self.split.get_children().append(
            HSplit(
                [
                    Frame(
                        self.current_area,
                        title=HTML(f"{formatted} - <lime>{datetime.now()}</lime>"),
                        style=f"class:chat_{role}",
                    )
                ]
            ),
        )
        try:
            get_app().layout.focus(self.current_area)
        except ValueError:
            # probably first run before .run() call
            pass


async def driver():
    app = App()

    @App.kb.add_binding("escape", "enter")
    async def complete_message(event):
        await app.user_complete_message()

    @App.kb.add("c-c")
    async def send_command(event):
        pass

    @App.kb.add("c-c")
    def _exit(event) -> None:
        get_app().exit()

    await app.run_async(app.startup)


if __name__ == "__main__":
    asyncio.run(driver())
