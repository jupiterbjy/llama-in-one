from prompt_toolkit import Application, HTML
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.lexers import PygmentsLexer

from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.styles import get_style_by_name

from markdown_it import MarkdownIt
from pygments.lexers import MarkdownLexer


# https://stylishthemes.github.io/Syntax-Themes/pygments/
STYLE = style_from_pygments_cls(get_style_by_name("monokai"))


buffer1 = Buffer()  # Editable buffer.


root_container =  Window(content=BufferControl(buffer=buffer1)),
        # A vertical line in the middle. We explicitly specify the width, to
        # make sure that the layout engine will not try to divide the whole
        # width by three for all these windows. The window will simply fill its
        # content by repeating this character.
        Window(width=1, char="|"),
        # Display the text 'Hello world' on the right.
        Window(
            content=BufferControl(buffer=buffer1, lexer=PygmentsLexer(MarkdownLexer))
        ),
    ]
)

layout = Layout(root_container)

app = Application(
    layout=layout, full_screen=True, color_depth=ColorDepth.TRUE_COLOR, style=STYLE
)
app.run()  # You won't be able to Exit this app
