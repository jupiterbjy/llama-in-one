"""
A cross-platform script for the automatic llama 3 8B setup cpu-only, single user setup for my laziness.

This is intended to be used for Godot plugin.

Dependency installation:
```
py -m pip install psutil llama-cpp-python httpx --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```


:Author: jupiterbjy@gmail.com

: MIT License
:
: Copyright (c) 2024 jupiterbjy@gmail.com
:
: Permission is hereby granted, free of charge, to any person obtaining a copy
: of this software and associated documentation files (the "Software"), to deal
: in the Software without restriction, including without limitation the rights
: to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
: copies of the Software, and to permit persons to whom the Software is
: furnished to do so, subject to the following conditions:
:
: The above copyright notice and this permission notice shall be included in all
: copies or substantial portions of the Software.
:
: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
: IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
: FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
: AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
: LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
: OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
: SOFTWARE.
"""

import base64
import itertools
import json
import pathlib
import logging
import argparse
import time
import zlib
from typing import List, TypedDict, Tuple, Callable, Any, Dict
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import httpx
from llama_cpp import Llama, LlamaCache
from psutil import cpu_count


# --- DEFAULT CONFIG ---

class Config:
    """Default config class holding envvars and constants.
    Instance this and change to your liking."""

    # -- GLOBAL SETUP --

    # Prefix for commands
    COMMAND_PREFIX = ":"

    # Subdirectory used for saving downloaded model files
    LLM_SUBDIR = "_llm"

    # Subdirectory used for saved sessions
    SESSION_SUBDIR = "_session"

    MODEL_PATH = pathlib.Path(__file__).parent / LLM_SUBDIR

    SESSION_PATH = MODEL_PATH.parent / SESSION_SUBDIR

    # chat seperator
    CHAT_SEP = "------------------------------------------------"

    def __init__(self):
        # link to model url.
        self.model_url = (
            "https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-32k-v0.1-GGUF/resolve/main/"
            "Llama-3-8B-Instruct-32k-v0.1.Q6_K.gguf"
        )
        # https://huggingface.co/bartowski/Gemma-2-9B-It-SPPO-Iter3-GGUF/resolve/main/Gemma-2-9B-It-SPPO-Iter3-Q6_K.gguf
        # TODO: Add lexi v2 if it releases

        self.model_name = pathlib.Path(self.model_url).name

        # Initial prompt added to start of chat.
        self.init_prompt = "You are an assistant who proficiently answers to questions."

        # Default seed. Change this value to get reproducible results.
        self.seed = -1

        # Default temperature for model
        self.temp = 0.7

        # CPU Thread Count
        self.n_threads = cpu_count(logical=False)

        # Input token length
        self.input_length = 32768

        # Context window length
        self.context_length = 32768

        # llama-cpp-python verbose flag
        self.verbose = False

        self.MODEL_PATH.mkdir(exist_ok=True)
        self.SESSION_PATH.mkdir(exist_ok=True)

    def json_serialize(self) -> str:
        """Serializes config to json string."""

        return json.dumps(
            {
                "model_url": self.model_url,
                "init_prompt": self.init_prompt,
                "seed": self.seed,
                "temp": self.temp,
                "n_threads": self.n_threads,
                "input_length": self.input_length,
                "context_length": self.context_length,
                "verbose": self.verbose,
            }
        )

    @classmethod
    def json_deserialize(cls, serialized: str) -> "Config":
        """Deserializes json string to Config instance."""

        data = json.loads(serialized)
        config = cls()

        config.model_url = data["model_url"]
        config.init_prompt = data["init_prompt"]
        config.seed = data["seed"]
        config.temp = data["temp"]
        config.n_threads = data["n_threads"]
        config.input_length = data["input_length"]
        config.context_length = data["context_length"]
        config.verbose = data["verbose"]
        config.command_prefix = data["command_prefix"]

        return config


# --- LOGGER CONFIG ---

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] <%(funcName)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# --- UTILITIES ---


@contextmanager
def progress_manager(size: int):
    """Prints progress. Execute manager with incremental.
    Written to skip TQDM requirement."""

    digits = len(str(size))
    accumulated = 0
    spinny_spin_boy = itertools.cycle("|/-\\")

    def progress(amount):
        nonlocal accumulated
        accumulated += amount
        print(
            f"{next(spinny_spin_boy)} {int(100 * accumulated / size):>3}% | {accumulated:{digits}}/{size}",
            end="\r",
        )

    # print the 0 progress first
    progress(0)

    # yield progress advancing function
    try:
        yield progress
    finally:
        # advance to newline as cursor is at \r
        print()


class Message(TypedDict):
    role: str
    content: str


def extract_message(response: dict) -> Tuple[str, Message]:
    """Extracts message from output and returns (stop reason, message)"""

    return response["choices"][0]["finish_reason"], response["choices"][0]["message"]


class StreamWrap:
    """Makes return reason available"""

    def __init__(self, gen: Generator):
        self._gen = gen
        self.reason = ""

    def __iter__(self):
        self.reason = yield from self._gen
        return self.reason


def get_user_input() -> str:
    """Get potentially multiline input from user
    Can't do multiline editing though."""

    lines = []

    print("[You]\n>> ", end="")

    # listen until sentinels - defaults are Ctrl+D for *nix and Ctrl+Z for windows.

    try:
        while True:
            line = input()

            # check if it ends with terminal. \x04 automatically stripped so no checks for it
            if line.endswith("\x1a"):
                lines.append(line.strip("\x1a"))
                break

            lines.append(line)
    except EOFError:
        pass

    return "\n".join(lines)


# --- WRAPPER ---


class LLMWrapper:
    """Wraps Llama for easier access"""

    def __init__(
        self,
        config: Config,
        *args,
        **kwargs,
    ):
        self.model_path = config.MODEL_PATH / pathlib.Path(config.model_url).name

        self.llm = Llama(
            self.model_path.as_posix(),
            seed=config.seed,
            n_ctx=config.context_length,
            verbose=config.verbose,
            n_threads=config.n_threads,
            *args,
            **kwargs,
        )

    def __str__(self):
        return f"LLMWrapper({self.model_path.name})"

    def create_chat_completion(self, messages: List[dict], **kwargs) -> Iterator:
        """Creates chat completion. This just wraps the original function for type hinting."""

        return self.llm.create_chat_completion(messages, **kwargs)

    def create_chat_completion_stream(self, messages: List[dict], **kwargs):
        """Creates stream chat completion."""

        return self.llm.create_chat_completion(messages, **kwargs)

    def set_cache(self, cache: LlamaCache):
        """
        # Uses cache for faster digest by keeping the state
        # https://github.com/abetlen/llama-cpp-python/issues/44#issuecomment-1509882229
        """
        self.llm.set_cache(cache)


# --- LLM Manager ---

class LLMInstances:
    """Manages global LLMWrapper instances per model"""

    # model url: LLMWrapper pair
    models: Dict[str, LLMWrapper] = {}

    @classmethod
    def get_model(cls, model_name: str, config: Config) -> LLMWrapper:
        """Get model by url. If not exists, create one and return."""

        if model_name not in cls.models:
            cls._ensure_downloaded(config)
            cls.models[model_name] = LLMWrapper(config)

        return cls.models[model_name]

    @classmethod
    def _ensure_downloaded(cls, config: Config):
        """Make sure file exists, if not download from self.model_url.
        This is to strip huggingface module dependency."""

        # if exists then return, no hash check cause lazy
        path = config.MODEL_PATH / config.model_name
        if path.exists():
            LOGGER.info(f"Found model {config.model_name}")
            return

        LOGGER.info(f"Downloading from {config.model_url}")

        # write with different extension
        temp = path.with_suffix(".temp")

        with (
            httpx.stream("GET", config.model_url, follow_redirects=True) as stream,
            temp.open("wb") as fp,
        ):
            length = int(stream.headers["content-length"])

            with progress_manager(length) as progress:
                for data in stream.iter_bytes():
                    fp.write(data)
                    progress(len(data))

        # rename back, we succeeded.
        temp.rename(path)
        LOGGER.info("Download complete")


class ChatSession:
    """Represents single chat session"""

    def __init__(
        self,
        title: str,
        config: Config,
        init_prompt: str = "",
    ):
        self.title = title
        self.llm = LLMInstances.get_model(config.model_name, config)

        self.config = config

        # self.resp_format = None
        # self.preprocessor = lambda x: x
        #
        # if output_json:
        #     self.preprocessor = json.loads
        #     prompt += " You outputs in JSON."
        #     self.resp_format = {"type": "json_object"}

        self.messages: List[Message] = []

        prompt = f"{config.init_prompt} {init_prompt}".strip()
        self.system_send(prompt)

        self.cache = LlamaCache()

    def __str__(self):
        return f"ChatSession({self.title})"

    def serialize(self) -> bytes:
        """Serializes and compress session into plain text"""

        raw = json.dumps(
            {
                "title": self.title,
                "config": self.config.json_serialize(),
                "messages": self.messages,
            }
        )
        # https://stackoverflow.com/a/4845324/10909029
        compressed = zlib.compress(raw.encode("utf8"))
        return compressed

    @classmethod
    def deserialize(cls, compressed: bytes):
        """Deserializes session"""

        serialized = zlib.decompress(base64.b64decode(compressed))
        data = json.loads(serialized)

        session = cls(
            data["title"],
            Config.json_deserialize(data["config"]),
        )
        session.messages = json.loads(data["messages"])

        return session

    def save_session(self):
        """Saves session in SESSION_SUBDIR."""

        with open(self.config.SESSION_PATH / f"{self.title}", "wb") as fp:
            fp.write(self.serialize())

    @classmethod
    def load_session(cls, uuid: str) -> "ChatSession":
        """Opens session from SESSION_SUBDIR.

        Raises:
            FileNotFoundError: When session file for given UUID is missing.
        """

        with open(Config.SESSION_PATH / f"{uuid}", "rb") as fp:
            return ChatSession.deserialize(fp.read())

    def clear(self, new_init_prompt):
        """Clears session history excluding first system prompt.
        Sets new first system prompt if provided."""

        first_msg = self.messages[0]
        self.messages.clear()

        if new_init_prompt is not None:
            first_msg["content"] = new_init_prompt

        self.messages.append(first_msg)

    def system_send(self, prompt: str):
        """Sends system prompt(system role message)."""

        self.messages.append(
            {
                "role": "system",
                "content": prompt,
            }
        )

    def send_discarding_reply(self, content: str):
        """Send message while ignoring reply it generate.
        This is alternative when system role is not supported."""

    def get_reply_stream(self, content: str, role="user") -> Generator[str]:
        """get_reply with streaming. Does not support json output mode.
        Returns finish reason."""

        # append user message
        self.messages.append({"role": role, "content": content})

        # generate message and append back to message list
        output = self.llm.create_chat_completion_stream(
            messages=self.messages,
            temperature=self.config.temp,
            max_tokens=self.config.input_length,
            stream=True,
        )

        # type hint to satisfy linter
        current_role = ""
        current_role_output = ""
        finish_reason = "None"

        for chunk in output:
            delta = chunk["choices"][0]["delta"]

            # if there's finish reason update it.
            if chunk["choices"][0]["finish_reason"] is not None:
                finish_reason = chunk["choices"][0]["finish_reason"]

            if "role" in delta:
                current_role = str(delta["role"])

            elif "content" in delta:
                current_role_output += delta["content"]
                yield delta["content"]

        self.messages.append({"role": current_role, "content": current_role_output})
        return finish_reason


# --- COMMANDS ---


class CommandMap:
    """Class that acts like a command map. Each method is single command in chat session.
    Commands can either return False to stop the session, or True to continue.

    Originally was `Dict[str, Callable[[ChatSession, Any], ...]]` like this:

    COMMAND_MAP: Dict[str, Callable[[ChatSession, Any], bool]] = {
        "exit": lambda _session, param: exit(),
        "clear": lambda _session, param: _session.clear(),
        "temp": lambda _session, param:
    }

    ... but changed to class to make type hint work and better readability.
    You can still add new methods to class in runtime anyway!

    Notes:
        This is to be instanced so new command can be added in runtime if needed.
    """

    def command(self, session: ChatSession, name: str, arg=None) -> bool:
        """Search command via getattr and executes it.

        Raises:
            NameError: When given command doesn't exist.
            ValueError: When given argument for command is invalid.

        Returns:
            True if chat should continue, else False.
        """

        try:
            func: Callable[[ChatSession, Any], bool] = getattr(self, name)
        except AttributeError as err:
            raise NameError("No such command exists.") from err

        return func(session, arg)

    @staticmethod
    def exit(_session, _) -> bool:
        """Exits the session."""
        print("Exiting!")
        return False

    @staticmethod
    def clear(session: ChatSession, new_init_prompt) -> bool:
        """Clears chat histories and set new initial prompt if any.
        Otherwise, will fall back to previous initial prompt."""

        session.clear(new_init_prompt)
        print("Session cleared.")
        return True

    @staticmethod
    def temperature(session: ChatSession, amount_str: str) -> bool:
        """Set model temperature to given value.

        Raises:
            ValueError: On invalid amount string
        """

        session.temperature = float(amount_str)
        print(f"Temperature set to {session.config.temp}.")
        return True

    @staticmethod
    def system(session: ChatSession, prompt: str) -> bool:
        """Give system prompt (system role message) to llm."""

        session.system_send(prompt)
        print(f"System prompt sent.")
        return True

    @staticmethod
    def save(session: ChatSession, _) -> bool:
        """Save the chat session."""

        session.save_session()
        print(f"Session saved as '{session.title}'.")
        return True


# --- STANDALONE MODE RUNNER ---


class StandaloneMode:
    def __init__(self, verbose=False, start_new_session=False, load_session="") -> None:
        self.session: ChatSession | None = None

        self.config: Config = Config()

        self.command_map = CommandMap()

        self.config.verbose = verbose

        self.start_new_session = start_new_session
        self.load_session = load_session

    def menu(self):
        """Show menu"""

        menus = ["Create new session", "Load session"]

        print("\n".join(f"{idx}. {line}" for idx, line in enumerate(menus, start=1)))

        while True:

            # validate choice
            try:
                choice = int(input(">> "))
                assert 0 < choice <= len(menus)

            except (ValueError, AssertionError):
                continue

            match choice:
                case 1:
                    self.session = ChatSession(input("Chat Title >> "), self.config)
                    return
                case 2:
                    # validate uuid
                    try:
                        self.session = ChatSession.load_session(
                            input("Chat Title >> ")
                        )
                    except FileNotFoundError:
                        continue

    def prep_session(self):
        """Prepares sessions by either creating new or loading existing one."""

        if self.start_new_session:
            self.session = ChatSession(
                f"new_chat_{int(time.time())}",
                self.config,
            )
        elif self.load_session:
            try:
                self.session = ChatSession.load_session(self.load_session)
            except FileNotFoundError:
                print(f"Session '{self.load_session}' not found.")
                self.menu()
        else:
            self.menu()

    def _run_command(self, user_input: str) -> bool:
        """Try to run given command return boolean whether session should continue or not."""

        if not user_input.startswith(Config.COMMAND_PREFIX):
            return False

        print("[Command]")

        # it's some sort of command. cut at first whitespace if any.
        sections = user_input[len(Config.COMMAND_PREFIX):].split(" ", maxsplit=1)

        continue_session = True

        try:
            continue_session = self.command_map.command(
                self.session,
                sections[0],
                None if len(sections) == 1 else sections[1],
            )
        except Exception as err:
            print(type(err).__name__, err)

        # print newline afterward for consistency
        print()
        return continue_session

    def _exchange_turn(self) -> bool:
        """Exchanges chat turn and returns False if chat should stop."""

        print(self.config.CHAT_SEP)
        user_input = get_user_input()

        # check if it was empty - if so, return so that we get new prompt.
        if not user_input.strip():
            return True

        print("\n" + self.config.CHAT_SEP)
        if user_input.startswith(self.config.COMMAND_PREFIX):
            return self._run_command(user_input)

        print("[Bot]")
        gen = StreamWrap(self.session.get_reply_stream(user_input))

        # flush token by token, so it doesn't group up and print at once
        # people willingly wait some extra overhead to complete the sentence to see the progress
        for token in gen:
            print(token, end="", flush=True)

        print(f"\n\n[Stop reason: {gen.reason}]")

        return True

    def run(self):
        """Runs standalone mode in loop."""

        self.prep_session()

        print("Ctrl+Z + Enter (win) / Ctrl+D (nix) to complete message.")

        run = True

        while run:
            run = self._exchange_turn()


# --- MAIN ---

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enables verbose output.",
    )
    _parser.add_argument(
        "-n",
        "--new-session",
        action="store_true",
        default=False,
        help="Create new session on start."
    )
    _parser.add_argument(
        "-l",
        "--load-session",
        type=str,
        default="",
        help="Load session by given title on start.",
    )

    _args = _parser.parse_args()

    _runner = StandaloneMode(_args.verbose, _args.new_session, _args.load_session)
    _runner.run()
