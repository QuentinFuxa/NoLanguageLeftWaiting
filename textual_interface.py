from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Input, Static, Select
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from nllw.core import TranslationBackend
from nllw.languages import LANGUAGES

class TranslationApp(App):
    """Interactive translation application with Textual."""

    CSS = """

    .lang-row {
        height: auto;
        margin-bottom: 1;
    }

    .column {
        margin-left: 3;
        margin-right: 3
    }

    .lang-label {
        color: $text-muted;
        margin: 0;
    }


    #input-container {
        layout: horizontal;
        border: round $primary;
        padding: 0 1;
        height: 3;
        align: left middle;
    }

    .prompt-symbol {
        color: $primary;
        text-style: bold;
        width: auto;
        height: auto;
    }

    #input-field {
        border: none;
        background: transparent;
        height: 1;
        padding: 0;
        width: 1fr;
    }

    #input-field:focus {
        border: none;
    }

    #output-container {
        border: round $accent;
        padding: 1 2;
        min-height: 8;
    }

    #output {
        padding: 0;
        min-height: 3;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+t", "toggle_theme", "Toggle Theme"),
    ]

    output_text = reactive("")

    def __init__(self, backend_type: str = "nllb", model_path: str = None,
                 heads_path: str = None):
        super().__init__()
        self.backend = None
        self.backend_loading = False
        self.last_words_count = 0
        self.current_worker = None
        self.debug_log = []
        self.current_input_text = ""

        self.backend_type = backend_type
        self.model_path = model_path
        self.heads_path = heads_path

        self.source_lang = "eng_Latn" if backend_type == "alignatt" else "fra_Latn"
        self.target_lang = "fra_Latn" if backend_type == "alignatt" else "eng_Latn"
        self.len_input_sent = 0
        self.validated_translation = str()

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            yield Horizontal(
                Vertical(
                    Static("From:", classes="lang-label"),
                    Select(
                        ((language['name'], language['nllb']) for language in LANGUAGES),
                        value=self.source_lang,
                        id="source-lang",
                        compact=True
                    ),
                    classes="column",
                ),
                Vertical(
                    Static("To:", classes="lang-label"),
                    Select(
                        ((language['name'], language['nllb']) for language in LANGUAGES),
                        value=self.target_lang,
                        id="target-lang",
                        compact=True
                    ),
                    classes="column",
                ),
            )

            with Container(id="input-container"):
                yield Static("> ", classes="prompt-symbol")
                yield Input(
                    placeholder="Type your text here...",
                    id="input-field"
                )

            with Container(id="output-container"):
                yield Static(id="output")

            with Container(id="debug-container"):
                yield Static(id="debug-output")

            yield Static("", id="status-bar")

    def on_mount(self) -> None:
        if self.backend_type == "alignatt":
            self.query_one("#output", Static).update(
                "[yellow]Loading AlignAtt model... (this may take a moment)[/]"
            )
            self.backend_loading = True
            self.run_worker(self._load_alignatt_async, thread=True, exclusive=True)
        else:
            self._load_backend()

    def action_toggle_theme(self) -> None:
        self.theme = "catppuccin-latte" if self.theme == "textual-dark" else "textual-dark"

    def _load_alignatt_async(self):
        """Load AlignAtt backend in a worker thread (model loading is slow)."""
        from nllw.alignatt_backend import AlignAttBackend
        return AlignAttBackend(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            model_path=self.model_path,
            heads_path=self.heads_path,
        )

    def _load_backend(self) -> None:
        if self.backend is None and not self.backend_loading:
            self.backend_loading = True
            try:
                self.query_one("#output", Static).update(
                    "[yellow]Loading translation model...[/]"
                )
                self.backend = TranslationBackend(
                    source_lang=self.source_lang,
                    target_lang=self.target_lang
                )
                self.query_one("#output", Static).update(
                    f"[green]Type {self.source_lang} text and press space to translate to {self.target_lang}.[/]"
                )
            except Exception as e:
                self.query_one("#output", Static).update(
                    f"[red]Error loading model: {str(e)}[/]"
                )
                self.backend = None
            finally:
                self.backend_loading = False

    def on_input_changed(self, event: Input.Changed) -> None:
        input_text = event.value

        if self.current_worker is not None and self.current_worker.state == WorkerState.RUNNING:
            self.current_worker.cancel()

        if not self._should_translate(input_text):
            status_text = self._get_status_text(input_text)
            if status_text:
                self.query_one("#output", Static).update(status_text)
        else:
            new_total_len = len(input_text)
            input_text = input_text[self.len_input_sent:]
            self.len_input_sent = new_total_len
            self.current_worker = self.run_worker(
                lambda: self._translate_async(input_text),
                thread=True,
                exclusive=True
            )

    def _should_translate(self, text: str) -> bool:
        if not self.backend:
            return False
        if text and text.endswith(' '):
                return True
        return False

    def _get_status_text(self, text: str) -> str:
        if not text:
            return "[dim italic]Waiting for text...[/]"

        if self.backend_loading:
            return "[yellow]Loading model...[/]"

        if not self.backend:
            return "[yellow]Loading model...[/]"

        word_count = len(text.strip().split())
        if word_count < 3 and self.backend_type != "alignatt":
            return f"[dim italic]Type at least 3 words to start translation... ({word_count}/3)[/]"

        return None

    def _translate_async(self, text: str) -> tuple:
        stable_translation, buffer = self.backend.translate(text)
        return stable_translation, buffer, text

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle language selection changes."""
        if event.select.id == "source-lang":
            self.source_lang = str(event.value)
        elif event.select.id == "target-lang":
            self.target_lang = str(event.value)
        if hasattr(self, 'backend') and self.backend is not None:
            if self.backend_type == "alignatt":
                try:
                    self.backend.set_target_lang(self.target_lang)
                    self.validated_translation = ""
                    self.len_input_sent = 0
                    self.query_one("#output", Static).update("")
                    self.query_one("#input-field", Input).value = ""
                    self.debug_log.clear()
                    self.query_one("#debug-output", Static).update("")
                    return
                except ValueError as e:
                    self.query_one("#output", Static).update(
                        f"[red]{e}[/]"
                    )
                    return
            self.backend = None
            self.validated_translation = ""
            self.len_input_sent = 0
            self.debug_log.clear()
            self.query_one("#debug-output", Static).update("")
            self._load_backend()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes and update UI with results."""
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result

            # Check if this is a model loading result (AlignAttBackend instance)
            if not isinstance(result, tuple):
                self.backend = result
                self.backend_loading = False
                self.query_one("#output", Static).update(
                    "[green]AlignAtt ready! Type text and press space to translate.[/]"
                )
                self.query_one("#status-bar", Static).update(
                    "AlignAtt + HY-MT | Press space after words to translate"
                )
                return

            new_stable_translation, buffer, original_text = result
            self.validated_translation += (' ' + new_stable_translation.strip(' ')) if new_stable_translation.strip(' ') else ""

            output = self.validated_translation
            if buffer:
                output += f"[$accent] {buffer.strip(' ')}[/]"

            self.query_one("#output", Static).update(output)

            debug_entry = f"""{original_text}| [$primary]"{new_stable_translation}"[/] [$accent]"{buffer}"[/] """
            self.debug_log.append(debug_entry)
            debug_text = "\n".join(self.debug_log)
            self.query_one("#debug-output", Static).update(debug_text)
        elif event.state == WorkerState.ERROR:
            self.backend_loading = False
            self.log(f"Error: {event.worker.error}")
            self.query_one("#output", Static).update(
                f"[red]Error: {event.worker.error}[/]"
            )
        elif event.state == WorkerState.CANCELLED:
            self.log("Translation cancelled")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key — finish the sentence (alignatt only)."""
        if self.backend_type != "alignatt" or not self.backend:
            return

        remaining = self.backend.finish()
        if remaining and remaining.strip():
            self.validated_translation += (
                (" " + remaining.strip()) if self.validated_translation else remaining.strip()
            )

        self.query_one("#output", Static).update(
            f"[green]{self.validated_translation}[/]"
        )

        self.backend.reset()
        self.validated_translation = ""
        self.len_input_sent = 0
        self.query_one("#input-field", Input).value = ""
        self.debug_log.clear()
        self.query_one("#debug-output", Static).update("")


def main():
    """Application entry point."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="NLLW Translation TUI")
    parser.add_argument(
        "--backend", choices=["nllb", "alignatt"], default="alignatt",
        help="Translation backend (default: alignatt)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HYMT_MODEL_PATH"),
        help="Path to GGUF model (for alignatt backend, or set HYMT_MODEL_PATH)"
    )
    parser.add_argument(
        "--heads", default=None,
        help="Path to alignment heads JSON (default: bundled universal heads)"
    )
    args = parser.parse_args()

    if args.backend == "alignatt" and not args.model:
        print(
            "AlignAtt backend requires a GGUF model path.\n\n"
            "Usage:\n"
            "  nllw-tui --model /path/to/HY-MT1.5-7B-Q8_0.gguf\n"
            "  nllw-tui --backend nllb  # fallback to NLLB\n\n"
            "Or set HYMT_MODEL_PATH:\n"
            "  export HYMT_MODEL_PATH=/path/to/HY-MT1.5-7B-Q8_0.gguf\n\n"
            "Download:\n"
            "  huggingface-cli download tencent/HY-MT1.5-7B-GGUF "
            "HY-MT1.5-7B-Q8_0.gguf --local-dir ."
        )
        import sys
        sys.exit(1)

    app = TranslationApp(
        backend_type=args.backend,
        model_path=args.model,
        heads_path=args.heads,
    )
    app.run()


if __name__ == "__main__":
    main()
