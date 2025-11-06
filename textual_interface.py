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
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+t", "toggle_theme", "Toggle Theme"),
    ]
    
    output_text = reactive("")
    
    def __init__(self):
        super().__init__()
        self.backend = None
        self.backend_loading = False
        self.last_words_count = 0
        self.current_worker = None
        self.debug_log = []
        self.current_input_text = ""

        self.source_lang = "fra_Latn"
        self.target_lang = "eng_Latn"
        self.len_input_sent = 0
        self.validated_translation = str()

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            yield Horizontal(
                Vertical(
                    Static("From:", classes="lang-label"),
                    Select(
                        ((language['name'], language['nllb']) for language in LANGUAGES),
                        value="fra_Latn",
                        id="source-lang", 
                        compact=True
                    ),
                    classes="column",
                ),
                Vertical(
                    Static("To:", classes="lang-label"),
                    Select(
                        ((language['name'], language['nllb']) for language in LANGUAGES),
                        value="eng_Latn",
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
    
    def on_mount(self) -> None:
        self._load_backend()
    
    def action_toggle_theme(self) -> None:
        self.theme = "catppuccin-latte" if self.theme == "textual-dark" else "textual-dark"
    
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
        if text and text.endswith(' '):
                return True
        return False
    
    def _get_status_text(self, text: str) -> str:
        if not text:
            return "[dim italic]Waiting for text...[/]"
        
        if self.backend_loading:
            return "[yellow]Loading model...[/]"
        
        word_count = len(text.strip().split())
        if word_count < 3:
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
            self.backend = None
            self.debug_log.clear()
            self.query_one("#debug-output", Static).update("")
            self._load_backend()
    
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes and update UI with results."""
        if event.state == WorkerState.SUCCESS:
            new_stable_translation, buffer, original_text = event.worker.result            
            self.validated_translation += (' ' + new_stable_translation.strip(' ')) if new_stable_translation.strip(' ') else "" #new for space since we send input with space but models return without except when puncuation
            # if not new_stable_translation and not buffer:
            #     return
            
            output = self.validated_translation
            if buffer:
                output += f"[$accent] {buffer.strip(' ')}[/]"
            
            self.query_one("#output", Static).update(output)
            
            debug_entry = f"""{original_text}| [$primary]"{new_stable_translation}"[/] [$accent]"{buffer}"[/] """
            self.debug_log.append(debug_entry)
            debug_text = "\n".join(self.debug_log)
            self.query_one("#debug-output", Static).update(debug_text)
        elif event.state == WorkerState.ERROR:
            self.log(f"Translation error: {event.worker.error}")
            self.query_one("#output", Static).update("[red]Translation error[/]")
        elif event.state == WorkerState.CANCELLED:
            self.log("Translation cancelled")
    

def main():
    """Application entry point."""
    app = TranslationApp()
    app.run()


if __name__ == "__main__":
    main()
