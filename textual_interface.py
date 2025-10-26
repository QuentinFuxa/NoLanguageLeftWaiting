from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Input, Static, Select
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from translation_backend import TranslationBackend


LANGUAGE_OPTIONS = [
    ("French", "fra_Latn"),
    ("English", "eng_Latn"),
    ("Spanish", "spa_Latn"),
    ("German", "deu_Latn"),
    ("Italian", "ita_Latn"),
    ("Portuguese", "por_Latn"),
    ("Dutch", "nld_Latn"),
    ("Russian", "rus_Cyrl"),
    ("Chinese (Simplified)", "zho_Hans"),
    ("Chinese (Traditional)", "zho_Hant"),
    ("Japanese", "jpn_Jpan"),
    ("Korean", "kor_Hang"),
    ("Arabic", "arb_Arab"),
    ("Hindi", "hin_Deva"),
    ("Bengali", "ben_Beng"),
    ("Vietnamese", "vie_Latn"),
    ("Thai", "tha_Thai"),
    ("Turkish", "tur_Latn"),
    ("Polish", "pol_Latn"),
    ("Czech", "ces_Latn"),
    ("Hungarian", "hun_Latn"),
    ("Finnish", "fin_Latn"),
    ("Swedish", "swe_Latn"),
    ("Norwegian Bokmål", "nob_Latn"),
    ("Danish", "dan_Latn"),
    ("Greek", "ell_Grek"),
    ("Hebrew", "heb_Hebr"),
    ("Romanian", "ron_Latn"),
    ("Bulgarian", "bul_Cyrl"),
    ("Croatian", "hrv_Latn"),
    ("Serbian", "srp_Cyrl"),
    ("Ukrainian", "ukr_Cyrl"),
    ("Catalan", "cat_Latn"),
]

class TranslationApp(App):
    """Interactive translation application with Textual."""
    
    CSS = """
    #main-container {
        padding: 1;
        background: white;
    }
    #input-label {
        margin-bottom: 1;
    }
    #output-label {
        margin-top: 2;
        margin-bottom: 1;
    }
    #output {
        padding: 1;
        min-height: 3;
    }
    #debug-label {
        margin-top: 2;
        margin-bottom: 1;
    }
    #debug-output {
        padding: 1;
        min-height: 3;
        border: solid $primary;
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
        self.theme = "catppuccin-latte"
        self.current_worker = None
        self.debug_log = []
        self.current_input_text = ""

        self.source_lang = "fra_Latn"
        self.target_lang = "eng_Latn"
            
    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            yield Horizontal(
                Vertical(
                    Static("From:", classes="lang-label"),
                    Select(
                        ((name, nllb_name) for name, nllb_name in LANGUAGE_OPTIONS),
                        value="fra_Latn",
                        id="source-lang"
                    ),
                    classes="column",
                ),
                Vertical(
                    Static("To:", classes="lang-label"),
                    Select(
                        ((name, nllb_name) for name, nllb_name in LANGUAGE_OPTIONS),
                        value="eng_Latn",
                        id="target-lang"
                    ),
                    classes="column",
                ),
            )

            yield Static("Input:", id="input-label")
            yield Input(
                placeholder="Type your text here...",
                id="input-field"
            )
            yield Static("Output:", id="output-label")
            yield Static(id="output")
            # yield Static("Debug Output:", id="debug-label")
            # yield Static(id="debug-output")
    
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
            self.current_input_text = input_text
            self.current_worker = self.run_worker(
                lambda: self._translate_async(input_text),
                thread=True,
                exclusive=True
            )
    
    def _should_translate(self, text: str) -> bool:
        if not text or self.backend_loading or self.backend is None:
            return False
        
        word_count = len(text.strip().split())
        if word_count < 3:
            return False
        
        if text.endswith(' '):
            if word_count >= self.last_words_count + 1:
                self.last_words_count = word_count
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
            # self.debug_log.clear()
            # self.query_one("#debug-output", Static).update("")
            self._load_backend()
    
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes and update UI with results."""
        if event.state == WorkerState.SUCCESS:
            stable_translation, buffer, original_text = event.worker.result            
            
            if not stable_translation and not buffer:
                return
            
            output = stable_translation
            if buffer:
                output += f"[yellow]{buffer}[/]"
            
            self.query_one("#output", Static).update(output)
            
            # debug_entry = f"{original_text}: {stable_translation} | {buffer}"
            # self.debug_log.append(debug_entry)
            # debug_text = "\n".join(self.debug_log)
            # self.query_one("#debug-output", Static).update(debug_text)
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
