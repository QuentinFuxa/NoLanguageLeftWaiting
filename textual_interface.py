from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, Static, Header, Footer
from textual.reactive import reactive
from translation_backend import TranslationBackend


class TranslationApp(App):
    """Interactive translation application with Textual."""
    
    CSS = """
    #main-container {
        height: auto;
        padding: 1;
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
            
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            yield Static("Input:", id="input-label")
            yield Input(
                placeholder="Type your text here...",
                id="input-field"
            )
            yield Static("Output:", id="output-label")
            yield Static(id="output")
        yield Footer()
    
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
                    source_lang='fra_Latn',
                    target_lang='eng_Latn'
                )
                self.query_one("#output", Static).update(
                    "[green]Type French text and press space to translate.[/]"
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
        
        processed_text = self.process_text(input_text)
        
        if processed_text:
            self.query_one("#output", Static).update(processed_text)
    
    def process_text(self, text: str) -> str:
        if not text:
            return "[dim italic]Waiting for text...[/]"
        
        if self.backend_loading:
            return "[yellow]Loading model...[/]"
        
        should_translate = False
        word_count = len(text.strip().split())
        if word_count < 3:
            return f"[dim italic]Type at least 3 words to start translation... ({word_count}/3)[/]"
        
        if text.endswith(' '):
            self.log('BBB:', str(text))
            if word_count >= self.last_words_count + 3: 
                should_translate = True
            
        if should_translate:
            self.log('Text ready to be translated:', text)
            stable_translation, buffer = self.backend.translate(text)
                        
            self.last_words_count = len(text.strip().split())
            if not stable_translation and not buffer:
                return None
            
            output = stable_translation
            if buffer:
                output += f"[#808080]{buffer}[/]"
            
            return output            
        return None

def main():
    """Application entry point."""
    app = TranslationApp()
    app.run()


if __name__ == "__main__":
    main()
