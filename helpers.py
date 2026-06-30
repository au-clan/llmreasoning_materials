from IPython.display import Latex, display
import re

def show(runs, stage):
    r"""Render one stage's answer; convert the model's \( \) / \[ \] math
    delimiters to $ $ / $$ $$ so the notebook renders the LaTeX nicely."""
    r = runs[stage]
    text = (r['response'].replace(r'\(', '$').replace(r'\)', '$')
                         .replace(r'\[', '$$').replace(r'\]', '$$'))
    display(Latex(text))

def extract_boxed(text):
    """Pull the last \\boxed{...} answer if present."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None

def think_span(text):
    """Return the thinking span text (between forced <think> and </think>)."""
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0]
    return text
