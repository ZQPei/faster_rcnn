from termcolor import cprint

def log_print(text, color=None, on_color=None, attrs=None):
    cprint(text, color=color, on_color=on_color, attrs=attrs)