from typing import Literal

PLOT_MODE_T = Literal["svg"] | Literal["widget"] | Literal["png"]

plot_mode: PLOT_MODE_T | None = None

IN_CLUSTER = False


def auto_mode_pure(mode: PLOT_MODE_T | None) -> PLOT_MODE_T:
    if mode is not None:
        return mode
    if IN_CLUSTER:
        return "svg"
    else:
        return "widget"


def auto_mode(mode: PLOT_MODE_T | None) -> PLOT_MODE_T:
    global plot_mode
    plot_mode = auto_mode_pure(mode)
    return plot_mode


def init_matplotlib(mode: PLOT_MODE_T | None = None):
    from IPython.core.getipython import get_ipython

    ipython = get_ipython()
    if ipython is None:
        raise RuntimeError("Not in a notebook!")
    mode = auto_mode(mode)
    if mode == "svg" or mode == "png":
        import matplotlib_inline

        ipython.run_line_magic("matplotlib", "inline")
        matplotlib_inline.backend_inline.set_matplotlib_formats(mode)
    elif mode == "widget":
        ipython.run_line_magic("matplotlib", "widget")
