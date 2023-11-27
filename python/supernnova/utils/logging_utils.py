from time import time
from colorama import init, Fore, Back, Style
from contextlib import ContextDecorator


class timer(ContextDecorator):
    """
    Args:
        msg (str): string message to display when timing
        verbose (bool): If True, will print the timing.
            Default: ``False``
    """

    def __init__(self, msg):

        self.msg = msg

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        total = time() - self.start
        print()
        print_green(f"{self.msg}", f"{pretty_time(total)}")


def pretty_time(orig_seconds):
    """Transforms seconds into a string with days, hours, minutes and seconds."""
    days, seconds = divmod(round(orig_seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    out = []
    if days > 0:
        out.append(f"{days}d")
    if hours > 0:
        out.append(f"{hours}h")
    if minutes > 0:
        out.append(f"{minutes}m")
    if seconds > 0:
        out.append(f"{seconds}s")
    else:
        if out:
            s = ""
        elif orig_seconds == 0:
            s = "0s"
        else:
            s = "<0s"
        out.append(s)

    return "".join(out)


def print_bright(s):

    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value="", verbose=True):
    if verbose is False:
        return

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_yellow(info, value="", verbose=True):
    if verbose is False:
        return

    print(Fore.YELLOW + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value="", verbose=True):
    if verbose is False:
        return
    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def print_blue(info, value="", verbose=True):
    if verbose is False:
        return
    print(Fore.BLUE + "[%s] " % info + Style.RESET_ALL + str(value))


def str_to_brightstr(string):

    return Style.BRIGHT + "%s" % string + Style.RESET_ALL


def str_to_redstr(string):

    return Fore.RED + "%s" % string + Style.RESET_ALL


def str_to_bluestr(string):

    return Fore.BLUE + "%s" % string + Style.RESET_ALL


def str_to_yellowstr(string):

    return Fore.YELLOW + "%s" % string + Style.RESET_ALL


def str_to_greenstr(string):

    return Fore.GREEN + "%s" % string + Style.RESET_ALL
