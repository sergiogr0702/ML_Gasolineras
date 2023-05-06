from typing import List


def get_script_name(argv0: str):
    """
	Returns the name of the executed file given the full path to it (argv[0])
	"""
    return argv0.replace("\\", "/").split("/")[-1]


# Copied from IOT_pi/master_loop.py
def pop_flag_param(args: List[str], flag: str) -> "str | None":
    """
	Given the list of arguments passed to the programs and a flag, returns the value of said flag and removes both
	the flag and the value from the argument list.
	If the flag isn't present or it doesn't have a value, returns None.
	"""

    try:
        pos = args.index(flag)
        if pos == len(args) - 1:
            return None
        val = args[pos + 1]
        del args[pos:pos + 2]
        return val
    except ValueError:
        return None
