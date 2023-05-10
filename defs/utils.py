from typing import List


def get_script_name(argv0: str):
    return argv0.replace("\\", "/").split("/")[-1]


# Copied from IOT_pi/master_loop.py
def pop_flag_param(args: List[str], flag: str) -> "str | None":
    try:
        pos = args.index(flag)
        if pos == len(args) - 1:
            return None
        val = args[pos + 1]
        del args[pos:pos + 2]
        return val
    except ValueError:
        return None
