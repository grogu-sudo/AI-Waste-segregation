COMMAND_MAP = {
    "WET": "W",
    "DRY": "D",
    "METAL": "M"
}


def decide_command(label):

    return COMMAND_MAP.get(label, "D")
