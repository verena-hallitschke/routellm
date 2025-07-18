"""Module for defining text styles, intents, audiences, and formality levels."""

import random

STYLES = {
    "academic": "Use an academic style.",
    "business": "Write in a business style.",
    "general": "",
    # "email": "Write like an email.",
    "casual": "Write in a casual style.",
    "creative": "Use a creative style.",
}

INTENTS = {
    "inform": "Your goal is to inform the reader.",
    "describe": "Your goal is to inform the reader.",
    "convince": "Your goal is to convince the reader.",
    "storytelling": "Your goal is to tell the reader a story.",
}

AUDIENCE = {
    "general": "Your audience has general knowledge.",
    "intermediate": "Your audience has intermediate knowledge.",
    "expert": "Your audience has expert knowledge.",
}

FORMALITY_LEVELS = {
    "informal": "Use an informal tone.",
    "neutral": "",
    "formal": "Use a formal tone.",
}


def get_style(
    style: str = "general",
    intent: str = "inform",
    audience: str = "general",
    formality: str = "neural",
) -> str:
    """
    Get a text style based on the provided parameters.

    Args:
        style (str, optional): Style of the text. Can be one of: "academic", \
            "business", "general", "email", "casual", "creative". Defaults to "general".
        intent (str, optional): Intent of the text. Can be one of: "inform", \
            "describe", "convince", "storytelling". Defaults to "inform".
        audience (str, optional): Audience of the text. Can be one of: "general", \
            "intermediate", "expert". Defaults to "general".
        formality (str, optional): Level of formality. Can be one of: "informal", \
            "neutral", "formal". Defaults to "neural".

    Returns:
        str: A string describing the text style, combining the provided parameters.

    """
    assert style in STYLES.keys(), f'Unknown style "{style}".'
    assert intent in INTENTS.keys(), f'Unknown intent "{intent}".'
    assert audience in AUDIENCE.keys(), f'Unknown audience "{audience}".'
    assert formality in FORMALITY_LEVELS.keys(), (
        f'Unknown level of formality "{formality}".'
    )

    assert style != "academic" or formality != "informal", (
        'Cannot use "academic" style with "informal" formality!'
    )
    output_str = (
        f"{STYLES[style]} {INTENTS[intent]} {AUDIENCE[audience]} "
        + f"{FORMALITY_LEVELS[formality]}"
    )
    return output_str


def get_random_style(ignore_styles: list[str] | None = None) -> dict[str, str]:
    """
    Get a random text style, intent, audience, and formality level.

    Args:
        ignore_styles (list[str] | None, optional): List of styles to ignore. \
            Defaults to None.

    Returns:
        dict[str, str]: A dictionary containing the random style, intent, audience, \
            and formality level.

    """
    if ignore_styles is None:
        ignore_styles = []
    style = random.choice(
        list(filter(lambda x: x not in ignore_styles, list(STYLES.keys())))
    )

    return {
        "style": style,
        "intent": random.choice(list(INTENTS.keys())),
        "audience": random.choice(list(AUDIENCE.keys())),
        "formality": random.choice(
            list(
                filter(
                    lambda x: style != "academic" or x != "informal",
                    list(FORMALITY_LEVELS.keys()),
                )
            )
        ),
    }
