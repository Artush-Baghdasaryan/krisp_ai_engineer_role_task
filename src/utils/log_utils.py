import logging


def log_usage(usage: object | None, context: str = "") -> None:
    if usage is None:
        return

    logger = logging.getLogger(__name__)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)

    if prompt is not None or completion is not None:
        msg = "Tokens usage" + (f" ({context})" if context else "") + ":"
        if prompt is not None:
            msg += f" prompt={prompt}"

        if completion is not None:
            msg += f" completion={completion}"

        if total is not None:
            msg += f" total={total}"

        logger.info(msg)
