"""Errors whose message is written for the person using the app, so it may be shown as it is."""


class InputError(ValueError):
    """A request the user can fix (HTTP 400); its message is shown, so it never quotes patient text."""


class BusyError(ValueError):
    """A job is already running for this browser session (HTTP 409)."""


class ConflictError(ValueError):
    """The request was based on data someone else has changed since (HTTP 409); its message is shown."""
