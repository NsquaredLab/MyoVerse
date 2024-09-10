import argparse
from typing import Any, Dict, List, Sequence, Tuple


class ArgParserPlus:
    """Wrapper for argparse for added safety.

    Parameters
    ----------
    description : str
        Description of the program.
    advanced : bool
        If True, arguments will not use the default if not given and will raise an error.
    arguments : List[Tuple[Sequence[str], Dict[str, Any]]]
        List of arguments to be added to the parser.
        Each argument is a tuple of a sequence of strings and a dictionary of keyword arguments.
        The sequence of strings is the name of the argument and its short form.
        The dictionary of keyword arguments is the keyword arguments for the argument.
        The default value is set to argparse.ArgumentError if advanced is True.

    References
    ----------
    https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
    """

    def __init__(
        self, description: str, advanced: bool = False, arguments: List[Tuple[Sequence[str], Dict[str, Any]]] = None
    ):
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.advanced = advanced

        # if arguments are given, add them
        if arguments is not None:
            for argument in arguments:
                self.add_argument(argument)

    def add_argument(self, argument: Tuple[Sequence[str], Dict[str, Any]]):
        """Add an argument to the parser.

        Arguments
        ---------
        argument : Tuple[Sequence[str], Dict[str, Any]]
            Tuple of a sequence of strings and a dictionary of keyword arguments.
            The sequence of strings is the name of the argument and its short form.
            The dictionary of keyword arguments is the keyword arguments for the argument.
            The default value is set to argparse.ArgumentError if advanced is True.
        """

        if self.advanced:
            argument[1]["default"] = argparse.ArgumentError

        self.parser.add_argument(*argument[0], **argument[1])

    def parse_args(self) -> argparse.Namespace:
        """Parse the arguments.

        If advanced is True, arguments that are required will raise an error if not given.

        Returns
        -------
        argparse.Namespace
            Namespace of the arguments.

        Raises
        ------
        argparse.ArgumentError
            If advanced is True and an argument is not given a value different from the default.
        """
        for argument in self.parser._actions:
            if argument.default == argparse.ArgumentError:
                raise argparse.ArgumentError(argument, "Argument is required")
        return self.parser.parse_args()


if __name__ == "__main__":
    parser = ArgParserPlus(
        "Test",
        advanced=True,
        arguments=[
            (["--experimentee", "-e"], {"type": int, "default": 1801211219}),
        ],
    )

    args = print(parser.parse_args())
