from __future__ import annotations

import abc
import os

import typing
from collections.abc import Callable, Iterable

import itertools

import numpy as np

if typing.TYPE_CHECKING:
    from .handler import Handler


class Stage(abc.ABC):
    """Abstract base class for stages

    Stage is run using ``run`` function.

    Runs every time if ``total<0`` else to completion.

    """

    name: str
    """Name of stage"""

    description: str = ""
    """Description of stage"""

    total: int
    """Total number of iterations"""

    def __init__(self, **kwargs):
        """All keyword arguments, ``kwargs``, are inserted into namespace"""

        self.update(**kwargs)

    def update(self, **kwargs):
        """Update namespace with keyword arguments, ``kwargs``"""

        return self.__dict__.update(kwargs)

    @property
    def kwargs(self):
        """Keyword arguments to clone the stage"""

        return self.__dict__

    @abc.abstractmethod
    def run(self, handler: Handler):
        """Run the stage with ``handler``"""


class SingleStage(Stage):
    """Single stage.

    Evaluates a function, ``func``.

    User-defined methods:

    - ``func(files) -> None`` : Primary function that does a calculation, and
    modifies the dictionary of ``files`` with the result.

    Runs only once.

    """

    total = 1

    @abc.abstractmethod
    def func(self, files: dict):
        """Primary function that does a calculation, and
        modifies the dictionary of ``files`` with the result."""

    def run(self, handler: Handler):
        """Run the stage with ``handler``"""

        self.func(handler.files)
        handler.set_progress(self.name, 1)
        handler.save()


class MultiStage(Stage):
    """Multi stage.

    User-defined methods:

    - ``setup_namespace(namespace) -> None`` : Sets up the stage's ``namespace``
    with the dictionary of ``files``. Runs every time.

    - ``setup_files(files) -> None`` : Sets up the dictionary of ``files``, e.g.
    adding new files. Only runs the first time.

    - ``func(n) -> y`` : Primary function that returns the result, ``y``, for each
    input, of index ``n``.

    - ``write_files(files, n, results) -> None`` : Writes the ``results`` of the
    primary function, ``func``, for the input indices, ``n``, to the ``files``
    dictionary. Periodically run according to the handler policy.

    Runs only to completion.

    """

    total = -1

    @abc.abstractmethod
    def setup_namespace(self, files: dict):
        """Setup stage namespace

        Sets up the dictionary of ``files``, e.g. adding new files. Only runs
        the first time.

        warning::
            ``total`` must be either an existing property or set here.

        """

    @abc.abstractmethod
    def setup_files(self, files: dict):
        """Setup dictionary of files

        Sets up the dictionary of ``files``, e.g. adding new files. Only runs
        the first time.

        """

    @abc.abstractmethod
    def write_files(self, files: dict, n: Iterable[int], results: tuple):
        """Write results to dictionary of files

        Writes the ``results`` of the primary function, ``func``, for the input
        indices, ``n``, to the ``files`` dictionary. Periodically run according
        to the handler policy.

        """

    @abc.abstractmethod
    def func(self, n: int) -> typing.Any:
        """Primary function that returns the result, ``y``, for each input, of
        index ``n``."""

    def _func_with_index_multi(self, n: Iterable[int]) -> tuple[Iterable[int], tuple]:
        """Function for iteration n - return value includes index"""

        return (n, tuple(map(self.func, n)))

    def _func_with_index_single(self, n: int) -> tuple[int, typing.Any]:
        """Function for iteration n - return value includes index"""

        return (n, self.func(n))

    def run(self, handler: Handler):
        """Run the stage with ``handler``"""

        self.setup_namespace(handler.files)
        handler.set_total(self.name, self.total)

        n_is_unfinished_file = f"{handler.data_directory}/n_is_unfinished.npy"
        progress = handler.get_progress(self.name)
        if progress == 0:
            self.setup_files(handler.files)
            n_is_unfinished = np.ones(self.total, dtype=np.bool_)
        else:
            n_is_unfinished = np.load(n_is_unfinished_file)
        unfinished_n = (
            n for n, is_unfinished in enumerate(n_is_unfinished) if is_unfinished
        )

        size = self.total - progress
        smallest_chunksize = handler.optimized_chunksize(size)

        if smallest_chunksize > 1:
            # chunks are tuples containing a similar number of elements
            largest_chunksize = smallest_chunksize + 1
            num_chunks, num_largest_chunks = divmod(size, smallest_chunksize)
            num_smallest_chunks = num_chunks - num_largest_chunks
            chunks = (
                tuple(itertools.islice(unfinished_n, chunksize))
                for chunksize in itertools.chain(
                    itertools.repeat(largest_chunksize, num_largest_chunks),
                    itertools.repeat(smallest_chunksize, num_smallest_chunks),
                )
            )

            def get_func(
                actor: typing.Type[MultiStage],
            ) -> Callable[[Iterable[int]], tuple[Iterable[int], typing.Any]]:
                return actor._func_with_index_multi

            # stitch together sequences of unzipped output
            process_output = lambda mixed_outputs: zip(
                *itertools.chain.from_iterable(
                    zip(*mixed_output) for mixed_output in mixed_outputs
                )
            )

        else:
            # chunks are a single element (scalar)
            num_chunks = size
            chunks = unfinished_n

            def get_func(
                actor: typing.Type[MultiStage],
            ) -> Callable[[int], tuple[int, typing.Any]]:
                return actor._func_with_index_single

            process_output = lambda mixed_outputs: zip(*mixed_outputs)

        for result_chunk in handler.evaluate_in_unordered_chunks(
            self, get_func, chunks, total=num_chunks
        ):
            n, results = process_output(result_chunk)
            n = np.asarray(n, dtype=int)
            num_finished = n.size

            n_is_unfinished[n] = False
            progress += num_finished
            handler.set_progress(self.name, progress)
            self.write_files(handler.files, n, results)

            handler.save()
            np.save(n_is_unfinished_file, n_is_unfinished)

        if progress != self.total or n_is_unfinished.any():
            raise RuntimeError("Map has been completed but results are missing")

        if os.path.exists(n_is_unfinished_file):
            os.remove(n_is_unfinished_file)


class PlotStage(Stage):
    """Plot stage.

    User-defined methods:

    - ``plot(files, data_directory) -> None`` : Plots the data in
    ``files``, optionally saving to the directory, ``data_directory``. Runs
    every time.

    Runs every time (``total=-1``).

    """

    total = -1

    @abc.abstractmethod
    def plot(self, files: dict, data_directory: str):
        """Plot function

        Plots the data in ``files``, optionally saving to the directory,
        ``data_directory``. Runs every time.

        """

    def run(self, handler: Handler):
        """Run the stage with ``handler``"""

        self.plot(handler.files, handler.data_directory)
