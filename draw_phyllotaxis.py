"""
Drawing a Phyllotaxis pattern via turtle graphics.
Idea: https://www.geeksforgeeks.org/python/algorithmic-botany-phyllotaxis-python/
"""

from typing import (
    Callable,
    Literal,
    Final,
    Generator,
    Protocol,
    NewType,
    Iterator,
    Self,
    overload,
    Optional,
)
from types import MappingProxyType
import math
import turtle
import random
from functools import partial, wraps
import logging
from collections.abc import Sequence
from contextlib import contextmanager
import sys
from time import perf_counter
from dataclasses import dataclass
from enum import StrEnum
from time import time
from memory_profiler import profile as profile_memory

type SpacingScalingCoefficient = int | float
type TurtleSpeed = int | float

ConverterCoefficient = NewType("ConverterCoefficient", float)
Degrees = NewType("Degrees", float)
Radians = NewType("Radians", float)

type TurtleShape = Literal["arrow", "turtle", "circle", "square", "triangle", "classic"]


class PetalColor(StrEnum):
    YELLOW = "yellow"
    GOLD = "gold"
    LIGHT_YELLOW = "lightyellow"


PETAL_COLORS: Final[Sequence[PetalColor]] = tuple(PetalColor)


PHYLLOTAXIS_CONSTANTS: Final[MappingProxyType[str, Degrees | ConverterCoefficient]] = (
    MappingProxyType(
        {
            "GOLDEN_ANGLE_IN_DEGREES": Degrees(137.50776405),
            "MULTIPLIER_FROM_DEGREE_TO_RADIANS": ConverterCoefficient(0.0174532925),
        }
    )
)


@overload
def get_phyllotaxis_constant(
    *,
    constant_name: Literal["GOLDEN_ANGLE_IN_DEGREES"],
) -> Degrees: ...


@overload
def get_phyllotaxis_constant(
    *,
    constant_name: Literal["MULTIPLIER_FROM_DEGREE_TO_RADIANS"],
) -> ConverterCoefficient: ...


def get_phyllotaxis_constant(
    *,
    constant_name: str,
) -> Degrees | ConverterCoefficient:
    return PHYLLOTAXIS_CONSTANTS[constant_name]


@dataclass(frozen=True, slots=True)
class PointInCartesianCoordinates:
    first_coordinate: float
    second_coordinate: float

    def __iter__(self) -> Iterator[float]:
        yield self.first_coordinate
        yield self.second_coordinate

    @classmethod
    def from_polar_coordinates(
        cls,
        *,
        distance_from_center: float,
        position_angle_in_radians: Radians,
    ) -> Self:
        return cls(
            first_coordinate=(
                distance_from_center * math.cos(position_angle_in_radians)
            ),
            second_coordinate=(
                distance_from_center * math.sin(position_angle_in_radians)
            ),
        )

    def __post_init__(self) -> None:
        if not all(math.isfinite(coordinate) for coordinate in self):
            raise CartesianCoordinatesMustBeFinite()


class SupportsShapeFilling(Protocol):
    def begin_fill(self) -> None: ...

    def end_fill(self) -> None: ...


@contextmanager
def fill_shape(*, actual_turtle: SupportsShapeFilling) -> Generator[None, None, None]:
    actual_turtle.begin_fill()
    try:
        yield
    finally:
        actual_turtle.end_fill()


class DefaultExceptionMessageMeta(type):
    def __call__(cls, *arguments: object) -> Exception:
        if not arguments:
            arguments = (cls.__doc__ or cls.__name__,)
        return super().__call__(*arguments)


class PhyllotaxisExceptions(Exception, metaclass=DefaultExceptionMessageMeta):
    """Base class for phyllotaxis exceptions."""


class AngularStepShouldBeAround137Point50776405Degrees(PhyllotaxisExceptions):
    """Angular step should be indeed around 137.50776405 degrees."""


class AngularStepShouldBeAroundTheGoldenAngle(PhyllotaxisExceptions):
    """Angular step should be indeed around the Golden Angle."""


class CartesianCoordinatesMustBeFinite(PhyllotaxisExceptions):
    """Cartesian coordinates must be finite numbers."""


class PetalColorIteratorExhausted(PhyllotaxisExceptions):
    """The color iterator must supply a color for every petal."""


def log_execution_time[**Parameters, ReturnValue](
    *,
    log_level: int = logging.INFO,
) -> Callable[
    [Callable[Parameters, ReturnValue]],
    Callable[Parameters, ReturnValue],
]:
    def decorator(
        function: Callable[Parameters, ReturnValue],
    ) -> Callable[Parameters, ReturnValue]:
        @wraps(function)
        def wrapper(
            *arguments: Parameters.args,
            **keyword_arguments: Parameters.kwargs,
        ) -> ReturnValue:
            start_time: float = perf_counter()

            try:
                return function(*arguments, **keyword_arguments)
            finally:
                end_time: float = perf_counter()
                total_elapsed_time: float = end_time - start_time
                logging.log(
                    log_level,
                    "%s took %.3f seconds.",
                    function.__qualname__,
                    total_elapsed_time,
                )

        return wrapper

    return decorator


def generate_random_petal_colors(
    *, random_seed: int | None = None
) -> Iterator[PetalColor]:
    random_number_generator: random.Random = random.Random(random_seed)

    while True:
        yield random_number_generator.choice(PETAL_COLORS)


def configure_logging(
    *, format_specifier_string: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> None:
    logging.basicConfig(level=logging.INFO, format=format_specifier_string)
    logging.info("Logging service was configured.")


def convert_degrees_to_radians(*, angle: Degrees) -> Radians:
    return Radians(
        angle
        * get_phyllotaxis_constant(constant_name="MULTIPLIER_FROM_DEGREE_TO_RADIANS")
    )


@log_execution_time()
@profile_memory(precision=4)
def draw_phyllotaxis_pattern(
    *,
    actual_turtle: turtle.Turtle,
    number_of_stamps_to_draw: int,
    number_of_petals_to_draw: int,
    spacing_scaling_coefficient: SpacingScalingCoefficient = 5,
    angular_step_between_consecutive_positions_in_degrees: Degrees = get_phyllotaxis_constant(
        constant_name="GOLDEN_ANGLE_IN_DEGREES"
    ),
    petal_color_iterator: Optional[Iterator[PetalColor]] = None,
) -> None:

    logging.info("Starting precondition checks.")

    assert number_of_stamps_to_draw > 0, "We should have at least one stamp."
    assert number_of_petals_to_draw > 0, "We should have at least one petal."
    assert type(number_of_stamps_to_draw) == int, "Stamp count should be an int."
    assert type(number_of_petals_to_draw) == int, "Petal count should be an int."
    if not math.isclose(
        angular_step_between_consecutive_positions_in_degrees,
        get_phyllotaxis_constant(constant_name="GOLDEN_ANGLE_IN_DEGREES"),
    ):
        raise ExceptionGroup(
            "Angular-step validation failed.",
            [
                AngularStepShouldBeAround137Point50776405Degrees(),
                AngularStepShouldBeAroundTheGoldenAngle(),
            ],
        )
    logging.info("Precondition checks were finished.")

    logging.info("Starting actual logic.")

    angular_step_between_consecutive_positions_in_radians: Radians = (
        convert_degrees_to_radians(
            angle=angular_step_between_consecutive_positions_in_degrees
        )
    )

    NUMBER_OF_TOTAL_SHAPES_TO_DRAW: Final[int] = (
        default_compute_total_number_of_shapes_to_be_drawn(
            number_of_stamps_to_draw=number_of_stamps_to_draw,
            number_of_petals_to_draw=number_of_petals_to_draw,
        )
    )

    logging.info("Starting to draw Phyllotaxis.")

    logging.info(f"{NUMBER_OF_TOTAL_SHAPES_TO_DRAW=}")
    logging.info(f"{number_of_stamps_to_draw=}")
    logging.info(f"{number_of_petals_to_draw=}")

    logging.info(f"Starting drawing stamps.")
    for current_iteration in range(NUMBER_OF_TOTAL_SHAPES_TO_DRAW):
        logging.debug(f"Currently at iteration: {current_iteration}.")
        distance_from_center: float = (
            spacing_scaling_coefficient * current_iteration**0.5
        )
        logging.debug(f"{distance_from_center=}")

        position_angle_in_radians: Radians = Radians(
            current_iteration * angular_step_between_consecutive_positions_in_radians
        )

        logging.debug(f"{position_angle_in_radians=}")

        coordinates: PointInCartesianCoordinates = (
            PointInCartesianCoordinates.from_polar_coordinates(
                distance_from_center=distance_from_center,
                position_angle_in_radians=position_angle_in_radians,
            )
        )

        first_coordinate, second_coordinate = coordinates

        logging.debug(f"{first_coordinate=}")
        logging.debug(f"{second_coordinate=}")

        position_turtle_before_drawing(
            actual_turtle=actual_turtle,
            first_coordinate=first_coordinate,
            second_coordinate=second_coordinate,
        )

        orient_turtle_before_drawing(
            actual_turtle=actual_turtle,
            current_iteration=current_iteration,
            angular_step_between_consecutive_positions_in_degrees=angular_step_between_consecutive_positions_in_degrees,
        )

        if current_iteration == number_of_stamps_to_draw:
            logging.info("Finished drawing stamps.")
            logging.info("Starting drawing petals.")
        if current_iteration >= number_of_stamps_to_draw:
            try:
                petal_color: PetalColor = next(petal_color_iterator)
            except StopIteration as exception:
                raise PetalColorIteratorExhausted(
                    f"No color was available for petal "
                    f"{current_iteration - number_of_stamps_to_draw + 1} "
                    f"of {number_of_petals_to_draw}."
                ) from exception
            draw_one_individual_petal(
                actual_turtle=actual_turtle,
                petal_color=petal_color,
            )
            continue
        draw_one_individual_stamp(actual_turtle=actual_turtle)
    logging.info("Finished drawing petals.")
    logging.info("Finished drawing Phyllotaxis.")
    logging.info("Actual logic was finished.")


def compute_total_number_of_shapes_to_be_drawn(
    *,
    number_of_stamps_to_draw: int,
    number_of_petals_to_draw: int,
    addition_function: Callable[[int, int], int],
) -> int:
    return addition_function(number_of_stamps_to_draw, number_of_petals_to_draw)


default_compute_total_number_of_shapes_to_be_drawn: partial[int] = partial(
    compute_total_number_of_shapes_to_be_drawn, addition_function=lambda x, y: x + y
)


def position_turtle_before_drawing(
    *, actual_turtle: turtle.Turtle, first_coordinate: float, second_coordinate: float
) -> None:
    logging.debug("Positioning turtle before drawing.")
    actual_turtle.up()
    actual_turtle.setpos(first_coordinate, second_coordinate)
    actual_turtle.down()
    logging.debug("Finished positioning the turtle.")


def orient_turtle_before_drawing(
    *,
    actual_turtle: turtle.Turtle,
    current_iteration: int,
    angular_step_between_consecutive_positions_in_degrees: Degrees,
) -> None:
    logging.debug("Orienting turtle before drawing.")
    turtle_orientation: Degrees = Degrees(
        current_iteration * angular_step_between_consecutive_positions_in_degrees
    )
    actual_turtle.setheading(turtle_orientation)
    logging.debug("Finished orienting turtle.")


def draw_one_individual_petal(
    *,
    actual_turtle: turtle.Turtle,
    right_movement: int = 20,
    left_movement: int = 40,
    left_movement_2: int = 140,
    forward_movement: int = 70,
    petal_color: PetalColor,
) -> None:
    logging.debug("Starting to draw individual petal.")
    actual_turtle.fillcolor(petal_color.value)

    with fill_shape(actual_turtle=actual_turtle):
        actual_turtle.right(right_movement)
        actual_turtle.forward(forward_movement)
        actual_turtle.left(left_movement)
        actual_turtle.forward(forward_movement)
        actual_turtle.left(left_movement_2)
        actual_turtle.forward(forward_movement)
        actual_turtle.left(left_movement)
        actual_turtle.forward(forward_movement)
        actual_turtle.penup()
    logging.debug("Finished drawing individual petal.")


def draw_one_individual_stamp(*, actual_turtle: turtle.Turtle) -> None:
    logging.debug("Starting to draw individual stamp.")
    actual_turtle.fillcolor("orange")
    actual_turtle.stamp()
    logging.debug("Finished drawing individual stamp.")


@log_execution_time()
@profile_memory(precision=4)
def setup_turtle(
    *,
    actual_turtle: turtle.Turtle,
    speed: TurtleSpeed = 0,
    shape: TurtleShape = "circle",
) -> None:
    logging.info("Setting up turtle.")
    logging.info("Setting up turtle speed.")
    logging.info(f"{speed=}")
    actual_turtle.speed(speed)
    logging.info("Setting up turtle shape.")
    logging.info(f"{shape=}")
    actual_turtle.shape(shape)
    logging.info("Turtle setup was finished.")


setup_fast_circle_turtle: partial[None] = partial(
    setup_turtle,
    speed=0,
    shape="circle",
)


@log_execution_time()
def hide_turtle(*, actual_turtle: turtle.Turtle) -> None:
    actual_turtle.hideturtle()


def ensure_that_window_does_not_close_after_animation_finishes() -> None:
    logging.warning("Ensuring that window does not close after animation finishes.")
    turtle.done()


def main(*args, **kwargs) -> None:
    """
    Drawing a Phyllotaxis pattern via turtle graphics.
    Idea: https://www.geeksforgeeks.org/python/algorithmic-botany-phyllotaxis-python/
    """
    configure_logging()
    current_turtle = turtle.Turtle()
    setup_fast_circle_turtle(actual_turtle=current_turtle)
    random_seed_for_petal_color_iterator: int = int(str(time())[5:9])
    logging.info(f"{random_seed_for_petal_color_iterator=}")
    try:
        draw_phyllotaxis_pattern(
            actual_turtle=current_turtle,
            number_of_stamps_to_draw=120,
            number_of_petals_to_draw=50,
            angular_step_between_consecutive_positions_in_degrees=get_phyllotaxis_constant(
                constant_name="GOLDEN_ANGLE_IN_DEGREES"
            ),
            petal_color_iterator=generate_random_petal_colors(random_seed=10),
        )
    except* PhyllotaxisExceptions as exception_group:
        for exception in exception_group.exceptions:
            logging.error(exception)
        logging.warning("Ending script.")
        sys.exit(1)
    hide_turtle(actual_turtle=current_turtle)
    ensure_that_window_does_not_close_after_animation_finishes()


if __name__ == "__main__":
    main()
