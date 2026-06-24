from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    def map(self, f: Callable[[T], U]) -> "Ok[U]":
        return Ok(f(self.value))

    def flat_map(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        return f(self.value)

    def map_err(self, f: Callable[[E], F]) -> "Ok[T]":
        return self

    def unwrap_or(self, default: T) -> T:
        return self.value

    def is_ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    def map(self, f: Callable[..., object]) -> "Err[E]":
        return self

    def flat_map(self, f: Callable[..., object]) -> "Err[E]":
        return self

    def map_err(self, f: Callable[[E], F]) -> "Err[F]":
        return Err(f(self.error))

    def unwrap_or(self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False


type Result[T, E] = Ok[T] | Err[E]
type Option[T] = Ok[T] | Err[None]


def Some(value: T) -> Ok[T]:
    return Ok(value)


def Nothing() -> Err[None]:
    return Err(None)
