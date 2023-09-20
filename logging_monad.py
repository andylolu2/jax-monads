from __future__ import annotations

from typing import TypeAlias, Any, Callable

import jax
import jax.numpy as jnp

PyTree: TypeAlias = Any

@jax.tree_util.register_pytree_node_class
class LoggedValue:
    def __init__(self, value: PyTree, logs: dict[str, PyTree] |
                 None = None) -> None:
        self.value = value
        self.logs = logs or {}

    def __repr__(self) -> str:
        return f"LoggedValue(value={self.value}, logs={self.logs})"

    def bind(self, f: Callable[[PyTree], LoggedValue]) -> LoggedValue:
        """Apply a function with side effects (logs)"""
        x = f(self.value)
        self.value = x.value
        self.logs.update(x.logs)
        return self

    def map(self, f: Callable[[PyTree], PyTree]) -> LoggedValue:
        """Apply a function with no side effects"""
        self.value = f(self.value)
        return self

    def tree_flatten(self):
        return (self.value, self.logs), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def bind_nary(f: Callable[..., LoggedValue], *xs: LoggedValue) -> LoggedValue:
    """Apply a n-ary function to n LoggedValues"""
    match xs:
        case []:
            return f()
        case [x]:
            return x.bind(f)
        case [x, *xs]:
            return x.bind(lambda x_: bind_nary(lambda *xs_: f(x_, *xs_), *xs))

def transform(f: Callable[..., LoggedValue]) -> Callable[..., LoggedValue]:
    return lambda *xs: bind_nary(f, *xs)

@transform
def add(x: PyTree, y: PyTree) -> LoggedValue:
    return LoggedValue(x + y, {"sum": jnp.sum(x + y)})

@jax.jit
def f():
    x = LoggedValue(jnp.array([1, 2, 3]), {"x": 1})
    y = LoggedValue(jnp.array([4, 5, 6]), {"y": 2})
    z = add(x, y)
    z = z.map(lambda z_: z_ * jnp.arange(z_.size))
    return z


if __name__ == "__main__":
    z = f()
    print(z)


