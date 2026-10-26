"""Registry of NL-to-SPARQL systems.

A system is a callable taking (question, kg, context) and returning a dict
with at least a "sparql" key; `kg` is a KnowledgeGraph. Register a function,
or a class whose constructor takes the system's parameters, with
@register("name"). `create(name, **params)` builds one, which is how
experiment configs pass parameters.

Packages can add systems without touching Hermes by declaring an entry point
in the "hermes.systems" group that points at the system.
"""

from importlib.metadata import entry_points

_SYSTEMS = {}
_plugins_loaded = False


def register(name):
    def decorator(system):
        if name in _SYSTEMS:
            raise ValueError(f"A system named {name!r} is already registered.")
        _SYSTEMS[name] = system
        return system

    return decorator


def create(name, **params):
    """Build the named system. Classes receive params; functions take none."""
    factory = _lookup(name)
    if isinstance(factory, type):
        return factory(**params)
    if params:
        raise TypeError(f"System {name!r} takes no parameters.")
    return factory


def get(name):
    return create(name)


def available():
    _load_plugins()
    return sorted(_SYSTEMS)


def _lookup(name):
    if name not in _SYSTEMS:
        _load_plugins()
    try:
        return _SYSTEMS[name]
    except KeyError:
        known = ", ".join(sorted(_SYSTEMS)) or "(none)"
        raise KeyError(f"Unknown system {name!r}. Available: {known}.") from None


def _load_plugins():
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for entry in entry_points(group="hermes.systems"):
        if entry.name not in _SYSTEMS:
            _SYSTEMS[entry.name] = entry.load()
