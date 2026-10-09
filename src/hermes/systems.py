"""Registry of NL-to-SPARQL systems.

A system is a callable taking (question, ontology, context) and returning a
dict with at least a "sparql" key. Register one with @register("name") and run
it with hermes.ask(..., system="name").
"""

_SYSTEMS = {}


def register(name):
    def decorator(system):
        if name in _SYSTEMS:
            raise ValueError(f"A system named {name!r} is already registered.")
        _SYSTEMS[name] = system
        return system

    return decorator


def get(name):
    try:
        return _SYSTEMS[name]
    except KeyError:
        known = ", ".join(sorted(_SYSTEMS)) or "(none)"
        raise KeyError(f"Unknown system {name!r}. Available: {known}.") from None


def available():
    return sorted(_SYSTEMS)
