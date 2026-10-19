import inspect
import json
from pathlib import Path

import py_trees

from arena_humansim.core.behavior import nodes


def _format_annotation(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    if isinstance(annotation, type) and not hasattr(annotation, "__args__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _node_classes() -> list[type]:
    result = []
    for name in nodes.__all__:
        obj = getattr(nodes, name)
        if isinstance(obj, type) and issubclass(obj, py_trees.behaviour.Behaviour):
            result.append(obj)
    result.sort(key=lambda c: c.__name__)
    return result


def document_nodes_descriptions(output_path: str | Path) -> None:
    library: dict[str, dict[str, object]] = {}
    for cls in _node_classes():
        sig = inspect.signature(cls.__init__)
        params: dict[str, dict[str, object]] = {}
        for pname, param in sig.parameters.items():
            if pname in ("self", "name"):
                continue
            params[pname] = {
                "type": _format_annotation(param.annotation),
                "required": param.default is inspect.Parameter.empty,
                "default": None if param.default is inspect.Parameter.empty else repr(param.default),
            }
        library[cls.__name__] = {
            "purpose": (cls.__doc__ or "").strip() or "No description",
            "parameters": params,
        }
    out = Path(output_path)
    out.write_text(json.dumps(library, indent=2) + "\n")
    print(f"Saved behavior nodes description to {out.resolve()}")


if __name__ == "__main__":
    default_out = Path(__file__).parent / "behavior_nodes_library.json"
    document_nodes_descriptions(default_out)
