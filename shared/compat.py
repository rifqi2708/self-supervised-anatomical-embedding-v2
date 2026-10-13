"""Forward historical entry points to one canonical implementation."""
import importlib
import runpy
import sys

def forward(name, target):
    """Alias imports (including private helpers) or execute a legacy CLI."""
    if name == "__main__":
        runpy.run_module(target, run_name="__main__")
    else:
        implementation = importlib.import_module(target)
        if hasattr(implementation, "__path__"):
            # Keep legacy package discovery on its forwarding modules. Using
            # the canonical __path__ here would load children twice under two
            # names, breaking class identity and monkeypatch compatibility.
            legacy = sys.modules[name]
            protected = {"__name__", "__package__", "__path__", "__spec__",
                         "__loader__", "__file__", "__cached__"}
            legacy.__dict__.update({key: value for key, value in
                                   implementation.__dict__.items()
                                   if key not in protected})
        else:
            sys.modules[name] = implementation
