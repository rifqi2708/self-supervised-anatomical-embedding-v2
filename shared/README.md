# Cross project shared code

Model interfaces, image utilities, training/evaluation launchers and cycle helpers are reused across publication, earlier master and demo workflows. They retain their existing scientific behavior and import the unchanged `sam/` package.

`compat.py` forwards historical Python entry points to one implementation and preserves class/module identity. `repository.py` resolves source members used in implementation signatures across the new layout. Explicit external-checkout lookup also supports frozen legacy layouts; current-checkout signature lookup stays canonical. Neither changes dataset/checkpoint/evidence path semantics. Compatibility paths are forwarding modules, shell launchers or relative links; they are not second implementations.
