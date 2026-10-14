# Cross project shared code

Model interfaces, image utilities, training/evaluation launchers and cycle helpers are reused across publication, earlier master and contributor workflows. They import the unchanged sam package.

Invoke these canonical modules and scripts directly. Obsolete Python aliases, shell launchers and relocation links have been removed.

`repository.py` is a source-only registry for implementation signatures and explicitly selected frozen external checkouts. Current signatures hash actual canonical members and fail on missing files. External bootstrap dispatch chooses the layout of its selected checkout without exposing legacy packages in the current tree. Dataset, checkpoint and evidence locations retain their original interpretation.
