"""Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.

Purpose: Earlier DICOM/PMCC landmark and registration-processing workflow, classified from data paths and helper callers.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: time.
Role: shared helper.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'master_project.shared'
import time

class FileMonitor():
    def __init__(self, filepath, respond:callable, time_interval=1):
        self.filepath = filepath
        self.last = self.read()
        self.time_interval = time_interval
        self.respond = respond

    def read(self):
        return open(self.filepath).read().strip()

    def start(self):
        print(f'Monitoring: {self.filepath}')
        while True:
            time.sleep(self.time_interval)

            current = self.read()
            if current == self.last:
                continue
            else:
                self.respond()
                self.last = current
