"""Prediction modules, with Python 3.9 support for the prediction method."""

import sys

if sys.version_info < (3, 10):
    import __future__
    import importlib.abc
    import importlib.machinery
    import importlib.util
    from pathlib import Path

    class _DeferredAnnotationsLoader(importlib.machinery.SourceFileLoader):
        """Evaluate Python 3.10-style type hints lazily on Python 3.9."""

        def get_code(self, fullname):
            # Bypass older .pyc files compiled without postponed annotations.
            path = self.get_filename(fullname)
            return self.source_to_code(self.get_data(path), path)

        def source_to_code(self, data, path, *, _optimize=-1):
            return compile(data, path, 'exec',
                           flags=__future__.annotations.compiler_flag,
                           dont_inherit=True, optimize=_optimize)

    class _PredictionMethodFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != __name__ + '.prediction_method':
                return None
            source = str(Path(__file__).with_name('prediction_method.py'))
            loader = _DeferredAnnotationsLoader(fullname, source)
            return importlib.util.spec_from_file_location(fullname, source, loader=loader)

    sys.meta_path.insert(0, _PredictionMethodFinder())
