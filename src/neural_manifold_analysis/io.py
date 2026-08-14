"""Data I/O: generic artifact storage and inspection utilities.

Provides a simple, extensible system for saving and loading analysis artifacts
(numpy arrays, sklearn objects, dicts, etc.) as individual pickle files organised
in a hierarchical folder structure, plus a helper to inspect nested dicts.

    ArtifactStore    — generic key-value pickle store with in-memory caching.
    print_dict_keys  — pretty-print the structure of a (possibly nested) dict.

Project-specific stores belong in the project that uses them: subclass
``ArtifactStore`` there and bind it to that project's directories and
artifact-name conventions.
"""

import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List


__all__ = [
    'print_dict_keys',
    'ArtifactStore',
]


# ==============================================================================
# Utility Functions
# ==============================================================================

def print_dict_keys(d: Any, name: str | None = None, indent: int = 0) -> None:
    """Pretty-print the structure of a (possibly nested) dictionary.

    For each key, prints the type and shape/length of the value.
    Recurses into nested dicts with increasing indentation.

    Args:
        d: A Python dictionary (potentially nested), or any object.
        name: Optional display name for the top-level dict.
        indent: Current indentation level (used internally for recursion).
    """
    # Handle non-dict inputs
    if not isinstance(d, dict):
        if name is not None and indent == 0:
            print(f"Structure for: {name}")
        shape = getattr(d, 'shape', None)
        if shape is not None:
            print(f"{type(d).__name__} (shape: {shape})")
        elif isinstance(d, (list, tuple, set)):
            print(f"{type(d).__name__} (length: {len(d)})")
        else:
            print(f"{type(d).__name__}")
        return

    # Top-level header
    if name is not None and indent == 0:
        print(f"Structure for: {name}")

    for key, value in d.items():
        prefix = ' ' * indent
        print(f"{prefix}{key}", end='')

        if isinstance(value, dict):
            print()
            print_dict_keys(value, name=None, indent=indent + 4)
        elif isinstance(value, (list, tuple, set)):
            print(f"  ({type(value).__name__}, length: {len(value)})")
        else:
            shape = getattr(value, 'shape', None)
            if shape is not None:
                print(f"  (shape: {shape})")
            else:
                print(f"  ({type(value).__name__})")


# ==============================================================================
# ArtifactStore — Base Class
# ==============================================================================

class ArtifactStore:
    """Generic key-value store backed by individual pickle files.

    Each artifact is saved as a separate .pkl file under:
        <base_dir>/<store_id>/<artifact_name>.pkl

    Artifact names can contain '/' to create subdirectories, e.g.:
        store.save('group/item', data)
    saves to:
        <base_dir>/<store_id>/group/item.pkl

    An in-memory cache avoids redundant disk reads within a session.

    Attributes:
        store_id: Identifier string for this store.
        base_dir: Root directory under which per-store folders live.
    """

    def __init__(self, store_id: str, base_dir: Path):
        """Initialize the artifact store.

        Args:
            store_id: Identifier for this store.
            base_dir: Root directory where store folders are created.
        """
        self.store_id = store_id
        self.base_dir = base_dir
        self._cache: Dict[str, Any] = {}

    def _artifact_path(self, name: str) -> Path:
        """Resolve an artifact name to its full filesystem path.

        Args:
            name: Artifact name, e.g. 'data_raw' or 'group/item'.

        Returns:
            Full path including .pkl extension.
        """
        return self.base_dir / self.store_id / f"{name}.pkl"

    def save(self, name: str, data: Any) -> None:
        """Save an artifact to disk and update the cache.

        Creates parent directories automatically if they don't exist.

        Args:
            name: Descriptive artifact name.
            data: Any pickle-serializable Python object.
        """
        path = self._artifact_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self._cache[name] = data

    def load(self, name: str) -> Any:
        """Load an artifact from cache or disk.

        Args:
            name: Artifact name to load.

        Returns:
            The deserialized Python object.

        Raises:
            FileNotFoundError: If the artifact does not exist on disk.
        """
        if name in self._cache:
            return self._cache[name]

        path = self._artifact_path(name)
        if not path.exists():
            available = self.list_artifacts()
            raise FileNotFoundError(
                f"Artifact '{name}' not found for store '{self.store_id}'.\n"
                f"Expected path: {path}\n"
                f"Available artifacts: {available}"
            )
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._cache[name] = data
        return data

    def exists(self, name: str) -> bool:
        """Check whether an artifact exists on disk (without loading it).

        Args:
            name: Artifact name to check.

        Returns:
            True if the .pkl file exists.
        """
        return self._artifact_path(name).exists()

    def list_artifacts(self) -> List[str]:
        """List all artifact names available on disk for this store.

        Returns:
            Sorted list of artifact name strings (without .pkl extension).
            Names with subdirectories use '/' separators (e.g., 'group/item').
        """
        store_dir = self.base_dir / self.store_id
        if not store_dir.exists():
            return []
        pkl_files = sorted(store_dir.rglob('*.pkl'))
        names = []
        for f in pkl_files:
            relative = f.relative_to(store_dir)
            name = str(relative.with_suffix(''))  # strip .pkl, keep subdirs
            names.append(name)
        return names

    def clear_cache(self, name: str | None = None) -> None:
        """Clear the in-memory cache to free RAM.

        Args:
            name: If provided, only clear this specific artifact from cache.
                  If None, clear the entire cache.

        Returns:
            None.
        """
        if name is not None:
            self._cache.pop(name, None)
        else:
            self._cache.clear()
        gc.collect()

    def __repr__(self) -> str:
        n_cached = len(self._cache)
        n_disk = len(self.list_artifacts())
        return (
            f"{self.__class__.__name__}('{self.store_id}', "
            f"artifacts_on_disk={n_disk}, cached={n_cached})"
        )
