"""Portable atomic artifacts, explicit paths, and byte-bound configuration."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    write_jsonl(path, [value], pretty=True)


def write_jsonl(path, rows, pretty=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            if pretty:
                json.dump(rows[0], stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
            else:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path):
    """Prediction artifacts only. Use data.load_qa for answer-bearing inputs."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def expand(value):
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if '${' in expanded:
            raise ValueError('configuration references an unset environment variable')
        return expanded
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def load_config(path):
    text = Path(path).read_text()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError('configuration must be an object')
    config = expand(config)
    # Relative paths are relative to the invocation directory, as in the examples.
    return config


@contextlib.contextmanager
def run_lock(path):
    """Never train the same run twice. GPU allocation remains with your scheduler."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('this run already has a live process')
        yield handle
