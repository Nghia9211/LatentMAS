"""Standard-library-only gates for delayed GPU work; never imports torch/CUDA."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

REST_SECONDS = 3 * 60 * 60


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def process_identity(pid):
    try:
        # field22=starttime; comm can contain spaces or parentheses.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(identity):
    current = process_identity(identity['pid'])
    return (current is not None and current['start_ticks'] == identity['start_ticks']
            and current['state'] not in ('Z', 'X'))


def profile_ready(profile):
    profile = Path(profile)
    files = [profile / n for n in ('manifest.json', 'finished.json', 'verification.json')]
    if not all(p.is_file() for p in files):
        return None
    manifest, finish, verification = [json.loads(p.read_text()) for p in files]
    if (finish.get('status') != 'complete' or finish.get('verified') is not True
            or verification.get('passed') is not True or verification.get('missing') != []
            or finish.get('train') != 256 or finish.get('validation') != 128
            or finish.get('main_B_completions') != 3072
            or finish.get('training_started') is not False
            or manifest.get('train_samples') != 256 or manifest.get('validation_samples') != 128):
        raise ValueError('Profile completion/verification contract is not satisfied')
    return {p.name: sha256(p) for p in files}


class RestGate:
    def __init__(self):
        self.started_monotonic = None
        self.started_unix = None

    def step(self, ready, busy, monotonic, unix):
        if not ready or busy:
            self.started_monotonic = self.started_unix = None
            return 'waiting_for_profile', None
        if self.started_monotonic is None:
            self.started_monotonic, self.started_unix = monotonic, unix
        # Both clocks must agree that a full three hours have passed. A backwards
        # wall-clock jump delays release; a forwards jump cannot shorten the rest.
        elapsed = min(monotonic - self.started_monotonic, unix - self.started_unix)
        return ('released' if elapsed >= REST_SECONDS else 'resting'), max(0., REST_SECONDS - elapsed)


def atomic_state(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def check_release(path, profile, output, now=None):
    release = json.loads(Path(path).read_text())
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    if (release.get('rest_seconds_required') != REST_SECONDS
            or release.get('elapsed_monotonic_seconds', 0) < REST_SECONDS
            or release.get('elapsed_wall_seconds', 0) < REST_SECONDS
            or now < release['not_before_unix']
            or Path(release['profile']).resolve() != Path(profile).resolve()
            or Path(release['training_output']).resolve() != Path(output).resolve()
            or release.get('iterations') != 80):
        raise ValueError('Missing/invalid three-hour release for this exact training job')
    plan = Path(release['plan_file'])
    if sha256(plan) != release['plan_sha256']:
        raise ValueError('Scheduled plan changed after approval/release')
    repo = Path(__file__).resolve().parents[1]
    sources = release['source_sha256']
    if 'scripts/train_controller_from_profile.py' not in sources:
        raise ValueError('Release does not lock the training implementation')
    for name, digest in sources.items():
        if sha256(repo / name) != digest:
            raise ValueError('Released source changed: ' + name)
    hashes = profile_ready(profile)
    if hashes is None or hashes != release['profile_sha256']:
        raise ValueError('Profile changed or is not verified complete')
    if any(identity_alive(i) for i in release['profiling_processes']):
        raise ValueError('Profiling processes are still alive')
    return release
