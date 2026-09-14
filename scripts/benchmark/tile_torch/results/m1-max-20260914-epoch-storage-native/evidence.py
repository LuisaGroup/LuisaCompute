"""Self-contained logical SHA256-blob reader; stdlib only, no extraction/native."""
import hashlib
import json
from pathlib import Path
import tarfile


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.cache, self.loaded = {}, set()

    def alias(self, original, full9=False):
        name = str(original)
        if name.startswith('/tmp/'):
            name = '/private' + name
        historical = self.manifest['full9_aliases']
        return historical[name] if full9 and name in historical else self.manifest['aliases'][name]

    def blob(self, digest):
        shard = self.manifest['objects'][digest]['shard']
        if shard not in self.loaded:
            with tarfile.open(self.directory / shard) as archive:
                found = set()
                for member in archive:
                    assert member.isfile() and member.name not in found and len(member.name) == 64
                    found.add(member.name)
                    expected = self.manifest['objects'][member.name]
                    data = archive.extractfile(member).read()
                    assert expected['shard'] == shard and len(data) == expected['size']
                    assert hashlib.sha256(data).hexdigest() == member.name
                    self.cache[member.name] = data
            assert found == {key for key, row in self.manifest['objects'].items() if row['shard'] == shard}
            self.loaded.add(shard)
        return self.cache[digest]

    def read(self, name):
        row = self.manifest['files'][name]
        data = self.blob(row['sha256'])
        assert len(data) == row['size']
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def digest(self, name):
        self.read(name)
        return self.manifest['files'][name]['sha256']
