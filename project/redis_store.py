import redis
from pickle import loads, dumps, dump, load
from collections.abc import MutableMapping


class RedisStore(MutableMapping):

    def __init__(self, db):
        self._store = redis.Redis(db=db)

    def __getitem__(self, key):
        return loads(self._store[key])

    def __setitem__(self, key, value):
        self._store[key] = dumps(value)

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._store.keys())

    def keys(self):
        return self._store.keys()

    def clear(self):
        self._store.flushdb()
