"""``_count_embedded_rows`` must not let one unreadable heap page abort startup.

``ensure_vector_extension`` and the embedding-dimension check run on every start
and used ``SELECT COUNT(*) … WHERE embedding IS NOT NULL`` only to learn whether
a table holds data. That scans every page, so a single page failing its checksum
(``psycopg2.errors.DataCorrupted: invalid page in block N``) raised out of the
lifespan and the API exited — on every restart, until the page was repaired.

These tests use a stub connection so they need no database and can inject the
exact failure without damaging a real page.
"""

from contextlib import contextmanager

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from hindsight_api.migrations import _count_embedded_rows


class _DataCorrupted(Exception):
    """Stands in for psycopg2.errors.DataCorrupted."""


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _StubConn:
    def __init__(self, count, estimate=0):
        self.count = count  # int, or an exception to raise from COUNT(*)
        self.estimate = estimate
        self.savepoints = 0

    @contextmanager
    def begin_nested(self):
        self.savepoints += 1
        yield

    def execute(self, statement, params=None):
        sql = str(statement)
        if "COUNT(*)" in sql:
            if isinstance(self.count, Exception):
                raise self.count
            return _Result(self.count)
        assert "reltuples" in sql
        return _Result(self.estimate)


def _db_error() -> DBAPIError:
    return DBAPIError("SELECT COUNT(*)", None, _DataCorrupted('invalid page in block 38543 of relation "base/1/2"'))


def test_returns_exact_count_when_readable():
    conn = _StubConn(count=1234)
    assert _count_embedded_rows(conn, "public", "memory_units") == 1234
    assert conn.savepoints == 1


def test_falls_back_to_planner_estimate_on_damaged_page(caplog):
    conn = _StubConn(count=_db_error(), estimate=143279)
    with caplog.at_level("ERROR", logger="hindsight_api.migrations"):
        assert _count_embedded_rows(conn, "public", "memory_units") == 143279
    assert "public.memory_units" in caplog.text
    assert "_DataCorrupted" in caplog.text


@pytest.mark.parametrize("estimate", [0, -1, None])
def test_unknown_estimate_is_treated_as_has_data(estimate):
    """reltuples is -1 on a never-analyzed table. A damaged table must never read
    as empty, or the index/dimension reconcile would treat it as safe to rebuild."""
    conn = _StubConn(count=_db_error(), estimate=estimate)
    assert _count_embedded_rows(conn, "public", "memory_units") == 1


def test_other_database_errors_also_degrade():
    conn = _StubConn(count=OperationalError("SELECT COUNT(*)", None, Exception("server closed")), estimate=7)
    assert _count_embedded_rows(conn, "public", "learnings") == 7


def test_non_database_errors_still_propagate():
    conn = _StubConn(count=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        _count_embedded_rows(conn, "public", "memory_units")
