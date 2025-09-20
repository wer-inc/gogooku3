import polars as pl
import pytest

from gogooku3.features_ext.audit import run_basic_audits, AuditError


def test_audit_stmt_nonnegative_passes():
    df = pl.DataFrame({"stmt_days_since_statement": [0, 1, 5]})
    notes = run_basic_audits(df)
    assert isinstance(notes, list)


def test_audit_stmt_nonnegative_fails_on_negative():
    df = pl.DataFrame({"stmt_days_since_statement": [0, -1]})
    with pytest.raises(AuditError):
        run_basic_audits(df)

