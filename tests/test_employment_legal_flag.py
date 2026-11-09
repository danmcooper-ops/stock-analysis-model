"""The Employment Legal Flag reads specific labour terms, not bare words."""
import pytest

from scripts.analyze_stock import _EMPLOYMENT_LEGAL_RE


@pytest.mark.parametrize('text', [
    # Real filings that set the flag on 2026-11-06.
    "o'neill employment agreement ex-10.1: o'neill employment agreement",
    'executive employment agreement, between the company and its ceo',
    'ex-10.16 amendment to r. thornberry employment agreement',
    'amended and restated employment agreement, effective as of january 1',
    'ex-10.1 thompson employment contract',
    'form of zero-strike call confirmation ex-10.1',
    # Bare words the old list would also have caught.
    'loan agreement with a federal credit union',
    'securities class action settlement',
    'supplemental salary continuation plan',
    'price discrimination claims under the robinson-patman act',
    'sewage treatment agreement',
])
def test_not_a_labour_matter(text):
    assert _EMPLOYMENT_LEGAL_RE.search(text) is None


@pytest.mark.parametrize('text', [
    'settlement of wage and hour claims',
    'charge filed with the nlrb',
    'collective bargaining agreement expired',
    'labor union strike at the plant',
    'employee class action alleging harassment',
    'allegations of racial discrimination',
    'wrongful termination suit',
    'notice of layoffs under the warn act',
])
def test_labour_matter(text):
    assert _EMPLOYMENT_LEGAL_RE.search(text) is not None
