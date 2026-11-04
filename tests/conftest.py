"""Gemeinsame Fixtures: die Mini-Netze, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sbi_algorithm as A  # noqa: E402
from sbi_scenario import Netz, grid_netz, hantel_netz  # noqa: E402


@pytest.fixture(scope="session")
def pfad():
    """Zwei Kanten hintereinander: zusammenhängend genau dann, wenn beide halten (Handrechnung bei p = ½: X ~ Bernoulli(¼), S_i = 1/3, T_i = 2/3)."""
    return Netz("Pfad", 3, ((0, 1), (1, 2)), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))


@pytest.fixture(scope="session")
def dreieck():
    return Netz("Dreieck", 3, ((0, 1), (1, 2), (0, 2)), ((0.0, 0.0), (1.0, 0.0), (0.5, 0.8)))


@pytest.fixture(scope="session")
def hantel():
    return hantel_netz()


@pytest.fixture(scope="session")
def tab_pfad(pfad):
    return A.conn_table(pfad.n, pfad.edges)


@pytest.fixture(scope="session")
def tab_dreieck(dreieck):
    return A.conn_table(dreieck.n, dreieck.edges)


@pytest.fixture(scope="session")
def tab_hantel(hantel):
    return A.conn_table(hantel.n, hantel.edges)


@pytest.fixture(scope="session")
def gitter33():
    return grid_netz(3, 3)


@pytest.fixture(scope="session")
def tab_gitter33(gitter33):
    return A.conn_table(gitter33.n, gitter33.edges)
