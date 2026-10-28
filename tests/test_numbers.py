import pytest

from vozregistro.numbers import normalize_numbers, strip_accents


def n(t):
    return normalize_numbers(strip_accents(t.lower()))


@pytest.mark.parametrize("spoken,expected", [
    ("seis mil quinientos soles", "6500 soles"),
    ("nueve mil trescientos cincuenta", "9350"),
    ("treinta y nueve con noventa", "39.90"),
    ("a cuatro cincuenta cada una", "a 4.50 cada 1"),
    ("ciento veintiún soles", "121 soles"),
    ("dos años y medio", "2.5 anos"),
    ("quince por ciento", "15 %"),
    ("tres mil ciento veinticinco", "3125"),
    ("6.500 soles", "6500 soles"),
    ("S/ 39,90", "s/ 39.90"),
    ("cuarenta y cinco minutos", "45 minutos"),
    ("litro y medio", "1.5 litros"),
    ("mil doscientos noventa", "1290"),
])
def test_spoken_numbers(spoken, expected):
    assert n(spoken) == expected
