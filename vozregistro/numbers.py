"""Capa de normalización numérica: números dichos en palabras -> dígitos.

Reglas (documentadas en README §Normalización numérica):
  R1  "por ciento" -> "%" antes de todo (evita que "ciento" se lea como 100).
  R2  Formatos de dígitos del ASR: 6.500 / 6,500 -> 6500 (miles); 39,90 / 39.90 -> 39.90 (decimal);
      "s/ 39.90" conserva el número.
  R3  Una corrida de palabras numéricas se convierte en un solo número mientras la gramática
      lo permita (centenas -> decenas -> "y" -> unidades; "mil"/"millón" multiplican).
      Cuando la siguiente palabra no puede continuar el número, empieza otro:
      "treinta y nueve noventa" -> "39 90".
  R4  "N con M" (M de dos dígitos) -> "N.M":  "catorce con noventa" -> 14.90.
  R5  "N M" adyacentes con N<1000 y M de dos dígitos, salvo que sigan unidades de tiempo -> "N.M":
      "cuatro cincuenta" -> 4.50 (forma peruana de decir precios).
  R6  "N y medio" / "año y medio" / "litro y medio" -> N.5.
  R7  "un/una/uno" -> 1 (también como artículo: se aplica igual a referencia e hipótesis).
Todas las reglas operan sobre texto en minúsculas y sin tildes.
"""
from __future__ import annotations

import re
import unicodedata

UNITS = {"cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
         "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
TEENS = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
         "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
         "veintiun": 21, "veintiuno": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
         "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
         "veintiocho": 28, "veintinueve": 29}
TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
        "ochenta": 80, "noventa": 90}
HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
            "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500,
            "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
            "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900,
            "novecientas": 900}
MULT = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | set(HUNDREDS) | set(MULT)

TIME_UNITS = r"(?:anos?|dias?|semanas?|meses|mes|minutos?|horas?|personas?|%|por)"
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z0-9]+|\s+|.", re.UNICODE)


def strip_accents(text: str) -> str:
    # quita todas las marcas, incluida la de la ñ (año -> ano): se aplica igual a ambos textos
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_digit_formats(text: str) -> str:
    # miles con separador: 6.500 / 6,500 / 1.250.000
    text = re.sub(r"\b\d{1,3}(?:[.,]\d{3})+\b(?![.,]\d)",
                  lambda m: re.sub(r"[.,]", "", m.group(0)), text)
    # decimal con coma -> punto
    text = re.sub(r"\b(\d+),(\d{1,2})\b", r"\1.\2", text)
    return text


class _Acc:
    """Acumulador de una corrida numérica con estado gramatical."""

    def __init__(self):
        self.total = 0
        self.chunk = 0
        self.stage = "start"  # start | hundreds | tens | tens_y | unit | mult

    def can_take(self, w: str) -> bool:
        s = self.stage
        if w == "y":
            return s == "tens"
        if w in MULT:
            return s in ("start", "hundreds", "tens", "unit")
        if w in HUNDREDS:
            return s in ("start", "mult")
        if w in TENS or w in TEENS:
            return s in ("start", "mult", "hundreds")
        if w in UNITS:
            return s in ("start", "mult", "hundreds", "tens_y")
        return False

    def take(self, w: str):
        if w == "y":
            self.stage = "tens_y"
        elif w in MULT:
            m = MULT[w]
            if m == 1000:
                self.total += (self.chunk or 1) * 1000
            else:
                self.total = (self.total + (self.chunk or 1)) * m
            self.chunk = 0
            self.stage = "mult"
        elif w in HUNDREDS:
            self.chunk += HUNDREDS[w]
            self.stage = "hundreds"
        elif w in TENS:
            self.chunk += TENS[w]
            self.stage = "tens"
        elif w in TEENS:
            self.chunk += TEENS[w]
            self.stage = "unit"
        elif w in UNITS:
            self.chunk += UNITS[w]
            self.stage = "unit"

    def value(self) -> int:
        return self.total + self.chunk


def words_to_digits(text: str) -> str:
    """Convierte corridas de palabras numéricas en dígitos (R3, R7). Espera texto sin tildes."""
    tokens = _TOKEN.findall(text)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in NUMBER_WORDS and tok != "y":
            acc = _Acc()
            j = i
            last_good = i
            while j < n:
                t = tokens[j]
                if t.isspace():
                    j += 1
                    continue
                if t in NUMBER_WORDS or t == "y":
                    if not acc.can_take(t):
                        break
                    # "y" solo es válido si lo sigue una unidad
                    if t == "y":
                        k = j + 1
                        while k < n and tokens[k].isspace():
                            k += 1
                        if k >= n or tokens[k] not in UNITS:
                            break
                    acc.take(t)
                    last_good = j
                    j += 1
                    continue
                break
            out.append(str(acc.value()))
            i = last_good + 1
            continue
        out.append(tok)
        i += 1
    return "".join(out)


def _join_prices(text: str) -> str:
    text = re.sub(r"\b(\d+) con (\d{2})\b", r"\1.\2", text)                                   # R4
    text = re.sub(rf"\b(\d{{1,3}}) (\d{{2}})\b(?!\.\d)(?! ?{TIME_UNITS}\b)", r"\1.\2", text)  # R5
    return text


def _half_rules(text: str) -> str:
    text = re.sub(r"\b(?:un |1 )?(ano|litro|hora|mes) y medio\b", lambda m: f"1.5 {m.group(1)}s", text)
    text = re.sub(rf"\b(\d+) ({TIME_UNITS}|litros?) y medio\b", lambda m: f"{m.group(1)}.5 {m.group(2)}", text)
    text = re.sub(r"\b(\d+) y medio\b", r"\1.5", text)
    return text


def normalize_numbers(text: str) -> str:
    """Pipeline numérico completo (R1-R7) sobre texto en minúsculas y sin tildes."""
    text = re.sub(r"\bpor ciento\b", "%", text)          # R1
    text = _normalize_digit_formats(text)                # R2
    text = words_to_digits(text)                         # R3, R7
    text = _join_prices(text)                            # R4, R5
    text = _half_rules(text)                             # R6
    return text


def parse_number(text: str) -> float | None:
    """Lee el primer número de un fragmento (palabras o dígitos)."""
    if text is None:
        return None
    norm = normalize_numbers(strip_accents(str(text).lower()))
    m = re.search(r"\d+(?:\.\d+)?", norm)
    return float(m.group(0)) if m else None


def fmt_number(value: float) -> str:
    """Forma canónica para buscar un valor en el texto normalizado: 6500, 2.5, 39.90."""
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"
