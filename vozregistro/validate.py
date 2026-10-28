"""Validación aritmética del pedido y diagnóstico de la causa cuando no cuadra.

Chequeo: sum(cantidad * precio_unitario) - descuento(promoción) == total_dicho (± tolerancia).
No necesita verdad de campo: los campos se controlan entre sí.

Regla de diagnóstico (cuando no cuadra):
  error_asr       si hay evidencia de que un número se oyó mal:
                    (a) algún precio unitario no coincide con la carta, o
                    (b) alguna palabra numérica del total dicho tiene confianza ASR < umbral, o
                    (c) recalculando con precios de carta el total dicho SÍ cuadra.
  error_hablante  si todos los números son coherentes con la carta y se oyeron con confianza,
                  y aun así la suma no da: quien habló se equivocó.
  incompleto      si faltan ítems, precios o el total (no se puede chequear).
"""
from __future__ import annotations

from .numbers import parse_number
from .textnorm import normalize_for_match


def discount(promo: str | None, items: list[dict], subtotal: float, cfg: dict) -> float:
    if not promo:
        return 0.0
    spec = cfg["order"]["promotions"].get(promo)
    if not spec:
        return 0.0
    if spec["type"] == "percent":
        return round(subtotal * spec["value"] / 100, 2)
    if spec["type"] == "fixed":
        return float(spec["value"])
    if spec["type"] == "free_item":
        prices = [i["precio_unitario"] for i in items if i["producto"] == spec["product"] and i.get("precio_unitario")]
        return float(min(prices)) if prices else 0.0
    return 0.0


def expected_total(items: list[dict], promo: str | None, cfg: dict, price_key=lambda i: i.get("precio_unitario")):
    subtotal = round(sum(int(i["cantidad"]) * float(price_key(i)) for i in items), 2)
    return subtotal, round(subtotal - discount(promo, items, subtotal, cfg), 2)


def catalog_price(item: dict, cfg: dict):
    spec = cfg["order"]["menu"].get(item["producto"])
    if not spec:
        return None
    size = item.get("tamano")
    for k, v in spec["sizes"].items():
        if size is not None and normalize_for_match(k) == normalize_for_match(str(size)):
            return float(v)
    return float(next(iter(spec["sizes"].values()))) if len(spec["sizes"]) == 1 else None


def _low_conf_numbers(words: list[dict], threshold: float, span: int = 12) -> list[dict]:
    """Palabras numéricas dichas justo después de la ÚLTIMA mención de "total" con confianza ASR baja."""
    words = words or []
    idx = [i for i, w in enumerate(words) if "total" in normalize_for_match(w.get("w", ""))]
    if not idx:
        return []
    window = words[idx[-1]: idx[-1] + span]
    return [w for w in window if parse_number(w.get("w", "")) is not None and w.get("p", 1.0) < threshold]


def validate_order(fields: dict, words: list[dict] | None, cfg: dict) -> dict:
    vcfg = cfg["validation"]
    items = fields.get("items") or []
    total = fields.get("total_dicho")
    if not items or total is None or any(i.get("precio_unitario") is None for i in items):
        return {"cuadra": None, "diagnostico": "incompleto", "esperado": None, "dicho": total,
                "diferencia": None, "senales": ["faltan ítems, precios o total"]}
    subtotal, exp = expected_total(items, fields.get("promocion"), cfg)
    diff = round(float(total) - exp, 2)
    result = {"subtotal": subtotal, "descuento": round(subtotal - exp, 2), "esperado": exp,
              "dicho": float(total), "diferencia": diff, "cuadra": abs(diff) <= vcfg["tolerance"], "senales": []}
    if result["cuadra"]:
        result["diagnostico"] = "ok"
        return result

    senales = []
    for it in items:
        cp = catalog_price(it, cfg)
        if cp is not None and abs(cp - float(it["precio_unitario"])) > vcfg["tolerance"]:
            senales.append(f"precio de {it['producto']} oído {it['precio_unitario']} ≠ carta {cp}")
    low = _low_conf_numbers(words, vcfg["low_confidence"])
    if low:
        senales.append("números con baja confianza ASR: " + ", ".join(f"{w['w']}({w['p']})" for w in low[:5]))
    cat_items = [i for i in items if catalog_price(i, cfg) is not None]
    if len(cat_items) == len(items):
        _, exp_cat = expected_total(items, fields.get("promocion"), cfg, price_key=lambda i: catalog_price(i, cfg))
        result["esperado_con_carta"] = exp_cat
        if abs(exp_cat - float(total)) <= vcfg["tolerance"]:
            senales.append("con precios de carta el total dicho sí cuadra")
    result["senales"] = senales
    result["diagnostico"] = "error_asr" if senales else "error_hablante"
    return result
