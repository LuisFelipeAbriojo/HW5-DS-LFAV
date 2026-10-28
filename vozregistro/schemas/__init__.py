"""Registro de esquemas: el mismo motor sirve a la ficha de candidato y al pedido."""
from . import interview, order

SCHEMAS = {interview.NAME: interview, order.NAME: order}


def get_schema(name: str):
    if name not in SCHEMAS:
        raise ValueError(f"Esquema desconocido: {name}. Opciones: {list(SCHEMAS)}")
    return SCHEMAS[name]
