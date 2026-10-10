# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# texto.py — Limpieza de texto compartida por el pipeline y la inferencia.

import re

RUIDO_INSTITUCIONAL = re.compile(r"\b(cdl|fcbcb|revoluciones|jum|juaf)\b")
PARENTESIS_VACIO = re.compile(r"\(\s*\)")
ESPACIOS = re.compile(r"\s+")


def _sin_parentesis_huerfanos(texto: str) -> str:
    """Elimina los paréntesis que no tienen su pareja."""
    caracteres = list(texto)
    abiertos = []
    for i, ch in enumerate(caracteres):
        if ch == "(":
            abiertos.append(i)
        elif ch == ")":
            if abiertos:
                abiertos.pop()
            else:
                caracteres[i] = ""
    for i in abiertos:
        caracteres[i] = ""
    return "".join(caracteres)


def limpiar_texto(texto: str) -> str:
    """Minúsculas, quita ruido institucional, paréntesis vacíos y espacios."""
    texto = texto.lower()
    texto = RUIDO_INSTITUCIONAL.sub("", texto)
    texto = PARENTESIS_VACIO.sub("", texto)
    texto = _sin_parentesis_huerfanos(texto)
    texto = ESPACIOS.sub(" ", texto).strip()
    return texto
