"""OWASP references written on findings and reports.

Poisoned facts and hidden instructions stored as memory map to the Agentic
Top 10 item ASI06. Secrets and personal data map to the LLM Top 10 item
LLM02. Prompt injection, including instructions saved for later retrieval,
maps to LLM01.
"""

from __future__ import annotations

ASI06_ID = "ASI06"
ASI06_TITLE = "Memory & Context Poisoning"
ASI06_REF = "ASI06: Memory & Context Poisoning"
ASI06_URL = "https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications/"

LLM01_ID = "LLM01"
LLM01_TITLE = "Prompt Injection"
LLM01_REF = "LLM01: Prompt Injection"
LLM01_URL = "https://genai.owasp.org/llm-top-10/"

LLM02_ID = "LLM02"
LLM02_TITLE = "Sensitive Information Disclosure"
LLM02_REF = "LLM02: Sensitive Information Disclosure"
LLM02_URL = "https://genai.owasp.org/llm-top-10/"

__all__ = [
    "ASI06_ID",
    "ASI06_REF",
    "ASI06_TITLE",
    "ASI06_URL",
    "LLM01_ID",
    "LLM01_REF",
    "LLM01_TITLE",
    "LLM01_URL",
    "LLM02_ID",
    "LLM02_REF",
    "LLM02_TITLE",
    "LLM02_URL",
]
