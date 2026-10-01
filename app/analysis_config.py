"""Analysis configuration without importing models, parsers or workflows."""
import os


def is_mock_mode() -> bool:
    return os.getenv("NODE3_MOCK_MODE", "false").strip().lower() == "true"


def validate_analysis_configuration() -> None:
    if is_mock_mode():
        return
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY não configurada. Defina uma chave válida para análise real "
            "ou ative NODE3_MOCK_MODE=true para demonstração."
        )
