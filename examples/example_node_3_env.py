"""
Exemplo de como usar variáveis de ambiente com Nó 3

Demonstra como configurar o Nó 3 usando .env
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from app.rag.node_3_analyzer import Node3RequirementAnalyzer
import os

# Carrega variáveis de ambiente
load_dotenv()

def example_with_env_variables():
    """Exemplo usando variáveis de ambiente do .env"""
    print("=" * 60)
    print("Nó 3 com Variáveis de Ambiente")
    print("=" * 60)
    
    # Lê configurações do .env
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
    temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
    mock_mode = os.getenv("NODE3_MOCK_MODE", "true").lower() == "true"
    
    print(f"\nConfigurações carregadas:")
    print(f"  API Key: {'***CONFIGURADA***' if api_key else 'NÃO CONFIGURADA'}")
    print(f"  Modelo: {model_name}")
    print(f"  Temperatura: {temperature}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Mock Mode: {mock_mode}")
    
    # Inicializa o Nó 3
    node = Node3RequirementAnalyzer(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        mock_mode=mock_mode
    )
    
    # Testa análise
    chunk = {
        "content": "Processador Intel Core i5 ou superior, 8GB de RAM mínimo",
        "section": "exigencias_tecnicas"
    }
    
    result = node.analyze_chunk(chunk)
    
    print(f"\nResultado da análise:")
    print(f"  Requisitos extraídos: {len(result['requirements'])}")
    print(f"  Análise: {result['analysis'][:100]}...")
    
    if mock_mode:
        print(f"\n[INFO] Rodando em modo mock (sem API key real)")
    else:
        print(f"\n[INFO] Rodando com API Groq real")

def example_manual_configuration():
    """Exemplo com configuração manual (sobrescreve .env)"""
    print("\n" + "=" * 60)
    print("Nó 3 com Configuração Manual")
    print("=" * 60)
    
    # Configuração manual explícita
    node = Node3RequirementAnalyzer(
        api_key="test_key",  # Ou sua API key real
        model_name="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=1000,
        mock_mode=True  # Force mock mode
    )
    
    chunk = {
        "content": "Prazo de entrega: 30 dias corridos",
        "section": "prazos"
    }
    
    result = node.identify_critical_points(chunk)
    
    print(f"\nAnálise crítica:")
    print(f"  Nível de risco: {result['risk_level']}")
    print(f"  Pontos críticos: {result['critical_points']}")
    print(f"  Recomendações: {result['recommendations']}")

if __name__ == "__main__":
    example_with_env_variables()
    example_manual_configuration()
