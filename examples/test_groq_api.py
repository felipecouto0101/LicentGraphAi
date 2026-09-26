"""
Teste simples do Nó 3 com API Groq real

Verifica se a chave da API está funcionando corretamente
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

def test_groq_api_connection():
    """Testa conexão com API Groq"""
    print("=" * 60)
    print("Teste de Conexão com API Groq")
    print("=" * 60)
    
    api_key = os.getenv("GROQ_API_KEY")
    mock_mode = os.getenv("NODE3_MOCK_MODE", "true").lower() == "true"
    
    print(f"\nConfiguração:")
    print(f"  API Key: {'***CONFIGURADA***' if api_key else 'NÃO CONFIGURADA'}")
    print(f"  Mock Mode: {mock_mode}")
    
    if not api_key and not mock_mode:
        print("\n[ERRO] API key não configurada e mock_mode=false")
        print("Configure GROQ_API_KEY no .env ou use mock_mode=true")
        return False
    
    try:
        # Inicializa o Nó 3
        node = Node3RequirementAnalyzer(
            api_key=api_key,
            mock_mode=mock_mode
        )
        
        print(f"\n[OK] Nó 3 inicializado com sucesso")
        print(f"  Modelo: {node.model_name}")
        print(f"  Temperatura: {node.temperature}")
        
        # Testa análise simples
        chunk = {
            "content": "Processador Intel Core i5 ou superior, 8GB de RAM mínimo",
            "section": "exigencias_tecnicas"
        }
        
        print(f"\nTestando análise de chunk...")
        result = node.analyze_chunk(chunk)
        
        print(f"\n[OK] Análise realizada com sucesso")
        print(f"  Requisitos extraídos: {len(result['requirements'])}")
        print(f"  Análise: {result['analysis'][:150]}...")
        
        if mock_mode:
            print(f"\n[INFO] Testado em modo mock (sem LLM real)")
        else:
            print(f"\n[INFO] Testado com LLM real (Groq API)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERRO] Falha ao testar API Groq: {e}")
        return False

if __name__ == "__main__":
    success = test_groq_api_connection()
    
    if success:
        print("\n" + "=" * 60)
        print("[OK] Teste concluído com sucesso!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[ERRO] Teste falhou - verifique configuração")
        print("=" * 60)
