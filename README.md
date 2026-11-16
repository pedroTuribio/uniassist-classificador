# Classificador de Mensagens da UniAssist

Compara classificação por **regras** (palavras-chave) e por **Machine Learning**
(TF-IDF + LinearSVC) para encaminhar mensagens aos setores Financeiro, Secretaria,
Suporte TI e Biblioteca.

## Como executar
```bash
pip install -r requirements.txt
python uniassist_pln.py
```

## Arquivos
- `uniassist_pln.py` - código completo e executável
- `parecer.txt` - parecer técnico com métricas e análise das mensagens ambíguas
- `requirements.txt` - dependências
