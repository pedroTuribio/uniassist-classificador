"""
Atividade Prática - Classificador de Mensagens da UniAssist

Compara duas soluções para encaminhar mensagens de estudantes ao setor correto:
  Solução A - Regras (palavras-chave pré-definidas)
  Solução B - Machine Learning (TfidfVectorizer + LinearSVC em um Pipeline)

Execução:  python uniassist_pln.py
Dependências: pip install -r requirements.txt
"""

import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
REVISAO = "REVISAO_HUMANA"  # sem correspondência ou empate nas regras

# --------------------------------------------------------------------------
# 1. Base de mensagens classificadas (exemplos rotulados)
# --------------------------------------------------------------------------
DADOS = {
    "Financeiro": [
        "Preciso da segunda via do boleto da mensalidade de novembro",
        "Quando vence a parcela da minha mensalidade?",
        "Paguei o boleto mas o pagamento ainda não foi confirmado",
        "Quero negociar uma dívida de mensalidades atrasadas",
        "Como faço para pedir desconto na mensalidade?",
        "O valor cobrado na minha fatura veio diferente do contrato",
        "Preciso do comprovante de pagamento para o imposto de renda",
        "Fui cobrado duas vezes no cartão de crédito",
        "Quero parcelar o valor da matrícula em mais vezes",
        "Como funciona o financiamento estudantil da faculdade?",
        "Meu boleto venceu, como emitir um novo com juros atualizados?",
        "Não recebi o reembolso da taxa que paguei por engano",
        "Qual o valor da taxa de renovação de matrícula?",
        "Gostaria de saber se minha bolsa de estudos foi aplicada na cobrança",
        "Preciso de um recibo das mensalidades pagas neste ano",
        "O débito automático não foi realizado este mês",
        "Tenho direito a desconto por pagar a anuidade à vista?",
        "Quero cancelar a cobrança recorrente no meu cartão",
        "Como faço para atualizar o vencimento das minhas parcelas?",
        "A cobrança da segunda parcela está incorreta",
    ],
    "Secretaria": [
        "Preciso de uma declaração de matrícula para o estágio",
        "Como solicito meu histórico escolar?",
        "Quero trancar minha matrícula no próximo semestre",
        "Gostaria de saber o prazo para o aproveitamento de disciplinas",
        "Minha nota não foi lançada no sistema acadêmico",
        "Preciso corrigir meu nome no diploma",
        "Como faço a transferência de turno do noturno para o diurno?",
        "Qual a data da colação de grau da minha turma?",
        "Quero solicitar a revisão da nota da prova final",
        "Preciso do ementário das disciplinas que cursei",
        "Como peço o certificado de conclusão do curso?",
        "Gostaria de me inscrever em uma disciplina optativa",
        "Qual é o calendário acadêmico do próximo semestre?",
        "Quero informações sobre a transferência de outra instituição",
        "Preciso justificar minhas faltas com atestado médico",
        "Como faço a reabertura de matrícula após o trancamento?",
        "Quero cancelar minha matrícula no curso",
        "Meu histórico mostra uma disciplina que eu não cursei",
        "Quando saem os resultados do processo de dispensa de disciplina?",
        "Preciso de uma declaração de conclusão de curso",
    ],
    "Suporte TI": [
        "Esqueci minha senha do portal do aluno",
        "Não consigo fazer login no ambiente virtual de aprendizagem",
        "O Wi-Fi do campus não está conectando no meu notebook",
        "Não recebo o e-mail institucional",
        "O sistema fica travado quando tento enviar a atividade",
        "Como redefinir a senha do e-mail da faculdade?",
        "Meu acesso ao laboratório de informática foi bloqueado",
        "O aplicativo da faculdade fecha sozinho no celular",
        "Preciso instalar o Office com a licença institucional",
        "A plataforma não carrega o vídeo da aula gravada",
        "Meu usuário foi bloqueado após várias tentativas de acesso",
        "Erro ao abrir o portal, aparece uma página em branco",
        "Como configurar a VPN para acessar os sistemas de casa?",
        "Não consigo baixar o material da disciplina, dá erro no download",
        "O token de autenticação não chega no meu celular",
        "Preciso de ajuda para configurar a rede sem fio no meu tablet",
        "A página de login do sistema está fora do ar",
        "Não consigo anexar arquivos no ambiente virtual",
        "Meu computador no laboratório não liga",
        "Perdi o acesso à minha conta institucional",
    ],
    "Biblioteca": [
        "Quero renovar o empréstimo do livro de cálculo",
        "Qual o horário de funcionamento da biblioteca?",
        "Devolvi o livro mas continua constando no meu nome",
        "Como faço para reservar uma sala de estudo em grupo?",
        "A biblioteca possui o livro de estruturas de dados?",
        "Perdi um exemplar emprestado, o que devo fazer?",
        "Como acesso o acervo digital de periódicos?",
        "Preciso da ficha catalográfica para o meu TCC",
        "Posso emprestar livros durante as férias?",
        "Quantos livros posso retirar por vez?",
        "Quero sugerir a compra de um livro para o acervo",
        "O livro que reservei já está disponível para retirada?",
        "Como funciona o serviço de comutação bibliográfica?",
        "Existe multa por atraso na devolução de obras?",
        "Preciso de ajuda para formatar as referências do trabalho nas normas ABNT",
        "Gostaria de consultar teses e dissertações do acervo",
        "Posso usar o cubículo de estudo individual hoje?",
        "O livro emprestado está danificado, como proceder?",
        "Quero a lista de livros da bibliografia básica do curso",
        "Como acessar as bases de dados científicas pela biblioteca?",
    ],
}

# --------------------------------------------------------------------------
# 2. Mensagens inéditas (ambíguas / desafiadoras) - sem rótulo único
#    "esperado" lista os setores plausíveis (usado apenas na análise).
# --------------------------------------------------------------------------
INEDITAS = [
    ("Não posso acessar o boleto porque esqueci minha senha.", {"Financeiro", "Suporte TI"}),
    ("O portal não abre e preciso da declaração de matrícula ainda hoje.", {"Suporte TI", "Secretaria"}),
    ("Fui cobrado multa da biblioteca e quero contestar o valor.", {"Biblioteca", "Financeiro"}),
    ("Quero trancar a matrícula, mas tenho mensalidades em aberto.", {"Secretaria", "Financeiro"}),
    ("Não consigo entrar no sistema para reservar o livro da biblioteca.", {"Suporte TI", "Biblioteca"}),
    ("Meu histórico escolar não aparece no portal do aluno.", {"Secretaria", "Suporte TI"}),
    ("Paguei a taxa do diploma mas o sistema não mostra o pagamento.", {"Financeiro", "Secretaria", "Suporte TI"}),
    ("Bom dia, tudo bem? Gostaria de uma informação.", set()),  # sem conteúdo útil
]

# --------------------------------------------------------------------------
# 3. Solução A - classificador baseado em regras
# --------------------------------------------------------------------------
REGRAS = {
    "Financeiro": ["boleto", "mensalidade", "pagamento", "pagar", "paguei", "parcela",
                   "cobranca", "cobrado", "valor", "desconto", "fatura", "reembolso",
                   "taxa", "divida", "bolsa", "recibo", "debito", "cartao"],
    "Secretaria": ["matricula", "historico", "declaracao", "diploma", "nota", "trancar",
                   "trancamento", "disciplina", "certificado", "colacao", "calendario",
                   "transferencia", "faltas", "ementario", "curso"],
    "Suporte TI": ["senha", "login", "acesso", "wifi", "wi-fi", "sistema", "portal",
                   "email", "e-mail", "aplicativo", "erro", "plataforma", "bloqueado",
                   "vpn", "token", "computador", "rede", "instalar", "ambiente virtual"],
    "Biblioteca": ["livro", "biblioteca", "emprestimo", "emprestar", "emprestado",
                   "acervo", "reservar", "reserva", "devolvi", "devolucao", "exemplar",
                   "periodicos", "tcc", "abnt", "bibliografia", "teses"],
}


def normalizar(texto: str) -> str:
    """minúsculas e sem acentos."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def pontuar_regras(texto: str) -> dict:
    t = normalizar(texto)
    return {
        setor: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw), t))
        for setor, kws in REGRAS.items()
    }


def classificar_regras(texto: str) -> str:
    """Retorna o setor com mais palavras-chave. Sem correspondência (nenhuma
    palavra-chave) ou empate entre setores -> encaminha para revisão humana."""
    pontos = pontuar_regras(texto)
    maximo = max(pontos.values())
    if maximo == 0:
        return REVISAO
    vencedores = [s for s, p in pontos.items() if p == maximo]
    return vencedores[0] if len(vencedores) == 1 else REVISAO


# --------------------------------------------------------------------------
# 4. Solução B - Machine Learning (TF-IDF + LinearSVC)
# --------------------------------------------------------------------------
def criar_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf", LinearSVC(random_state=SEED)),
    ])


def encaminhar_com_politica(modelo: Pipeline, texto: str, margem_min: float = 0.30):
    """Política de tratamento de ambiguidade (proposta no parecer):
    se a diferença entre os dois maiores scores do modelo for pequena,
    ou se as regras acusarem Empate/Indefinido, a mensagem vai para triagem humana."""
    scores = modelo.decision_function([texto])[0]
    ordem = np.argsort(scores)[::-1]
    classes = modelo.classes_
    top1, top2 = classes[ordem[0]], classes[ordem[1]]
    margem = float(scores[ordem[0]] - scores[ordem[1]])
    regra = classificar_regras(texto)
    if margem < margem_min or regra == REVISAO:
        return "TRIAGEM HUMANA", top1, top2, margem
    return top1, top1, top2, margem


# --------------------------------------------------------------------------
# 5. Execução principal
# --------------------------------------------------------------------------
def main():
    textos = [m for msgs in DADOS.values() for m in msgs]
    rotulos = [s for s, msgs in DADOS.items() for _ in msgs]
    setores = sorted(DADOS.keys())

    X_tr, X_te, y_tr, y_te = train_test_split(
        textos, rotulos, test_size=0.25, random_state=SEED, stratify=rotulos
    )
    print(f"Total de mensagens: {len(textos)} | treino: {len(X_tr)} | teste: {len(X_te)}\n")

    # ---- Solução A
    pred_regras = [classificar_regras(t) for t in X_te]
    labels_a = setores + [REVISAO]
    print("=" * 70)
    print("SOLUÇÃO A - REGRAS")
    print("=" * 70)
    print(f"Acurácia: {accuracy_score(y_te, pred_regras):.3f}\n")
    print(classification_report(y_te, pred_regras, labels=setores, zero_division=0))
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print("Ordem:", labels_a)
    print(confusion_matrix(y_te, pred_regras, labels=labels_a), "\n")

    # ---- Solução B
    modelo = criar_pipeline()
    modelo.fit(X_tr, y_tr)
    pred_ml = modelo.predict(X_te)
    print("=" * 70)
    print("SOLUÇÃO B - MACHINE LEARNING (TF-IDF + LinearSVC)")
    print("=" * 70)
    print(f"Acurácia: {accuracy_score(y_te, pred_ml):.3f}\n")
    print(classification_report(y_te, pred_ml, labels=setores, zero_division=0))
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print("Ordem:", setores)
    print(confusion_matrix(y_te, pred_ml, labels=setores), "\n")

    # ---- Erros de cada método no teste
    print("=" * 70)
    print("ERROS NO CONJUNTO DE TESTE")
    print("=" * 70)
    for nome, preds in (("Regras", pred_regras), ("ML", pred_ml)):
        print(f"\n[{nome}]")
        erros = [(t, r, p) for t, r, p in zip(X_te, y_te, preds) if r != p]
        if not erros:
            print("  (nenhum erro)")
        for t, r, p in erros:
            print(f"  - '{t}'\n      real={r} | previsto={p}")

    # ---- Mensagens inéditas / ambíguas
    print("\n" + "=" * 70)
    print("OITO MENSAGENS INÉDITAS (AMBÍGUAS)")
    print("=" * 70)
    for i, (msg, plausiveis) in enumerate(INEDITAS, 1):
        a = classificar_regras(msg)
        b = modelo.predict([msg])[0]
        dec, top1, top2, margem = encaminhar_com_politica(modelo, msg)
        print(f"\n{i}. \"{msg}\"")
        print(f"   Pontos nas regras : {pontuar_regras(msg)}")
        print(f"   Regras            : {a}")
        print(f"   ML                : {b}  (2º lugar: {top2}, margem={margem:.2f})")
        print(f"   Política proposta : {dec}")
        if plausiveis:
            print(f"   Setores plausíveis: {sorted(plausiveis)}")


if __name__ == "__main__":
    main()
