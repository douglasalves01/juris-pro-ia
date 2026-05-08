
!pip install transformers datasets accelerate scikit-learn -q

# %% [markdown]
# ## 2. Imports

# %%
import re
import json
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, f1_score as sk_f1
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# %% [markdown]
# ## 3. Regras de supervisão fraca
#
# Estas funções aplicam regras jurídicas especializadas para gerar labels
# automáticos em textos de contratos sem anotação manual.
# O modelo aprende a generalizar essas regras para novos documentos.

# %%
# ── CLASSES ─────────────────────────────────────────────────
# 0 = baixo risco
# 1 = médio risco
# 2 = alto risco
# 3 = crítico

RISK_LABELS = {0: "baixo", 1: "médio", 2: "alto", 3: "crítico"}

# ── REGRAS DE RISCO ──────────────────────────────────────────

def detectar_lgpd(texto: str) -> dict | None:
    """Verifica ausência de conformidade com a LGPD."""
    tem_dados_pessoais = bool(re.search(
        r"dados pessoais|tratamento de dados|coleta de dados|dados sens[íi]veis",
        texto, re.I
    ))
    tem_base_legal = bool(re.search(
        r"base legal|consentimento|leg[íi]timo interesse|contrato|obrigação legal|"
        r"DPO|encarregado de dados|Art\. 7|Art\. 11|LGPD",
        texto, re.I
    ))
    if tem_dados_pessoais and not tem_base_legal:
        return {
            "tipo": "lgpd_compliance",
            "severidade": "alta",
            "descricao": "Possível não conformidade com LGPD detectada: "
                         "tratamento de dados pessoais sem base legal explícita.",
        }
    return None

def detectar_multa_abusiva(texto: str) -> dict | None:
    """Detecta cláusulas de multa acima do padrão de mercado."""
    matches = re.findall(
        r"multa[^.]{0,60}?(\d{1,3})\s*%", texto, re.I
    )
    for m in matches:
        pct = int(m)
        if pct > 20:
            return {
                "tipo": "penalty_clause",
                "severidade": "alta" if pct > 30 else "média",
                "descricao": f"Cláusula de multa com {pct}% — acima do padrão "
                             f"de mercado (10-20%). Risco de nulidade parcial.",
            }
    return None

def detectar_rescisao_unilateral(texto: str) -> dict | None:
    """Detecta prazo de rescisão unilateral abaixo do mínimo razoável."""
    matches = re.findall(
        r"rescis[aã]o\s+unilateral[^.]{0,100}?(\d{1,3})\s*dias?",
        texto, re.I
    )
    for m in matches:
        dias = int(m)
        if dias < 30:
            return {
                "tipo": "termination_clause",
                "severidade": "alta" if dias < 15 else "média",
                "descricao": f"Rescisão unilateral com apenas {dias} dias de aviso "
                             f"— abaixo do prazo razoável de 30 dias.",
            }
    return None

def detectar_foro_desfavoravel(texto: str) -> dict | None:
    """Detecta eleição de foro potencialmente desfavorável."""
    if re.search(r"foro.*?exterior|jurisdição.*?estranger|lei estrangeira", texto, re.I):
        return {
            "tipo": "jurisdiction_clause",
            "severidade": "alta",
            "descricao": "Cláusula de eleição de foro estrangeiro ou aplicação "
                         "de lei estrangeira — pode dificultar a defesa.",
        }
    return None

def detectar_ausencia_limite_responsabilidade(texto: str) -> dict | None:
    """Detecta ausência de limitação de responsabilidade."""
    tem_contrato_valor_alto = bool(re.search(
        r"R\$\s*[\d.,]{6,}", texto  # valores acima de 100k
    ))
    tem_limite = bool(re.search(
        r"limita[çc][aã]o de responsabilidade|responsabilidade m[aá]xima|"
        r"teto de responsabilidade|cap de responsabilidade",
        texto, re.I
    ))
    if tem_contrato_valor_alto and not tem_limite:
        return {
            "tipo": "liability_limitation",
            "severidade": "média",
            "descricao": "Ausência de cláusula de limitação de responsabilidade "
                         "em contrato de alto valor.",
        }
    return None

def detectar_propriedade_intelectual(texto: str) -> dict | None:
    """Detecta ambiguidade em cláusulas de propriedade intelectual."""
    tem_pi = bool(re.search(
        r"propriedade intelectual|software|código-fonte|invenção|criação|"
        r"direitos autorais|patente",
        texto, re.I
    ))
    tem_cessao_clara = bool(re.search(
        r"cede|transfere|pertence.*?contratante|titularidade.*?contratante",
        texto, re.I
    ))
    if tem_pi and not tem_cessao_clara:
        return {
            "tipo": "intellectual_property",
            "severidade": "média",
            "descricao": "Titularidade de propriedade intelectual não definida "
                         "claramente — risco de conflito futuro.",
        }
    return None

def detectar_arbitragem(texto: str) -> dict | None:
    """Sinaliza ausência de cláusula arbitral em contratos de alto valor."""
    tem_valor_alto = bool(re.search(r"R\$\s*[\d.,]{7,}", texto))
    tem_arbitragem = bool(re.search(
        r"arbitragem|árbitro|câmara arbitral|CAMARB|CIESP|ICC|CCI",
        texto, re.I
    ))
    if tem_valor_alto and not tem_arbitragem:
        return {
            "tipo": "dispute_resolution",
            "severidade": "baixa",
            "descricao": "Ausência de cláusula arbitral em contrato de alto valor — "
                         "considere incluir para agilizar resolução de conflitos.",
        }
    return None

REGRAS = [
    detectar_lgpd,
    detectar_multa_abusiva,
    detectar_rescisao_unilateral,
    detectar_foro_desfavoravel,
    detectar_ausencia_limite_responsabilidade,
    detectar_propriedade_intelectual,
    detectar_arbitragem,
]

def calcular_risco(texto: str) -> tuple[int, list]:
    """
    Aplica todas as regras e retorna (risk_label, achados).
    risk_label: 0=baixo, 1=médio, 2=alto, 3=crítico
    """
    achados = []
    for regra in REGRAS:
        resultado = regra(texto)
        if resultado:
            achados.append(resultado)

    # Score baseado na severidade dos achados
    score = 0
    for a in achados:
        if a["severidade"] == "crítica":
            score += 40
        elif a["severidade"] == "alta":
            score += 25
        elif a["severidade"] == "média":
            score += 15
        elif a["severidade"] == "baixa":
            score += 5

    score = min(score, 100)

    if score >= 70:
        label = 3  # crítico
    elif score >= 45:
        label = 2  # alto
    elif score >= 20:
        label = 1  # médio
    else:
        label = 0  # baixo

    return label, achados

# %% [markdown]
# ## 4. Dataset seed de contratos

# %%
CONTRATOS_SEED = [
    # ── BAIXO RISCO ──────────────────────────────────────────────────────────────
    (
        "Contrato de prestação de serviços de limpeza com prazo de 12 meses, "
        "reajuste anual pelo IPCA, rescisão com aviso prévio de 30 dias, "
        "multa rescisória de 10% sobre o saldo contratual. Eleição de foro: "
        "comarca de São Paulo. Responsabilidade limitada ao valor do contrato.",
        0
    ),
    (
        "Locação de imóvel comercial por 24 meses, aluguel mensal de R$ 5.000,00, "
        "reajuste pelo IGP-M, fiança de 3 aluguéis, prazo de 60 dias para "
        "desocupação, sem previsão de dados pessoais. Multa por rescisão antecipada: "
        "3 aluguéis proporcionais.",
        0
    ),
    (
        "Contrato de consultoria empresarial com honorários de R$ 8.000,00 mensais, "
        "sigilo das informações, propriedade intelectual dos relatórios cedida "
        "integralmente ao contratante, multa de 15% por inadimplemento.",
        0
    ),

    # ── MÉDIO RISCO ───────────────────────────────────────────────────────────────
    (
        "Contrato de desenvolvimento de software para gestão hospitalar. "
        "Coleta de dados pessoais dos pacientes mediante consentimento. "
        "Prazo de entrega: 6 meses. Multa por atraso: 25% sobre valor total. "
        "Propriedade do código-fonte não definida explicitamente.",
        1
    ),
    (
        "Fornecimento de equipamentos médicos no valor de R$ 1.200.000,00. "
        "Garantia de 12 meses, assistência técnica on-site. "
        "Sem cláusula de limitação de responsabilidade. "
        "Rescisão unilateral com 20 dias de aviso.",
        1
    ),
    (
        "Prestação de serviços de TI com tratamento de dados dos clientes. "
        "Contrato menciona LGPD mas não especifica base legal para cada operação. "
        "Multa por descumprimento: 18% sobre valor mensal.",
        1
    ),

    # ── ALTO RISCO ────────────────────────────────────────────────────────────────
    (
        "Contrato de outsourcing de TI com acesso a dados pessoais sensíveis "
        "de colaboradores sem menção à LGPD, DPO ou base legal. "
        "Multa rescisória de 35% sobre valor total. "
        "Rescisão unilateral permitida com apenas 10 dias de aviso prévio. "
        "Contrato no valor de R$ 5.000.000,00 sem limitação de responsabilidade.",
        2
    ),
    (
        "Acordo de joint venture para operação em mercado digital coletando "
        "dados de usuários sem consentimento expresso conforme LGPD. "
        "Cláusula de não concorrência por 5 anos sem delimitação territorial. "
        "Multa por violação: 40% sobre o faturamento anual. "
        "Propriedade intelectual das criações conjuntas indefinida.",
        2
    ),
    (
        "Contrato de fornecimento de serviços de saúde coletando dados sensíveis "
        "de pacientes. Sem previsão de DPO. Multa por inadimplemento: 30%. "
        "Rescisão unilateral em 7 dias. Foro eleito em Miami, EUA.",
        2
    ),

    # ── CRÍTICO ────────────────────────────────────────────────────────────────────
    (
        "Contrato de processamento de dados financeiros e de saúde de R$ 50.000.000,00. "
        "Sem qualquer menção à LGPD, base legal ou DPO. "
        "Multa unilateral de 50% sobre valor total sem contrapartida. "
        "Rescisão em 5 dias sem justificativa. "
        "Foro eleito: Câmara de Comércio Internacional de Paris. "
        "Sem limitação de responsabilidade. Propriedade intelectual ambígua.",
        3
    ),
    (
        "Acordo de licenciamento de software crítico de infraestrutura com "
        "transferência de dados pessoais para servidores no exterior sem "
        "garantias adequadas conforme LGPD Art. 33. "
        "Cláusula de rescisão imediata sem indenização. "
        "Multa de 60% por qualquer descumprimento contratual. "
        "Arbitragem em câmara privada estrangeira sem regulamento claro.",
        3
    ),

    # Mais exemplos para cada nível
    ("Serviço de jardinagem mensal, R$ 1.200,00, sem dados pessoais, "
     "multa 5%, rescisão 30 dias, foro São Paulo.", 0),
    ("Consultoria tributária mensal R$ 3.000,00, sigilo, sem PI, "
     "rescisão 30 dias, multa 12%, sem dados pessoais.", 0),
    ("Serviço de segurança patrimonial com câmeras. Imagens armazenadas "
     "sem política de privacidade. Multa 20%. Rescisão 30 dias.", 1),
    ("ERP empresarial com dados de RH e folha de pagamento. LGPD mencionada "
     "mas sem base legal detalhada. Multa atraso 22%.", 1),
    ("Contrato SaaS para saúde — dados de pacientes, sem DPO, sem base legal LGPD. "
     "Multa 28%. Rescisão 15 dias. Valor R$ 2.000.000.", 2),
    ("Plataforma de crédito — dados financeiros sensíveis, "
     "tratamento indefinido, multa 32%, rescisão 3 dias, R$ 8.000.000.", 3),
]

print(f"Total exemplos seed: {len(CONTRATOS_SEED)}")

from collections import Counter
dist = Counter(label for _, label in CONTRATOS_SEED)
print("Distribuição:", {RISK_LABELS[k]: v for k, v in sorted(dist.items())})

# %% [markdown]
# ## 5. Ampliar com supervisão fraca
#
# Pegar contratos do Portal da Transparência (dados.gov.br) e aplicar regras.

# %%
# Textos sintéticos adicionais para ampliar o dataset via weak labeling
CONTRATOS_EXTRA = [
    # Gerados com variações para treino
    "Fornecimento de medicamentos hospitalares. Dados de pacientes tratados "
    "conforme LGPD Art. 7 inciso III. Multa 10%. Rescisão 30 dias.",

    "Desenvolvimento de app de telemedicina. Tratamento de dados sensíveis de saúde. "
    "Sem DPO nomeado. Sem base legal para dados sensíveis. Multa atraso 25%.",

    "Contrato de publicidade digital com coleta de cookies e dados comportamentais. "
    "Base legal: legítimo interesse conforme Art. 7 IX LGPD. Multa 12%.",

    "Licença de software contábil. Dados financeiros da empresa armazenados em nuvem. "
    "Sem menção à LGPD. Multa 30%. Rescisão unilateral em 7 dias. R$ 3.500.000.",

    "Obra de construção civil R$ 12.000.000. Sem limitação de responsabilidade. "
    "Sem arbitragem. Multa atraso 2% ao mês. Rescisão 60 dias.",

    "Terceirização de call center. Acesso a dados pessoais dos clientes. "
    "DPO nomeado. Consentimento coletado. Multa 15%. Foro São Paulo.",

    "Plataforma de RH com dados biométricos de funcionários. "
    "Sem base legal para biometria. Multa 35%. Valor R$ 4.200.000.",

    "Contrato de energia solar. Sem dados pessoais relevantes. "
    "Multa por atraso 0,5% ao dia. Rescisão 90 dias. Foro local.",

    "Sistema bancário com dados financeiros sensíveis. LGPD completa, DPO nomeado, "
    "base legal contrato. Limitação responsabilidade contratual. Arbitragem CAMARB.",

    "Franquia de alimentos. Sem dados pessoais. Multa 8%. "
    "Royalties 5% faturamento. Rescisão 90 dias. Foro SP.",
]

# Gerar labels via regras
extra_data = []
for texto in CONTRATOS_EXTRA:
    label, _ = calcular_risco(texto)
    extra_data.append((texto, label))

print(f"\nExemplos gerados via weak labeling: {len(extra_data)}")
for t, l in extra_data:
    print(f"  [{RISK_LABELS[l]:8}] {t[:70]}...")

# Combinar tudo
all_data = CONTRATOS_SEED + extra_data
random.shuffle(all_data)
print(f"\nTotal final: {len(all_data)}")

# %% [markdown]
# ## 6. Split e tokenização

# %%
texts  = [t for t, _ in all_data]
labels = [l for _, l in all_data]

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def make_dataset(X, y):
    enc = tokenizer(X, truncation=True, max_length=512, padding=False)
    enc["label"] = y
    return Dataset.from_dict(enc)

tok_train = make_dataset(X_train, y_train)
tok_val   = make_dataset(X_val,   y_val)
tok_test  = make_dataset(X_test,  y_test)

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# %% [markdown]
# ## 7. Modelo

# %%
ID2LABEL = RISK_LABELS
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# %% [markdown]
# ## 8. Métricas

# %%
def compute_metrics(p):
    preds  = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    return {
        "accuracy":    float((preds == labels).mean()),
        "f1_macro":    sk_f1(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": sk_f1(labels, preds, average="weighted", zero_division=0),
    }

# %% [markdown]
# ## 9. Treinar

# %%
training_args = TrainingArguments(
    output_dir="./analise_risco_output",
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## 10. Avaliar

# %%
results = trainer.evaluate(tok_test)
print("=== Resultado no teste ===")
for k, v in results.items():
    print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")

preds_out = trainer.predict(tok_test)
preds     = np.argmax(preds_out.predictions, axis=-1)
print("\n=== Relatório por nível de risco ===")
print(classification_report(
    preds_out.label_ids, preds,
    labels=[0, 1, 2, 3],
    target_names=["baixo", "médio", "alto", "crítico"],
    zero_division=0
))

# %% [markdown]
# ## 11. Salvar

# %%
import shutil
from google.colab import files

SAVE_DIR = "./hf_models/analise_risco"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("analise_risco.zip")
print("Download iniciado: analise_risco.zip")

# %% [markdown]
# ## 12. Inferência completa — análise de risco de um contrato

# %%
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok_inf   = AutoTokenizer.from_pretrained(SAVE_DIR)
model_inf = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model_inf.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inf.to(device)

def analisar_risco_contrato(texto: str) -> dict:
    """
    Função principal chamada pelo pipeline da API.
    Retorna risk_score, risk_level e attention_points detalhados.
    """
    # 1. Predição do nível de risco pelo modelo
    enc = tok_inf(texto, return_tensors="pt",
                  truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_inf(**enc).logits
        probs  = F.softmax(logits, dim=-1)[0].cpu().tolist()

    risk_label = int(np.argmax(probs))
    risk_level = ID2LABEL[risk_label]

    # Converter probabilidades em score 0-100
    risk_score = int(
        probs[0] * 10 +   # baixo → 0-10
        probs[1] * 40 +   # médio → 0-40
        probs[2] * 75 +   # alto  → 0-75
        probs[3] * 100    # crítico → 0-100
    )
    risk_score = min(risk_score, 100)

    # 2. Aplicar regras para pontos de atenção específicos
    _, achados = calcular_risco(texto)

    # Adicionar número de cláusula se encontrado (regex simples)
    for achado in achados:
        match = re.search(
            r"(?:cl[aá]usula|art(?:igo)?|§|item)\s*(\d+[\w.]*)",
            texto, re.I
        )
        if match:
            achado["clausula_referencia"] = match.group(0)

    return {
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "risk_probabilities": {
            "baixo":   round(probs[0], 4),
            "médio":   round(probs[1], 4),
            "alto":    round(probs[2], 4),
            "crítico": round(probs[3], 4),
        },
        "attention_points": achados,
        "total_findings":   len(achados),
    }


# Teste com contrato problemático
contrato_teste = """
CONTRATO DE PRESTAÇÃO DE SERVIÇOS DE TECNOLOGIA

Cláusula 1 — Objeto: Desenvolvimento de plataforma de saúde digital com
coleta e processamento de dados pessoais sensíveis de pacientes.

Cláusula 8 — Dados Pessoais: A contratada terá acesso aos dados dos pacientes
para fins de análise e processamento, sem necessidade de consentimento individual.

Cláusula 12 — Multa: Em caso de rescisão antecipada pela contratante, incidirá
multa de 35% sobre o valor total do contrato de R$ 8.500.000,00.

Cláusula 15 — Rescisão: Qualquer das partes poderá rescindir este contrato
com aviso prévio de apenas 7 dias.

Cláusula 20 — Foro: Fica eleita a Câmara de Comércio Internacional de Paris
para dirimir quaisquer controvérsias.

Valor total: R$ 8.500.000,00 (oito milhões e quinhentos mil reais).
"""

resultado = analisar_risco_contrato(contrato_teste)

print("=" * 60)
print("ANÁLISE DE RISCO DO CONTRATO")
print("=" * 60)
print(f"Score de Risco   : {resultado['risk_score']}/100")
print(f"Nível de Risco   : {resultado['risk_level'].upper()}")
print(f"Total de achados : {resultado['total_findings']}")
print("\nProbabilidades por nível:")
for k, v in resultado["risk_probabilities"].items():
    bar = "█" * int(v * 30)
    print(f"  {k:8}: {bar} {v:.1%}")

print(f"\n{'─'*60}")
print("PONTOS DE ATENÇÃO:")
for i, a in enumerate(resultado["attention_points"], 1):
    print(f"\n  [{i}] Tipo      : {a['tipo']}")
    print(f"      Severidade: {a['severidade'].upper()}")
    print(f"      Descrição : {a['descricao']}")
    if "clausula_referencia" in a:
        print(f"      Referência: {a['clausula_referencia']}")
