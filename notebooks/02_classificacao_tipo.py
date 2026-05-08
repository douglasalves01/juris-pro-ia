
!pip install transformers datasets accelerate scikit-learn -q

# %% [markdown]
# ## 2. Imports

# %%
import random
import json
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
# ## 3. Dataset seed — exemplos representativos por tipo
#
# Cada entrada contém fragmentos reais de contratos/processos brasileiros.
# A base seed garante que o modelo aprenda o vocabulário de cada domínio.
# Você pode aumentar o dataset adicionando mais textos reais.

# %%
SEED_DATA = [
    # ── TECNOLOGIA ──────────────────────────────────────────────────────────────
    ("Contrato de prestação de serviços de desenvolvimento de software sob medida "
     "para gestão de processos internos, incluindo suporte técnico e manutenção "
     "evolutiva do sistema, mediante remuneração mensal de R$ 25.000,00.",
     "Tecnologia"),
    ("Contratação de serviços de computação em nuvem (cloud computing), "
     "hospedagem de servidores, backup automatizado e segurança da informação "
     "conforme normas ISO 27001 e LGPD.", "Tecnologia"),
    ("Licença de uso de software ERP para controle financeiro, RH e logística, "
     "com implantação, treinamento de usuários e suporte helpdesk 24h.", "Tecnologia"),
    ("Contrato de desenvolvimento de aplicativo mobile para iOS e Android, "
     "com integração a APIs bancárias e conformidade com PCI-DSS.", "Tecnologia"),
    ("Aquisição de equipamentos de TI: servidores, switches, firewalls e "
     "infraestrutura de rede com garantia on-site de 36 meses.", "Tecnologia"),
    ("Serviços de cibersegurança: pentest, análise de vulnerabilidades, SOC 24/7 "
     "e resposta a incidentes conforme framework NIST.", "Tecnologia"),
    ("Implantação de sistema de gestão jurídica com módulos de contratos, "
     "processos, compliance e relatórios gerenciais.", "Tecnologia"),
    ("Contrato de inteligência artificial para análise preditiva de dados "
     "comerciais com modelos de machine learning e dashboards em tempo real.", "Tecnologia"),

    # ── SERVIÇOS ─────────────────────────────────────────────────────────────────
    ("Contrato de prestação de serviços de limpeza e conservação predial, "
     "incluindo fornecimento de materiais e uniformes aos colaboradores.", "Serviços"),
    ("Contratação de empresa de segurança patrimonial para vigilância eletrônica "
     "e ronda ostensiva nas dependências da contratante.", "Serviços"),
    ("Prestação de serviços de consultoria empresarial em gestão estratégica, "
     "reestruturação organizacional e planejamento financeiro.", "Serviços"),
    ("Contrato de serviços de contabilidade, escrituração fiscal, apuração de "
     "impostos e elaboração de demonstrações financeiras.", "Serviços"),
    ("Serviços de transporte executivo e locação de veículos com motorista, "
     "disponíveis 24 horas, 7 dias por semana.", "Serviços"),
    ("Prestação de serviços de manutenção preventiva e corretiva de "
     "equipamentos industriais com técnicos especializados.", "Serviços"),
    ("Contrato de assessoria de imprensa, relações públicas e comunicação "
     "corporativa incluindo gestão de redes sociais.", "Serviços"),
    ("Serviços de auditoria interna e externa conforme normas CFC e CVM.", "Serviços"),

    # ── FORNECIMENTO ─────────────────────────────────────────────────────────────
    ("Contrato de fornecimento de insumos industriais, matérias-primas e "
     "embalagens com entrega JIT (just-in-time) e controle de estoque mínimo.", "Fornecimento"),
    ("Fornecimento de medicamentos e material hospitalar para unidades de saúde, "
     "conforme registro ANVISA e especificações técnicas do edital.", "Fornecimento"),
    ("Contrato de compra e venda de equipamentos médico-hospitalares com "
     "instalação, calibração e certificação de conformidade.", "Fornecimento"),
    ("Fornecimento de uniformes, EPIs e materiais de escritório pelo período "
     "de 12 meses, com possibilidade de prorrogação.", "Fornecimento"),
    ("Contrato de fornecimento de combustíveis e lubrificantes para frota de "
     "veículos, com controle via sistema de abastecimento.", "Fornecimento"),
    ("Fornecimento de gêneros alimentícios para refeitório corporativo, "
     "observadas as normas da ANVISA e HACCP.", "Fornecimento"),

    # ── PARCERIA ─────────────────────────────────────────────────────────────────
    ("Acordo de parceria estratégica para desenvolvimento conjunto de produtos "
     "inovadores, com compartilhamento de custos P&D e royalties.", "Parceria"),
    ("Memorando de entendimento (MOU) para exploração conjunta de mercado "
     "internacional, com constituição de joint venture.", "Parceria"),
    ("Acordo de cooperação técnica entre universidade e empresa para pesquisa "
     "aplicada, transferência de tecnologia e propriedade intelectual.", "Parceria"),
    ("Contrato de franquia: cessão de uso de marca, know-how operacional e "
     "suporte continuado mediante royalties mensais de 5% sobre faturamento.", "Parceria"),
    ("Acordo de distribuição exclusiva com compartilhamento de território e "
     "metas de desempenho trimestrais.", "Parceria"),
    ("Parceria público-privada (PPP) para implantação e operação de "
     "infraestrutura de saneamento básico.", "Parceria"),

    # ── OBRAS ─────────────────────────────────────────────────────────────────────
    ("Contrato de empreitada global para construção de edifício comercial de "
     "12 andares, regime de empreitada por preço global, prazo 24 meses.", "Obras"),
    ("Contrato de reforma e adequação de unidade fabril, incluindo obras civis, "
     "instalações elétricas e HVAC conforme normas ABNT.", "Obras"),
    ("Contratação de empresa para execução de obra de pavimentação asfáltica "
     "de 15 km de rodovia municipal.", "Obras"),
    ("Contrato de construção civil para residencial multifamiliar com 200 "
     "unidades habitacionais de interesse social.", "Obras"),
    ("Serviços de engenharia para ampliação de planta industrial: fundações, "
     "estrutura metálica e cobertura.", "Obras"),

    # ── TRABALHISTA ───────────────────────────────────────────────────────────────
    ("Reclamação trabalhista movida pelo empregado requerendo pagamento de horas "
     "extras, adicional noturno e verbas rescisórias conforme CLT.", "Trabalhista"),
    ("Ação de reconhecimento de vínculo empregatício cumulada com pedido de "
     "FGTS, férias proporcionais e 13º salário.", "Trabalhista"),
    ("Acordo coletivo de trabalho entre sindicato dos trabalhadores e empresa, "
     "fixando piso salarial, jornada e benefícios.", "Trabalhista"),
    ("Reclamação trabalhista por assédio moral no ambiente de trabalho com "
     "pedido de indenização por danos morais e materiais.", "Trabalhista"),
    ("Rescisão contratual por justa causa: apuração de falta grave, "
     "inquérito judicial e defesa do empregado.", "Trabalhista"),
    ("Ação coletiva do Ministério Público do Trabalho por irregularidades "
     "no contrato de terceirização de mão de obra.", "Trabalhista"),

    # ── PREVIDENCIÁRIO ────────────────────────────────────────────────────────────
    ("Recurso ao INSS para revisão de benefício de aposentadoria por tempo de "
     "contribuição, revisão do salário de benefício e competência do cálculo.", "Previdenciário"),
    ("Ação de concessão de auxílio-doença negado administrativamente, com "
     "pedido de tutela antecipada para imediato restabelecimento.", "Previdenciário"),
    ("Pedido de reconhecimento de atividade especial para fins de aposentadoria "
     "antecipada, com perícia técnica e PPP.", "Previdenciário"),
    ("Ação regressiva do INSS em face de empregador por acidente de trabalho "
     "decorrente de negligência nas normas de segurança.", "Previdenciário"),
    ("Recurso de revisão de benefício com aplicação do teto do RGPS e "
     "atualização monetária pelo INPC.", "Previdenciário"),

    # ── TRIBUTÁRIO ────────────────────────────────────────────────────────────────
    ("Mandado de segurança contra cobrança indevida de ICMS sobre operação "
     "de exportação, com pedido liminar para suspensão da exigibilidade.", "Tributário"),
    ("Ação anulatória de auto de infração lavrado pela Receita Federal por "
     "suposta omissão de receitas e irregularidade no IRPJ.", "Tributário"),
    ("Embargos à execução fiscal opostos pelo contribuinte contestando a "
     "certidão de dívida ativa e defendendo a prescrição do crédito.", "Tributário"),
    ("Consulta tributária sobre regime de tributação de software SaaS: "
     "incidência de ISSQN versus ICMS conforme ADI 1.945.", "Tributário"),
    ("Planejamento tributário para reestruturação societária com foco em "
     "redução da carga de CSLL e IRPJ.", "Tributário"),

    # ── CONSUMIDOR ────────────────────────────────────────────────────────────────
    ("Ação de indenização por danos morais e materiais decorrentes de produto "
     "defeituoso adquirido pelo consumidor, com pedido de devolução do valor.", "Consumidor"),
    ("Demanda consumerista contra operadora de plano de saúde por recusa "
     "de cobertura de procedimento médico necessário.", "Consumidor"),
    ("Ação revisional de contrato bancário por cobrança abusiva de juros "
     "acima do limite legal e capitalização ilegal.", "Consumidor"),
    ("Reclamação por negativação indevida do nome do consumidor em cadastros "
     "de inadimplentes sem débito pendente.", "Consumidor"),
    ("Ação coletiva em defesa de consumidores lesados por prática comercial "
     "abusiva de empresa de telecomunicações.", "Consumidor"),

    # ── FAMÍLIA ───────────────────────────────────────────────────────────────────
    ("Ação de divórcio litigioso com pedido de partilha de bens adquiridos na "
     "constância do casamento e guarda dos filhos menores.", "Família"),
    ("Ação de alimentos: pedido de fixação de pensão alimentícia para filho "
     "menor com base na capacidade econômica do alimentante.", "Família"),
    ("Interdição judicial e curatela de pessoa com transtorno mental grave, "
     "com nomeação de curador e prestação de contas.", "Família"),
    ("Ação de investigação de paternidade cumulada com pedido de alimentos "
     "e registro de nascimento.", "Família"),
    ("Inventário e arrolamento de bens com partilha amigável entre herdeiros "
     "conforme plano de partilha acordado.", "Família"),

    # ── CRIMINAL ──────────────────────────────────────────────────────────────────
    ("Defesa criminal em ação penal por crime de estelionato qualificado "
     "mediante fraude eletrônica, com pedido de absolvição.", "Criminal"),
    ("Habeas corpus preventivo para trancar ação penal por atipicidade da "
     "conduta e falta de justa causa.", "Criminal"),
    ("Resposta à acusação em processo penal por crime de peculato doloso "
     "com requerimento de diligências e oitiva de testemunhas.", "Criminal"),
    ("Recurso em sentido estrito contra decisão de pronúncia em caso de "
     "homicídio qualificado perante o Tribunal do Júri.", "Criminal"),
    ("Defesa em processo administrativo disciplinar com requerimento de "
     "anulação por cerceamento de defesa.", "Criminal"),

    # ── OUTROS ────────────────────────────────────────────────────────────────────
    ("Contrato de locação comercial de imóvel para fins de varejo com prazo "
     "de 60 meses, reajuste anual pelo IGP-M e fiança locatícia.", "Outros"),
    ("Acordo extrajudicial de mediação para resolução de conflito entre "
     "sócios com previsão de dissolução parcial.", "Outros"),
    ("Contrato de cessão de direitos autorais sobre obra literária, com "
     "exclusividade territorial e prazo de 10 anos.", "Outros"),
    ("Contrato de seguro de responsabilidade civil profissional (E&O) com "
     "cobertura de R$ 5.000.000,00 e franquia de 10%.", "Outros"),
]

print(f"Total de exemplos seed: {len(SEED_DATA)}")

labels_uniq = sorted(set(l for _, l in SEED_DATA))
label2id    = {l: i for i, l in enumerate(labels_uniq)}
id2label    = {i: l for l, i in label2id.items()}
print("Classes:", labels_uniq)

# %% [markdown]
# ## 4. Montar DataFrame e split

# %%
texts  = [t for t, _ in SEED_DATA]
labels = [label2id[l] for _, l in SEED_DATA]

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42  # sem stratify: subset pequeno demais
)

print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")

train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
val_ds   = Dataset.from_dict({"text": X_val,   "label": y_val})
test_ds  = Dataset.from_dict({"text": X_test,  "label": y_test})

full_ds  = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

# %% [markdown]
# ## 5. Tokenização

# %%
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

tokenized_ds = full_ds.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# %% [markdown]
# ## 6. Modelo

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels_uniq),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# %% [markdown]
# ## 7. Métricas

# %%
def compute_metrics(p):
    preds  = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    f1_macro = sk_f1(labels, preds, average="macro", zero_division=0)
    f1_weighted = sk_f1(labels, preds, average="weighted", zero_division=0)
    acc = (preds == labels).mean()
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

# %% [markdown]
# ## 8. Treinar

# %%
training_args = TrainingArguments(
    output_dir="./classificacao_tipo_output",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=30,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## 9. Avaliar

# %%
results = trainer.evaluate(tokenized_ds["test"])
print("=== Resultado no teste ===")
for k, v in results.items():
    print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

preds_out = trainer.predict(tokenized_ds["test"])
preds     = np.argmax(preds_out.predictions, axis=-1)
print("\n=== Relatório por classe ===")
print(classification_report(
    preds_out.label_ids, preds,
    target_names=labels_uniq, zero_division=0
))

# %% [markdown]
# ## 10. Salvar no Google Drive

# %%
import shutil, json
from google.colab import files

SAVE_DIR = "./hf_models/classificacao_tipo"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(f"{SAVE_DIR}/label_map.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("classificacao_tipo.zip")
print("Download iniciado: classificacao_tipo.zip")

# %% [markdown]
# ## 11. Teste de inferência

# %%
from transformers import pipeline

clf_pipe = pipeline(
    "text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    device=0 if torch.cuda.is_available() else -1,
)

testes = [
    "Contrato de desenvolvimento de sistema web com API REST e banco PostgreSQL.",
    "Reclamação trabalhista por horas extras não pagas e FGTS em atraso.",
    "Fornecimento de medicamentos para rede hospitalar pública conforme ANVISA.",
    "Ação de divórcio com guarda compartilhada e partilha de imóveis.",
    "Mandado de segurança contra cobrança abusiva de ICMS em exportação.",
]

print("=== Classificações ===")
for t in testes:
    r = clf_pipe(t, truncation=True)[0]
    print(f"  [{r['label']:20}] score={r['score']:.3f} | {t[:70]}...")
