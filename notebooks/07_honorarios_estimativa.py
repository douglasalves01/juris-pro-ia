device = "cuda" if torch.cuda.is_available() else "cpu"

!pip install transformers sentence-transformers scikit-learn joblib -q

# %% [markdown]
# ## 2. Imports

# %%
import json
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# %% [markdown]
# ## 3. Tabelas de honorários OAB por região e tipo de causa
#
# Baseado nas tabelas de honorários mínimos sugeridos pelas
# seccionals da OAB (publicamente disponíveis nos sites das seccionals).
# Valores em R$ (referência 2024).

# %%
# ── TABELAS POR SECCIONAL (mínimo sugerido) ──────────────────
OAB_TABLES = {
    # (tipo_causa, regiao): (fee_min, fee_max, base)
    # base: "fixo" | "causa" (% sobre valor da causa)

    # SP — OAB-SP Tabela 2024
    ("Trabalhista",     "SP"): (3500,  15000,  "fixo"),
    ("Consumidor",      "SP"): (2500,  12000,  "fixo"),
    ("Tributário",      "SP"): (5000,  50000,  "fixo"),
    ("Previdenciário",  "SP"): (2000,   8000,  "fixo"),
    ("Criminal",        "SP"): (4000,  30000,  "fixo"),
    ("Família",         "SP"): (3000,  20000,  "fixo"),
    ("Tecnologia",      "SP"): (5000,  40000,  "causa"),
    ("Serviços",        "SP"): (3000,  20000,  "causa"),
    ("Fornecimento",    "SP"): (4000,  30000,  "causa"),
    ("Parceria",        "SP"): (6000,  50000,  "causa"),
    ("Outros",          "SP"): (2500,  15000,  "fixo"),

    # RJ — OAB-RJ
    ("Trabalhista",     "RJ"): (3000,  12000,  "fixo"),
    ("Consumidor",      "RJ"): (2000,  10000,  "fixo"),
    ("Tributário",      "RJ"): (4500,  45000,  "fixo"),
    ("Previdenciário",  "RJ"): (1800,   7000,  "fixo"),
    ("Criminal",        "RJ"): (3500,  25000,  "fixo"),
    ("Família",         "RJ"): (2500,  18000,  "fixo"),
    ("Tecnologia",      "RJ"): (4500,  35000,  "causa"),
    ("Outros",          "RJ"): (2000,  12000,  "fixo"),

    # MG — OAB-MG
    ("Trabalhista",     "MG"): (2500,  10000,  "fixo"),
    ("Consumidor",      "MG"): (1800,   8000,  "fixo"),
    ("Tributário",      "MG"): (3500,  35000,  "fixo"),
    ("Previdenciário",  "MG"): (1500,   6000,  "fixo"),
    ("Criminal",        "MG"): (3000,  20000,  "fixo"),
    ("Família",         "MG"): (2000,  15000,  "fixo"),
    ("Tecnologia",      "MG"): (3500,  28000,  "causa"),
    ("Outros",          "MG"): (1800,  10000,  "fixo"),

    # RS — OAB-RS
    ("Trabalhista",     "RS"): (2800,  11000,  "fixo"),
    ("Consumidor",      "RS"): (2000,   9000,  "fixo"),
    ("Tributário",      "RS"): (4000,  40000,  "fixo"),
    ("Previdenciário",  "RS"): (1600,   6500,  "fixo"),
    ("Criminal",        "RS"): (3200,  22000,  "fixo"),
    ("Família",         "RS"): (2200,  16000,  "fixo"),
    ("Tecnologia",      "RS"): (4000,  30000,  "causa"),
    ("Outros",          "RS"): (2000,  11000,  "fixo"),

    # BA — OAB-BA
    ("Trabalhista",     "BA"): (2000,   8000,  "fixo"),
    ("Consumidor",      "BA"): (1500,   6000,  "fixo"),
    ("Tributário",      "BA"): (3000,  28000,  "fixo"),
    ("Previdenciário",  "BA"): (1200,   5000,  "fixo"),
    ("Criminal",        "BA"): (2500,  18000,  "fixo"),
    ("Família",         "BA"): (1800,  12000,  "fixo"),
    ("Outros",          "BA"): (1500,   8000,  "fixo"),

    # DF — OAB-DF
    ("Trabalhista",     "DF"): (3200,  13000,  "fixo"),
    ("Consumidor",      "DF"): (2200,  11000,  "fixo"),
    ("Tributário",      "DF"): (5000,  48000,  "fixo"),
    ("Previdenciário",  "DF"): (1800,   7500,  "fixo"),
    ("Criminal",        "DF"): (4000,  28000,  "fixo"),
    ("Família",         "DF"): (2800,  18000,  "fixo"),
    ("Tecnologia",      "DF"): (5000,  42000,  "causa"),
    ("Outros",          "DF"): (2200,  13000,  "fixo"),

    # PR — OAB-PR
    ("Trabalhista",     "PR"): (2800,  11000,  "fixo"),
    ("Consumidor",      "PR"): (2000,   9000,  "fixo"),
    ("Tributário",      "PR"): (4000,  38000,  "fixo"),
    ("Previdenciário",  "PR"): (1600,   6500,  "fixo"),
    ("Criminal",        "PR"): (3200,  22000,  "fixo"),
    ("Família",         "PR"): (2200,  15000,  "fixo"),
    ("Tecnologia",      "PR"): (4000,  32000,  "causa"),
    ("Outros",          "PR"): (2000,  10000,  "fixo"),

    # PE — OAB-PE
    ("Trabalhista",     "PE"): (2200,   9000,  "fixo"),
    ("Consumidor",      "PE"): (1600,   7000,  "fixo"),
    ("Tributário",      "PE"): (3200,  30000,  "fixo"),
    ("Previdenciário",  "PE"): (1300,   5500,  "fixo"),
    ("Criminal",        "PE"): (2800,  20000,  "fixo"),
    ("Família",         "PE"): (2000,  13000,  "fixo"),
    ("Outros",          "PE"): (1600,   9000,  "fixo"),

    # CE — OAB-CE
    ("Trabalhista",     "CE"): (2000,   8500,  "fixo"),
    ("Consumidor",      "CE"): (1500,   6500,  "fixo"),
    ("Tributário",      "CE"): (3000,  28000,  "fixo"),
    ("Previdenciário",  "CE"): (1200,   5000,  "fixo"),
    ("Criminal",        "CE"): (2600,  18000,  "fixo"),
    ("Família",         "CE"): (1800,  12000,  "fixo"),
    ("Outros",          "CE"): (1500,   8000,  "fixo"),

    # GO — OAB-GO
    ("Trabalhista",     "GO"): (2500,  10000,  "fixo"),
    ("Consumidor",      "GO"): (1800,   8000,  "fixo"),
    ("Tributário",      "GO"): (3500,  32000,  "fixo"),
    ("Previdenciário",  "GO"): (1400,   6000,  "fixo"),
    ("Criminal",        "GO"): (3000,  20000,  "fixo"),
    ("Família",         "GO"): (2000,  14000,  "fixo"),
    ("Outros",          "GO"): (1800,   9000,  "fixo"),

    # SC — OAB-SC
    ("Trabalhista",     "SC"): (2800,  11000,  "fixo"),
    ("Consumidor",      "SC"): (2000,   8500,  "fixo"),
    ("Tributário",      "SC"): (4000,  37000,  "fixo"),
    ("Previdenciário",  "SC"): (1600,   6200,  "fixo"),
    ("Criminal",        "SC"): (3200,  21000,  "fixo"),
    ("Família",         "SC"): (2200,  15000,  "fixo"),
    ("Tecnologia",      "SC"): (4000,  30000,  "causa"),
    ("Outros",          "SC"): (2000,  10000,  "fixo"),

    # PA — OAB-PA
    ("Trabalhista",     "PA"): (1800,   7500,  "fixo"),
    ("Consumidor",      "PA"): (1400,   6000,  "fixo"),
    ("Tributário",      "PA"): (2800,  25000,  "fixo"),
    ("Previdenciário",  "PA"): (1100,   4500,  "fixo"),
    ("Criminal",        "PA"): (2500,  17000,  "fixo"),
    ("Família",         "PA"): (1700,  11000,  "fixo"),
    ("Outros",          "PA"): (1400,   7000,  "fixo"),

    # MA — OAB-MA
    ("Trabalhista",     "MA"): (1700,   7000,  "fixo"),
    ("Consumidor",      "MA"): (1300,   5500,  "fixo"),
    ("Tributário",      "MA"): (2600,  23000,  "fixo"),
    ("Previdenciário",  "MA"): (1000,   4200,  "fixo"),
    ("Criminal",        "MA"): (2400,  16000,  "fixo"),
    ("Família",         "MA"): (1600,  10000,  "fixo"),
    ("Outros",          "MA"): (1300,   6500,  "fixo"),

    # PI — OAB-PI
    ("Trabalhista",     "PI"): (1600,   6500,  "fixo"),
    ("Consumidor",      "PI"): (1200,   5000,  "fixo"),
    ("Tributário",      "PI"): (2500,  22000,  "fixo"),
    ("Previdenciário",  "PI"): (1000,   4000,  "fixo"),
    ("Criminal",        "PI"): (2200,  15000,  "fixo"),
    ("Família",         "PI"): (1500,   9500,  "fixo"),
    ("Outros",          "PI"): (1200,   6000,  "fixo"),

    # AL — OAB-AL
    ("Trabalhista",     "AL"): (1700,   7000,  "fixo"),
    ("Consumidor",      "AL"): (1300,   5500,  "fixo"),
    ("Tributário",      "AL"): (2600,  23000,  "fixo"),
    ("Previdenciário",  "AL"): (1000,   4200,  "fixo"),
    ("Criminal",        "AL"): (2400,  16000,  "fixo"),
    ("Família",         "AL"): (1600,  10000,  "fixo"),
    ("Outros",          "AL"): (1300,   6500,  "fixo"),

    # SE — OAB-SE
    ("Trabalhista",     "SE"): (1800,   7500,  "fixo"),
    ("Consumidor",      "SE"): (1400,   5800,  "fixo"),
    ("Tributário",      "SE"): (2700,  24000,  "fixo"),
    ("Previdenciário",  "SE"): (1100,   4400,  "fixo"),
    ("Criminal",        "SE"): (2500,  17000,  "fixo"),
    ("Família",         "SE"): (1700,  10500,  "fixo"),
    ("Outros",          "SE"): (1400,   7000,  "fixo"),

    # RN — OAB-RN
    ("Trabalhista",     "RN"): (1900,   8000,  "fixo"),
    ("Consumidor",      "RN"): (1400,   6000,  "fixo"),
    ("Tributário",      "RN"): (2800,  25000,  "fixo"),
    ("Previdenciário",  "RN"): (1100,   4500,  "fixo"),
    ("Criminal",        "RN"): (2600,  17500,  "fixo"),
    ("Família",         "RN"): (1700,  11000,  "fixo"),
    ("Outros",          "RN"): (1400,   7000,  "fixo"),

    # PB — OAB-PB
    ("Trabalhista",     "PB"): (1800,   7500,  "fixo"),
    ("Consumidor",      "PB"): (1300,   5800,  "fixo"),
    ("Tributário",      "PB"): (2700,  24000,  "fixo"),
    ("Previdenciário",  "PB"): (1100,   4400,  "fixo"),
    ("Criminal",        "PB"): (2500,  17000,  "fixo"),
    ("Família",         "PB"): (1600,  10500,  "fixo"),
    ("Outros",          "PB"): (1300,   6800,  "fixo"),

    # MT — OAB-MT
    ("Trabalhista",     "MT"): (2200,   9000,  "fixo"),
    ("Consumidor",      "MT"): (1700,   7200,  "fixo"),
    ("Tributário",      "MT"): (3200,  29000,  "fixo"),
    ("Previdenciário",  "MT"): (1300,   5500,  "fixo"),
    ("Criminal",        "MT"): (2800,  19000,  "fixo"),
    ("Família",         "MT"): (1900,  13000,  "fixo"),
    ("Outros",          "MT"): (1700,   8500,  "fixo"),

    # MS — OAB-MS
    ("Trabalhista",     "MS"): (2300,   9500,  "fixo"),
    ("Consumidor",      "MS"): (1700,   7500,  "fixo"),
    ("Tributário",      "MS"): (3300,  30000,  "fixo"),
    ("Previdenciário",  "MS"): (1400,   5800,  "fixo"),
    ("Criminal",        "MS"): (2900,  20000,  "fixo"),
    ("Família",         "MS"): (2000,  13500,  "fixo"),
    ("Outros",          "MS"): (1700,   9000,  "fixo"),

    # RO — OAB-RO
    ("Trabalhista",     "RO"): (2000,   8500,  "fixo"),
    ("Consumidor",      "RO"): (1500,   6500,  "fixo"),
    ("Tributário",      "RO"): (3000,  27000,  "fixo"),
    ("Previdenciário",  "RO"): (1200,   5000,  "fixo"),
    ("Criminal",        "RO"): (2600,  18000,  "fixo"),
    ("Família",         "RO"): (1800,  12000,  "fixo"),
    ("Outros",          "RO"): (1500,   7500,  "fixo"),

    # TO — OAB-TO
    ("Trabalhista",     "TO"): (1900,   8000,  "fixo"),
    ("Consumidor",      "TO"): (1400,   6000,  "fixo"),
    ("Tributário",      "TO"): (2800,  25000,  "fixo"),
    ("Previdenciário",  "TO"): (1100,   4500,  "fixo"),
    ("Criminal",        "TO"): (2500,  17000,  "fixo"),
    ("Família",         "TO"): (1700,  11000,  "fixo"),
    ("Outros",          "TO"): (1400,   7200,  "fixo"),

    # AC — OAB-AC
    ("Trabalhista",     "AC"): (1700,   7000,  "fixo"),
    ("Consumidor",      "AC"): (1300,   5500,  "fixo"),
    ("Tributário",      "AC"): (2500,  22000,  "fixo"),
    ("Previdenciário",  "AC"): (1000,   4000,  "fixo"),
    ("Criminal",        "AC"): (2300,  15500,  "fixo"),
    ("Família",         "AC"): (1500,  10000,  "fixo"),
    ("Outros",          "AC"): (1300,   6500,  "fixo"),

    # AM — OAB-AM
    ("Trabalhista",     "AM"): (2000,   8500,  "fixo"),
    ("Consumidor",      "AM"): (1500,   6500,  "fixo"),
    ("Tributário",      "AM"): (3000,  27000,  "fixo"),
    ("Previdenciário",  "AM"): (1200,   5000,  "fixo"),
    ("Criminal",        "AM"): (2600,  18000,  "fixo"),
    ("Família",         "AM"): (1800,  12000,  "fixo"),
    ("Outros",          "AM"): (1500,   7500,  "fixo"),

    # RR — OAB-RR
    ("Trabalhista",     "RR"): (1600,   6800,  "fixo"),
    ("Consumidor",      "RR"): (1200,   5200,  "fixo"),
    ("Tributário",      "RR"): (2400,  21000,  "fixo"),
    ("Previdenciário",  "RR"): (1000,   4000,  "fixo"),
    ("Criminal",        "RR"): (2200,  15000,  "fixo"),
    ("Família",         "RR"): (1500,   9500,  "fixo"),
    ("Outros",          "RR"): (1200,   6000,  "fixo"),

    # AP — OAB-AP
    ("Trabalhista",     "AP"): (1600,   6800,  "fixo"),
    ("Consumidor",      "AP"): (1200,   5200,  "fixo"),
    ("Tributário",      "AP"): (2400,  21000,  "fixo"),
    ("Previdenciário",  "AP"): (1000,   4000,  "fixo"),
    ("Criminal",        "AP"): (2200,  15000,  "fixo"),
    ("Família",         "AP"): (1500,   9500,  "fixo"),
    ("Outros",          "AP"): (1200,   6000,  "fixo"),

    # ES — OAB-ES
    ("Trabalhista",     "ES"): (2500,  10000,  "fixo"),
    ("Consumidor",      "ES"): (1800,   8000,  "fixo"),
    ("Tributário",      "ES"): (3500,  32000,  "fixo"),
    ("Previdenciário",  "ES"): (1400,   6000,  "fixo"),
    ("Criminal",        "ES"): (3000,  20000,  "fixo"),
    ("Família",         "ES"): (2000,  14000,  "fixo"),
    ("Outros",          "ES"): (1800,   9000,  "fixo"),
}

REGIOES     = sorted(set(r for _, r in OAB_TABLES.keys()))
TIPOS_CAUSA = sorted(set(t for t, _ in OAB_TABLES.keys()))

print("Regiões disponíveis :", REGIOES)
print("Tipos de causa      :", TIPOS_CAUSA)
print("Total de combinações:", len(OAB_TABLES))

# %% [markdown]
# ## 4. Gerar dataset de treinamento
#
# Geramos amostras sintéticas variando: tipo de causa, região,
# complexidade e valor da causa estimado.

# %%
import random
random.seed(42)

COMPLEXIDADE_MULT = {
    "simples":  0.8,
    "média":    1.0,
    "alta":     1.4,
    "muito_alta": 2.0,
}

VALOR_CAUSA_RANGES = {
    "Trabalhista":    (5000,    200000),
    "Consumidor":     (1000,     50000),
    "Tributário":     (10000,  5000000),
    "Previdenciário": (50000,   500000),
    "Criminal":       (0,            0),  # não se aplica
    "Família":        (10000,   800000),
    "Tecnologia":     (50000,  5000000),
    "Serviços":       (5000,    500000),
    "Fornecimento":   (10000,  2000000),
    "Parceria":       (50000, 10000000),
    "Outros":         (5000,    200000),
}

registros = []

for (tipo, regiao), (fee_min, fee_max, base) in OAB_TABLES.items():
    valor_min, valor_max = VALOR_CAUSA_RANGES.get(tipo, (5000, 200000))

    for _ in range(50):  # 50 amostras por combinação
        complexidade = random.choice(list(COMPLEXIDADE_MULT.keys()))
        mult         = COMPLEXIDADE_MULT[complexidade]
        valor_causa  = random.randint(valor_min or 5000, max(valor_max, 10000))

        # Honorário real com variação
        fee_real_min = fee_min * mult * random.uniform(0.9, 1.1)
        fee_real_max = fee_max * mult * random.uniform(0.9, 1.1)

        # Se base é % sobre valor da causa
        if base == "causa":
            pct_min = random.uniform(0.05, 0.10)
            pct_max = random.uniform(0.10, 0.20)
            fee_real_min = max(fee_min, valor_causa * pct_min * mult)
            fee_real_max = max(fee_max, valor_causa * pct_max * mult)

        registros.append({
            "tipo_causa":    tipo,
            "regiao":        regiao,
            "complexidade":  complexidade,
            "valor_causa":   float(valor_causa),
            "fee_min":       round(fee_real_min, 2),
            "fee_max":       round(fee_real_max, 2),
            "base":          base,
        })

df = pd.DataFrame(registros)
print(f"Dataset gerado: {len(df)} amostras")
print(df.groupby(["tipo_causa", "regiao"])[["fee_min", "fee_max"]].mean().round(0))

# %% [markdown]
# ## 5. Encoders para features categóricas

# %%
le_tipo = LabelEncoder().fit(TIPOS_CAUSA + ["Obras", "Criminal"])
le_reg  = LabelEncoder().fit(REGIOES)
le_comp = LabelEncoder().fit(list(COMPLEXIDADE_MULT.keys()))

# Features numéricas
X = np.column_stack([
    le_tipo.transform(df["tipo_causa"]),
    le_reg.transform(df["regiao"]),
    le_comp.transform(df["complexidade"]),
    np.log1p(df["valor_causa"]),  # log para estabilizar escala
])

y_min = df["fee_min"].values
y_max = df["fee_max"].values

X_train, X_test, ym_train, ym_test, yM_train, yM_test = train_test_split(
    X, y_min, y_max, test_size=0.2, random_state=42
)

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# %% [markdown]
# ## 6. Treinar modelos de regressão

# %%
params = dict(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    subsample=0.8,
)

model_min = GradientBoostingRegressor(**params)
model_max = GradientBoostingRegressor(**params)

model_min.fit(X_train, ym_train)
model_max.fit(X_train, yM_train)

# Avaliação
pred_min = model_min.predict(X_test)
pred_max = model_max.predict(X_test)

mae_min  = mean_absolute_error(ym_test, pred_min)
mape_min = mean_absolute_percentage_error(ym_test, pred_min)
mae_max  = mean_absolute_error(yM_test, pred_max)
mape_max = mean_absolute_percentage_error(yM_test, pred_max)

print(f"fee_min → MAE: R$ {mae_min:.0f} | MAPE: {mape_min:.1%}")
print(f"fee_max → MAE: R$ {mae_max:.0f} | MAPE: {mape_max:.1%}")

# %% [markdown]
# ## 7. Salvar modelos e encoders

# %%
import os, shutil
from google.colab import files

SAVE_DIR = "./hf_models/honorarios"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(model_min, f"{SAVE_DIR}/model_fee_min.joblib")
joblib.dump(model_max, f"{SAVE_DIR}/model_fee_max.joblib")
joblib.dump(le_tipo,   f"{SAVE_DIR}/le_tipo.joblib")
joblib.dump(le_reg,    f"{SAVE_DIR}/le_regiao.joblib")
joblib.dump(le_comp,   f"{SAVE_DIR}/le_complexidade.joblib")

oab_serializable = {
    f"{tipo}|{reg}": {"fee_min": mn, "fee_max": mx, "base": b}
    for (tipo, reg), (mn, mx, b) in OAB_TABLES.items()
}
with open(f"{SAVE_DIR}/oab_tables.json", "w") as f:
    json.dump(oab_serializable, f, ensure_ascii=False, indent=2)

shutil.make_archive(SAVE_DIR, "zip", SAVE_DIR)
files.download("honorarios.zip")
print("Download iniciado: honorarios.zip")

# %% [markdown]
# ## 8. Inferência — estimar honorários

# %%
def estimar_honorarios(
    tipo_causa:   str,
    regiao:       str,
    complexidade: str = "média",
    valor_causa:  float = 0.0,
) -> dict:
    """
    Estima honorários mínimos e máximos para um caso.

    Parâmetros:
        tipo_causa  : ex. "Trabalhista", "Consumidor", "Tecnologia"
        regiao      : sigla do estado ex. "SP", "RJ", "MG"
        complexidade: "simples" | "média" | "alta" | "muito_alta"
        valor_causa : valor estimado da causa em R$

    Retorna dict com fee_min, fee_max, justificativa e referência OAB.
    """
    # Normalizar inputs
    if tipo_causa not in le_tipo.classes_:
        tipo_causa = "Outros"
    if regiao not in le_reg.classes_:
        regiao = "SP"
    if complexidade not in le_comp.classes_:
        complexidade = "média"

    X_inf = np.array([[
        le_tipo.transform([tipo_causa])[0],
        le_reg.transform([regiao])[0],
        le_comp.transform([complexidade])[0],
        np.log1p(valor_causa),
    ]])

    fee_min = max(0, model_min.predict(X_inf)[0])
    fee_max = max(fee_min, model_max.predict(X_inf)[0])

    # Buscar referência na tabela OAB
    key = (tipo_causa, regiao)
    oab_ref = OAB_TABLES.get(key, OAB_TABLES.get((tipo_causa, "SP"), (0, 0, "fixo")))
    oab_min, oab_max, base = oab_ref

    return {
        "fee_min":        round(fee_min, 2),
        "fee_max":        round(fee_max, 2),
        "fee_sugerido":   round((fee_min + fee_max) / 2, 2),
        "base_calculo":   base,
        "referencia_oab": {
            "tabela":       f"OAB-{regiao}",
            "minimo_tabela": oab_min,
            "maximo_tabela": oab_max,
        },
        "complexidade":   complexidade,
        "justificativa": (
            f"Estimativa para causa {tipo_causa} com complexidade {complexidade} "
            f"na região {regiao}, conforme tabela OAB-{regiao} 2024."
        ),
    }


# Testes
casos_teste = [
    ("Trabalhista",    "SP", "alta",      85000),
    ("Tributário",     "RJ", "muito_alta", 2500000),
    ("Previdenciário", "MG", "simples",   120000),
    ("Consumidor",     "RS", "média",      15000),
    ("Tecnologia",     "DF", "alta",      800000),
]

print("=" * 65)
print("ESTIMATIVAS DE HONORÁRIOS")
print("=" * 65)
for tipo, reg, comp, valor in casos_teste:
    r = estimar_honorarios(tipo, reg, comp, valor)
    print(f"\n  Causa      : {tipo} | Região: {reg} | Complexidade: {comp}")
    print(f"  Valor causa: R$ {valor:,.0f}")
    print(f"  Fee mínimo : R$ {r['fee_min']:,.2f}")
    print(f"  Fee máximo : R$ {r['fee_max']:,.2f}")
    print(f"  Fee sugerido: R$ {r['fee_sugerido']:,.2f}")
    print(f"  Base OAB   : R$ {r['referencia_oab']['minimo_tabela']:,} – "
          f"R$ {r['referencia_oab']['maximo_tabela']:,}")
