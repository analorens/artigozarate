# =============================================================
# ETAPA 9 — ÁRVORE DE DECISÃO COM AVALIAÇÃO COMPLETA
# =============================================================
# Execute APÓS o pipeline principal (etapas 1-8).
# Requer: pns2019_base_limpa_v3.xlsx (gerado pela Etapa 7)
#
# SOBRE VARIÁVEIS DIRETAS (verificado no dicionário PNS 2019):
#
# Variáveis do Excel (abas Hipertensos / Saudaveis):
#   C008     → Idade do morador                           (demográfica) [REMOVIDA DAS FEATURES]
#   V0026    → Tipo de situação censitária                (demográfica)
#   VDF003   → Renda domiciliar per capita                (socioeconômica)
#   VDD004A  → Nível de instrução                         (socioeconômica)
#   P006     → Dias/semana que come feijão                (alimentação)
#   P00901   → Dias/semana que come verdura/legume        (alimentação)
#   P01001   → Costuma comer verdura/legume (forma)       (alimentação)
#   P018     → Dias/semana que come frutas                (alimentação)
#   P019     → Vezes/dia que come frutas                  (alimentação)
#   P01601   → Dias/semana que toma suco natural          (alimentação)
#   P02001   → Dias/semana que toma suco de caixa/refresco(alimentação)
#   P02002   → Dias/semana que toma refrigerante          (alimentação)
#   P02601   → Percepção sobre consumo de sal             (alimentação)
#   P02602   → Dias/semana substitui refeição por lanche  (alimentação)
#   P027     → Frequência de consumo de álcool            (comportamental)
#   P02801   → Dias/semana que consome álcool             (comportamental)
#   P034     → Praticou exercício/esporte nos últ. 3 meses(comportamental)
#   Q00201   → Diagnóstico médico de hipertensão arterial → ALVO (y)
#
# Variáveis usadas APENAS como filtro de seleção das linhas
# (não exportadas para o Excel, logo sem risco de vazamento):
#   Q03001  → diabetes          Q068  → AVC / derrame
#   Q060    → colesterol alto   Q074  → asma
#   Q06306  → doença do coração Q079  → artrite / reumatismo
#   Q084    → problema de coluna Q088 → DORT
#   Q092    → depressão         Q11006→ outra doença mental
#   Q11604  → DPOC / enfisema   Q120  → câncer
#   Q124    → insuf. renal      Q128  → outra doença crônica
#
# CONCLUSÃO: nenhuma feature presente no Excel é proxy direto
# de hipertensão. Além do alvo (Q00201), a variável C008 (Idade)
# foi removida a pedido, deixando de ser usada como preditor.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

COR_H   = "#E8704C"
COR_S   = "#4C9BE8"
COR_ACC = "#5C8A5C"
COR_AUC = "#8A5C8A"

# =============================================================
# 0. LEITURA DO EXCEL
# =============================================================
print("="*65)
print("ETAPA 9 — ÁRVORE DE DECISÃO (sem a variável Idade)")
print("="*65)

arquivo_excel = "pns2019_base_limpa_v3.xlsx"
df_h = pd.read_excel(arquivo_excel, sheet_name="Hipertensos")
df_s = pd.read_excel(arquivo_excel, sheet_name="Saudaveis")

df_h["Q00201"] = 1   # hipertenso
df_s["Q00201"] = 0   # saudável

df = pd.concat([df_h, df_s], ignore_index=True)
print(f"  Total de registros carregados: {len(df):,}")
print(f"  Distribuição — Hipertensos: {(df['Q00201']==1).sum():,} | "
      f"Saudáveis: {(df['Q00201']==0).sum():,}")

# =============================================================
# 1. SEPARAÇÃO ALVO / FEATURES
# =============================================================
# Q00201 é o alvo (removido das features).
# C008 (Idade) foi removida a pedido — deixa de ser usada como
# preditor no modelo, restando apenas fatores socioeconômicos,
# comportamentais e alimentares.

y = df["Q00201"].values
X_raw = df.drop(columns=["Q00201", "C008"]).copy()
feature_names_orig = list(X_raw.columns)

print(f"\n  Features utilizadas ({len(feature_names_orig)}):")
# Mapeamento legível pelo dicionário (Idade removida)
descricoes = {
    "V0026":   "Situação censitária (urbano/rural)",
    "VDF003":  "Renda domiciliar per capita",
    "VDD004A": "Nível de instrução",
    "P006":    "Dias/sem. come feijão",
    "P00901":  "Dias/sem. come verdura/legume",
    "P01001":  "Forma de consumo de verdura/legume",
    "P018":    "Dias/sem. come frutas",
    "P019":    "Vezes/dia come frutas",
    "P01601":  "Dias/sem. toma suco natural",
    "P02001":  "Dias/sem. toma suco caixa/refresco",
    "P02002":  "Dias/sem. toma refrigerante",
    "P02601":  "Percepção de consumo de sal",
    "P02602":  "Dias/sem. substitui refeição por lanche",
    "P027":    "Frequência de consumo de álcool",
    "P02801":  "Dias/sem. consome álcool",
    "P034":    "Praticou exercício físico (últ. 3 meses)",
}
for col in feature_names_orig:
    label = descricoes.get(col, col)
    # Detecta features cartesianas (PC_xxx_yyy)
    if col.startswith("PC_"):
        label = f"Feature cartesiana: {col}"
    print(f"    {col:<14} → {label}")

# =============================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================
encoders = {}
for col in X_raw.columns:
    if X_raw[col].dtype == object or str(X_raw[col].dtype).startswith("category"):
        le = LabelEncoder()
        X_raw[col] = X_raw[col].astype(str).fillna("__NaN__")
        X_raw[col] = le.fit_transform(X_raw[col])
        encoders[col] = le
    else:
        med = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(med)

X = X_raw.values
feature_names = list(X_raw.columns)
print(f"\n  Shape final de X: {X.shape}")

# =============================================================
# 3. ÁRVORE DE DECISÃO — VALIDAÇÃO CRUZADA (10-fold)
# =============================================================
print("\n" + "="*65)
print("  VALIDAÇÃO CRUZADA 10-FOLD")
print("="*65)

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision":         "precision",
    "recall":            "recall",
    "f1":                "f1",
    "roc_auc":           "roc_auc",
}

cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring,
                             return_train_score=True, n_jobs=-1)

def fmt(arr):
    return f"{arr.mean():.4f} ± {arr.std():.4f}"

print(f"\n  {'Métrica':<25} {'Treino':>22}  {'Validação':>22}")
print("  " + "-"*72)
metricas_cv = {
    "Acurácia":            ("accuracy",          "train_accuracy"),
    "Acurácia Balanceada": ("balanced_accuracy",  "train_balanced_accuracy"),
    "Precisão":            ("precision",          "train_precision"),
    "Recall (Sensib.)":    ("recall",             "train_recall"),
    "F1-Score":            ("f1",                 "train_f1"),
    "AUC-ROC":             ("roc_auc",            "train_roc_auc"),
}
for label, (val_key, train_key) in metricas_cv.items():
    print(f"  {label:<25} {fmt(cv_results[f'train_{val_key}']):>24}  "
          f"{fmt(cv_results[f'test_{val_key}']):>24}")

# =============================================================
# 4. MODELO FINAL (treino em todos os dados)
# =============================================================
clf.fit(X, y)
y_pred  = clf.predict(X)
y_proba = clf.predict_proba(X)[:, 1]

print("\n" + "="*65)
print("  RELATÓRIO — CONJUNTO COMPLETO (modelo final)")
print("="*65)
print(classification_report(y, y_pred, target_names=["Saudável", "Hipertenso"]))

# =============================================================
# 5. GRÁFICOS
# =============================================================
print("\n  Gerando gráficos…")

# ── 5.1 Curva ROC (por fold + média) ─────────────────────────
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
tprs, aucs_fold = [], []
mean_fpr = np.linspace(0, 1, 200)

for train_idx, test_idx in cv.split(X, y):
    clf_f = DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42,
    )
    clf_f.fit(X[train_idx], y[train_idx])
    prob = clf_f.predict_proba(X[test_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y[test_idx], prob)
    aucs_fold.append(auc(fpr, tpr))
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    ax_roc.plot(fpr, tpr, color=COR_AUC, alpha=0.15, lw=1)

mean_tpr = np.mean(tprs, axis=0)
std_tpr  = np.std(tprs, axis=0)
mean_auc = np.mean(aucs_fold); std_auc = np.std(aucs_fold)

ax_roc.plot(mean_fpr, mean_tpr, color=COR_AUC, lw=2.5,
            label=f"Média ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
ax_roc.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr,
                    alpha=0.2, color=COR_AUC, label="± 1 desvio padrão")
ax_roc.plot([0,1],[0,1],"k--", lw=1, label="Aleatório (AUC = 0.50)")
ax_roc.set_xlabel("Taxa de Falsos Positivos (1 − Especificidade)", fontsize=11)
ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=11)
ax_roc.set_title("Curva ROC — Validação Cruzada 10-Fold\nÁrvore de Decisão | PNS 2019 (sem Idade)", fontsize=12)
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.set_xlim([-0.01,1.01]); ax_roc.set_ylim([-0.01,1.01])
ax_roc.grid(alpha=0.3); ax_roc.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig_roc.savefig("e9_curva_roc.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5.2 Matriz de confusão ────────────────────────────────────
fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Saudável", "Hipertenso"])
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
ax_cm.set_title("Matriz de Confusão — Modelo Final\n(Conjunto Completo, sem Idade)", fontsize=12)
plt.tight_layout()
fig_cm.savefig("e9_matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5.3 Importância das variáveis (com nomes legíveis) ────────
fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
importancias = pd.Series(clf.feature_importances_, index=feature_names)
top_n  = min(15, len(importancias))
top_imp = importancias.nlargest(top_n).sort_values()
labels_leg = [descricoes.get(n, n) for n in top_imp.index]
cores_imp  = [COR_H if v > top_imp.median() else COR_S for v in top_imp.values]
ax_imp.barh(labels_leg, top_imp.values, color=cores_imp, edgecolor="white")
ax_imp.set_xlabel("Importância (Gini)", fontsize=10)
ax_imp.set_title(f"Top {top_n} Variáveis Mais Importantes\nÁrvore de Decisão | PNS 2019 (sem Idade)", fontsize=12)
ax_imp.spines[["top","right"]].set_visible(False)
ax_imp.grid(axis="x", alpha=0.3)
for i, (v, c) in enumerate(zip(top_imp.values, top_imp.index)):
    ax_imp.text(v + 0.0005, i, f"{v:.4f}  [{c}]", va="center", fontsize=7.5, color="#333")
plt.tight_layout()
fig_imp.savefig("e9_importancia_variaveis.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5.4 Curva de aprendizado ──────────────────────────────────
fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
tr_sizes, tr_scores, va_scores = learning_curve(
    clf, X, y, cv=cv, scoring="balanced_accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
)
tr_m = tr_scores.mean(axis=1); tr_s = tr_scores.std(axis=1)
va_m = va_scores.mean(axis=1); va_s = va_scores.std(axis=1)

ax_lc.plot(tr_sizes, tr_m, "o-", color=COR_H, lw=2, label="Treino")
ax_lc.fill_between(tr_sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color=COR_H)
ax_lc.plot(tr_sizes, va_m, "s--", color=COR_S, lw=2, label="Validação (CV)")
ax_lc.fill_between(tr_sizes, va_m-va_s, va_m+va_s, alpha=0.15, color=COR_S)
ax_lc.set_xlabel("Tamanho do conjunto de treino", fontsize=10)
ax_lc.set_ylabel("Acurácia Balanceada", fontsize=10)
ax_lc.set_title("Curva de Aprendizado\nÁrvore de Decisão | PNS 2019 (sem Idade)", fontsize=12)
ax_lc.legend(fontsize=9); ax_lc.set_ylim([0, 1.05])
ax_lc.grid(alpha=0.3); ax_lc.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig_lc.savefig("e9_curva_aprendizado.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5.5 Boxplot de métricas por fold ─────────────────────────
fig_box, ax_box = plt.subplots(figsize=(9, 5))
metricas_box = {
    "Acurácia":       cv_results["test_accuracy"],
    "Ac. Balanceada": cv_results["test_balanced_accuracy"],
    "Precisão":       cv_results["test_precision"],
    "Recall":         cv_results["test_recall"],
    "F1-Score":       cv_results["test_f1"],
    "AUC-ROC":        cv_results["test_roc_auc"],
}
bp = ax_box.boxplot(
    metricas_box.values(), labels=metricas_box.keys(),
    patch_artist=True,
    medianprops=dict(color="black", lw=2),
    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
)
cores_box = [COR_ACC, COR_ACC, COR_H, COR_S, COR_AUC, COR_AUC]
for patch, color in zip(bp["boxes"], cores_box):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for i, (k, v) in enumerate(metricas_box.items(), start=1):
    ax_box.text(i, v.mean(), f"{v.mean():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax_box.set_ylabel("Score", fontsize=10)
ax_box.set_ylim([0, 1.1])
ax_box.set_title("Distribuição das Métricas — 10-Fold CV\nÁrvore de Decisão | PNS 2019 (sem Idade)", fontsize=12)
ax_box.grid(axis="y", alpha=0.3); ax_box.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig_box.savefig("e9_metricas_fold.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5.6 Visualização da árvore (até profundidade 3) ──────────
fig_tree, ax_tree = plt.subplots(figsize=(22, 9))
# Usa nomes legíveis na visualização
feature_names_display = [descricoes.get(f, f) for f in feature_names]
plot_tree(
    clf, max_depth=3,
    feature_names=feature_names_display,
    class_names=["Saudável", "Hipertenso"],
    filled=True, rounded=True, fontsize=8,
    impurity=True, proportion=False, ax=ax_tree,
)
ax_tree.set_title(
    "Árvore de Decisão — PNS 2019 (max_depth=5, exibindo até nível 3, sem Idade)\n"
    "Verde → Saudável  |  Laranja → Hipertenso",
    fontsize=13,
)
plt.tight_layout()
fig_tree.savefig("e9_arvore_decisao.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 5.7 AUC × profundidade (análise de overfitting) ──────────
fig_depth, ax_depth = plt.subplots(figsize=(8, 5))
depths = range(1, 16)
tr_aucs, va_aucs = [], []
for d in depths:
    clf_d = DecisionTreeClassifier(
        criterion="gini", max_depth=d, min_samples_leaf=20,
        class_weight="balanced", random_state=42,
    )
    res = cross_validate(clf_d, X, y, cv=cv,
                         scoring={"auc": "roc_auc"},
                         return_train_score=True, n_jobs=-1)
    tr_aucs.append(res["train_auc"].mean())
    va_aucs.append(res["test_auc"].mean())

ax_depth.plot(depths, tr_aucs, "o-", color=COR_H, lw=2, label="Treino AUC")
ax_depth.plot(depths, va_aucs, "s--", color=COR_S, lw=2, label="Validação AUC (CV)")
ax_depth.axvline(5, color="gray", ls=":", lw=1.5, label="Profundidade escolhida (5)")
ax_depth.set_xlabel("Profundidade Máxima", fontsize=10)
ax_depth.set_ylabel("AUC-ROC", fontsize=10)
ax_depth.set_title("AUC × Profundidade da Árvore\nAnálise de Overfitting | PNS 2019 (sem Idade)", fontsize=12)
ax_depth.legend(fontsize=9)
ax_depth.grid(alpha=0.3); ax_depth.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig_depth.savefig("e9_analise_profundidade.png", dpi=150, bbox_inches="tight")
plt.close()

# =============================================================
# 6. RESUMO FINAL
# =============================================================
print("\n" + "="*65)
print("  RESUMO FINAL — ETAPA 9 (sem a variável Idade)")
print("="*65)
print(f"  Modelo         : DecisionTreeClassifier (max_depth=5, min_samples_leaf=20)")
print(f"  Critério       : Gini | Pesos: balanced")
print(f"  Validação      : Stratified K-Fold (k=10)")
print(f"  Features usadas: {len(feature_names)}")
print(f"  Variável alvo  : Q00201 (hipertensão arterial — removida das features)")
print(f"  Variável removida a pedido: C008 (Idade)")
print(f"  Variáveis diretas excluídas indevidamente antes: NENHUMA")
print(f"  (Q03001=diabetes, Q060=colesterol, Q06306=cardiopatia são comorbidades")
print(f"   de exclusão, não proxy de hipertensão, e não constam no Excel)")
print()
for label, (val_key, _) in metricas_cv.items():
    arr = cv_results[f"test_{val_key}"]
    print(f"  {label:<25}: {arr.mean():.4f} ± {arr.std():.4f}")
print()
print("  Gráficos gerados:")
for g in [
    "e9_curva_roc.png",
    "e9_matriz_confusao.png",
    "e9_importancia_variaveis.png",
    "e9_curva_aprendizado.png",
    "e9_metricas_fold.png",
    "e9_arvore_decisao.png",
    "e9_analise_profundidade.png",
]:
    print(f"    → {g}")
print("="*65)
# =============================================================
# 7. SALVAR RESULTADOS (JSON / CSV / TXT)
# =============================================================
try:
    resultados = {}

    # Métricas CV (média, desvio e folds)
    resultados["cv_metrics"] = {}
    for label, (val_key, _) in metricas_cv.items():
        arr = cv_results[f"test_{val_key}"]
        resultados["cv_metrics"][val_key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "folds": [float(x) for x in arr.tolist()],
        }

    # AUC média das folds (já calculada anteriormente)
    resultados["cv_mean_auc"] = {"mean": float(mean_auc), "std": float(std_auc), "folds": [float(x) for x in aucs_fold]}

    # Relatório final e matriz de confusão
    resultados["classification_report"] = classification_report(y, y_pred, target_names=["Saudável", "Hipertenso"], output_dict=True)
    resultados["confusion_matrix"] = cm.tolist()

    # Importâncias das features
    imp_ser = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    resultados["feature_importances"] = [{"feature": k, "importance": float(v)} for k, v in imp_ser.items()]

    # Análise por profundidade
    resultados["depth_analysis"] = {"depths": list(depths), "train_aucs": [float(x) for x in tr_aucs], "val_aucs": [float(x) for x in va_aucs]}

    # Parâmetros do modelo
    resultados["model_params"] = clf.get_params()

    # Nota sobre a remoção da variável Idade
    resultados["nota"] = "Variável C008 (Idade) removida das features a pedido do usuário."

    # Salva JSON completo
    with open("e9_resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # Salva importâncias em CSV
    imp_ser.to_csv("e9_importancias.csv", header=["importance"] )

    # Salva métricas por fold em CSV
    df_cv = pd.DataFrame({k: cv_results[f"test_{k}"] for k in scoring.keys()})
    df_cv.to_csv("e9_cv_folds.csv", index=False)

    # Salva resumo legível em TXT
    with open("e9_resultados.txt", "w", encoding="utf-8") as fh:
        fh.write("RESUMO FINAL — ETAPA 9 (sem a variável Idade)\n")
        fh.write("Modelo: DecisionTreeClassifier (max_depth=5, min_samples_leaf=20)\n")
        fh.write(f"Features usadas: {len(feature_names)}\n")
        fh.write("Variável removida a pedido: C008 (Idade)\n")
        fh.write("\nMétricas (CV - média ± std):\n")
        for k, v in resultados["cv_metrics"].items():
            fh.write(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}\n")
        fh.write("\nMatriz de Confusão (conjunto completo):\n")
        fh.write(np.array2string(cm))
        fh.write("\n\nTop 20 importâncias:\n")
        for d in resultados["feature_importances"][:20]:
            fh.write(f"  {d['feature']}: {d['importance']:.6f}\n")

    print("\n  Resultados salvos em: e9_resultados.json, e9_importancias.csv, e9_cv_folds.csv, e9_resultados.txt")
except Exception as e:
    print(f"Erro ao salvar resultados: {e}")