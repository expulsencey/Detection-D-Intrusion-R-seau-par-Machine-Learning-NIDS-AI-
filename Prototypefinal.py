import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io

# ═══════════════════════════════════════════════
# CONFIG & CONSTANTES
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="NIDS — Détection d'intrusions réseau",
    layout="wide",
    initial_sidebar_state="collapsed"
)

matplotlib.rcParams.update({
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#d1dce8', 'axes.labelcolor': '#6b80a0',
    'xtick.color': '#6b80a0', 'ytick.color': '#6b80a0',
    'text.color': '#1e3148', 'grid.color': '#e8eef5', 'grid.alpha': 1.0,
})

# Colonnes NSL-KDD complètes (43 colonnes, avec last_flag)
COLUMNS_43 = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
]

# Colonnes NSL-KDD sans last_flag (42 colonnes)
COLUMNS_42 = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack"
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
ACCEPTED = ["csv", "txt", "xlsx", "xls"]

DESCRIPTIONS = {
    "duration": "Durée de la connexion (secondes)", "src_bytes": "Octets source",
    "dst_bytes": "Octets destination", "serror_rate": "Taux erreurs SYN (0–1)",
    "rerror_rate": "Taux erreurs REJ (0–1)", "same_srv_rate": "Taux même service",
    "diff_srv_rate": "Taux services différents", "dst_host_srv_count": "Nb connexions service (hôte dest.)",
    "dst_host_same_srv_rate": "Taux même service (hôte dest.)", "dst_host_diff_srv_rate": "Taux services diff. (hôte dest.)",
    "dst_host_same_src_port_rate": "Taux même port source (hôte dest.)", "count": "Nb connexions même hôte (2s)",
    "srv_count": "Nb connexions même service (2s)", "srv_serror_rate": "Taux SYN (même service)",
    "srv_rerror_rate": "Taux REJ (même service)", "srv_diff_host_rate": "Taux hôtes diff. (même service)",
    "dst_host_count": "Nb connexions hôte dest.", "dst_host_serror_rate": "Taux SYN (hôte dest.)",
    "dst_host_rerror_rate": "Taux REJ (hôte dest.)", "dst_host_srv_serror_rate": "Taux SYN (service+hôte)",
    "dst_host_srv_rerror_rate": "Taux REJ (service+hôte)", "hot": "Indicateurs 'hot'",
    "logged_in": "Connexion réussie (0/1)", "num_compromised": "Conditions compromises",
    "protocol_type": "Protocole encodé", "service": "Service encodé", "flag": "Flag encodé",
}

# ═══════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #f0f4f8;
    --surface:    #ffffff;
    --surface2:   #e8eef5;
    --border:     #d1dce8;
    --teal:       #0ea5b0;
    --teal-dark:  #0b8a94;
    --teal-soft:  #e0f7f9;
    --navy:       #1b2f4e;
    --text:       #1e3148;
    --muted:      #6b80a0;
    --red:        #e05a6a;
    --red-soft:   #fdeef0;
    --green:      #14a87a;
    --green-soft: #e4f8f2;
    --amber-soft: #fef9ec;
    --amber:      #d9950a;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}
.stApp { background-color: var(--bg) !important; }
header[data-testid="stHeader"] { display: none; }

/* ── TOP BAR ── */
.topbar {
    background: linear-gradient(135deg, #1b2f4e 0%, #0f4a52 60%, #0ea5b0 100%);
    border-radius: 16px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.topbar::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:rgba(14,165,176,.22);
}
.topbar::after {
    content:''; position:absolute; bottom:-40px; left:35%;
    width:130px; height:130px; border-radius:50%;
    background:rgba(255,255,255,.06);
}
.topbar-tag {
    font-family:'JetBrains Mono',monospace; font-size:.65rem;
    letter-spacing:.22em; text-transform:uppercase;
    color:#5de0e9; margin-bottom:.45rem;
}
.topbar h1 {
    font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
    color:#fff; margin:0 0 .35rem 0; letter-spacing:-.01em;
}
.topbar p { font-size:.88rem; color:rgba(255,255,255,.55); margin:0; font-weight:300; }

/* ── STEP BLOCK ── */
.step-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(27,47,78,.06);
}
.step-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--surface2);
}
.step-num {
    font-family:'JetBrains Mono',monospace;
    font-size:.7rem; letter-spacing:.15em; font-weight:600;
    background: var(--teal); color: #fff;
    padding: .28rem .65rem; border-radius: 6px;
    text-transform: uppercase;
}
.step-num.done { background: var(--green); }
.step-num.warn { background: var(--amber); }
.step-title {
    font-family:'Syne',sans-serif; font-size:1.05rem;
    font-weight:700; color:var(--navy);
}
.step-desc { font-size:.8rem; color:var(--muted); margin-top:.1rem; }

/* ── BANNER ── */
.banner {
    border-radius:9px; padding:.65rem 1rem; font-size:.83rem;
    margin-bottom:.9rem; display:flex; align-items:center; gap:.6rem;
}
.banner-ok   { background:var(--green-soft); color:#0e7a57; border:1px solid #a3e8d4; }
.banner-err  { background:var(--red-soft);   color:#c0253a; border:1px solid #f5b8c0; }
.banner-info { background:var(--teal-soft);  color:#0a7780; border:1px solid #9ddde3; }
.banner-warn { background:var(--amber-soft); color:var(--amber); border:1px solid #f5dea0; }
.b-badge {
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    font-weight:600; letter-spacing:.1em; text-transform:uppercase;
    padding:.14rem .45rem; border-radius:4px; color:#fff; white-space:nowrap;
}
.banner-ok   .b-badge { background:#14a87a; }
.banner-err  .b-badge { background:#e05a6a; }
.banner-info .b-badge { background:#0ea5b0; }
.banner-warn .b-badge { background:var(--amber); }

/* ── METRICS ── */
.metric-row { display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem; margin:1rem 0; }
.mtile {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.2rem;
}
.mtile .mlabel {
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:.4rem;
}
.mtile .mvalue {
    font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800;
    color:var(--navy); line-height:1;
}
.mtile .mvalue.teal { color:var(--teal-dark); }
.mtile .mvalue.red  { color:var(--red); }

/* ── RESULT CARD ── */
.result-card { border-radius:12px; padding:1.6rem 2rem; margin:1rem 0; text-align:center; }
.result-card.ok  { background:var(--green-soft); border:1.5px solid #a3e8d4; }
.result-card.err { background:var(--red-soft);   border:1.5px solid #f5b8c0; }
.result-card .r-label {
    font-family:'JetBrains Mono',monospace; font-size:.65rem;
    letter-spacing:.18em; text-transform:uppercase; margin-bottom:.4rem;
}
.result-card.ok  .r-label { color:#0b6b4e; }
.result-card.err .r-label { color:#9e1f30; }
.result-card .r-title { font-family:'Syne',sans-serif; font-size:1.45rem; font-weight:800; }
.result-card.ok  .r-title { color:#0e7a57; }
.result-card.err .r-title { color:#c0253a; }

.proba-row { display:flex; gap:.9rem; margin-top:.8rem; }
.proba-item {
    flex:1; background:var(--surface); border:1px solid var(--border);
    border-radius:9px; padding:.8rem 1rem; text-align:center;
}
.proba-item .p-label {
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-bottom:.25rem;
}
.proba-item .p-val { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:var(--navy); }

/* ── CARD TITLE ── */
.ctitle {
    font-family:'Syne',sans-serif; font-size:.95rem; font-weight:700;
    color:var(--navy); margin-bottom:.9rem;
    display:flex; align-items:center; gap:.5rem;
}
.ctitle .dot { width:7px; height:7px; border-radius:50%; background:var(--teal); display:inline-block; }

/* ── INPUTS ── */
[data-testid="stFileUploader"] {
    background:var(--surface2) !important;
    border:1.5px dashed var(--border) !important;
    border-radius:9px !important;
}
[data-testid="stSelectbox"] > div > div {
    background:var(--surface2) !important; border-color:var(--border) !important;
    border-radius:8px !important; color:var(--text) !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1b2f4e,#0b7a84) !important;
    border:none !important; color:#fff !important;
    font-family:'Syne',sans-serif !important; font-size:.82rem !important;
    font-weight:700 !important; letter-spacing:.04em !important;
    border-radius:8px !important; padding:.5rem 1.6rem !important;
    box-shadow:0 2px 10px rgba(14,165,176,.3) !important;
    transition:opacity .15s, transform .1s !important;
}
.stButton > button:hover { opacity:.85 !important; transform:translateY(-1px) !important; }
[data-testid="stNumberInput"] input {
    background:var(--surface2) !important; border-color:var(--border) !important;
    border-radius:7px !important; color:var(--text) !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.8rem !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family:'JetBrains Mono',monospace !important;
    font-size:.73rem !important; color:var(--muted) !important;
}
[data-testid="stExpander"] {
    background:var(--surface2) !important; border:1px solid var(--border) !important;
    border-radius:9px !important;
}
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:9px !important; }

hr { border-color:var(--border) !important; margin:0 !important; }

.nids-footer {
    text-align:center; font-family:'JetBrains Mono',monospace;
    font-size:.68rem; color:var(--muted); letter-spacing:.08em;
    margin-top:2rem; padding:1rem 0; border-top:1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════

def load_file(f):
    """Charge un fichier CSV, TXT, XLSX ou XLS en DataFrame sans en-tête."""
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
            return pd.read_excel(f, header=None, engine=engine)
        except ImportError as e:
            st.error(f"Librairie manquante : {e}. Installez-la avec : pip install openpyxl xlrd")
            st.stop()
    elif name.endswith(".txt"):
        content = f.read()
        for sep in [",", "\t", ";", " "]:
            try:
                return pd.read_csv(io.BytesIO(content), header=None, sep=sep)
            except Exception:
                continue
        st.error("Impossible de parser le fichier .txt. Vérifiez le séparateur.")
        st.stop()
    else:
        return pd.read_csv(f, header=None)


def assign_columns(df):
    """
    Assigne les noms de colonnes NSL-KDD selon le nombre de colonnes du DataFrame.
    Accepte 42 colonnes (sans last_flag) ou 43 colonnes (avec last_flag).
    Retourne (df_avec_colonnes, message_info) ou lève une ValueError.
    """
    n = df.shape[1]
    if n == 43:
        df.columns = COLUMNS_43
        return df, f"{n} colonnes détectées (format NSL-KDD complet avec last_flag)."
    elif n == 42:
        df.columns = COLUMNS_42
        return df, f"{n} colonnes détectées (format NSL-KDD sans last_flag)."
    else:
        raise ValueError(
            f"Format non reconnu : {n} colonnes trouvées. "
            f"Le fichier doit avoir 42 ou 43 colonnes (format NSL-KDD)."
        )


# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════

st.markdown("""
<div class="topbar">
    <div class="topbar-tag">Network Intrusion Detection System</div>
    <h1>Analyse du trafic réseau</h1>
    <p>Classification automatique des connexions — Dataset NSL-KDD &nbsp;·&nbsp; Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# BLOC 1 — CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════

st.markdown("""
<div class="step-block">
  <div class="step-header">
    <div><div class="step-num">01</div></div>
    <div>
      <div class="step-title">Chargement des données</div>
      <div class="step-desc">Formats acceptés : CSV, TXT, XLSX, XLS &nbsp;·&nbsp; 42 ou 43 colonnes NSL-KDD</div>
    </div>
  </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    train_file = st.file_uploader("Fichier d'entraînement", type=ACCEPTED, key="train")
with col2:
    test_file = st.file_uploader("Fichier de test", type=ACCEPTED, key="test")

train, test = None, None
data_ready = False

if train_file and test_file:
    try:
        train_raw = load_file(train_file)
        test_raw  = load_file(test_file)

        train, msg_train = assign_columns(train_raw)
        test,  msg_test  = assign_columns(test_raw)

        # Vérification : les deux fichiers doivent avoir la même structure
        if train.shape[1] != test.shape[1]:
            st.markdown(
                f'<div class="banner banner-err"><span class="b-badge">Erreur</span> '
                f'Les fichiers n\'ont pas le même nombre de colonnes '
                f'(train={train.shape[1]}, test={test.shape[1]}). '
                f'Utilisez deux fichiers du même format.</div>',
                unsafe_allow_html=True
            )
        else:
            data_ready = True
            st.markdown(
                f'<div class="banner banner-ok"><span class="b-badge">Chargé</span> '
                f'Train : <strong>{train.shape[0]:,} lignes</strong> &nbsp;·&nbsp; '
                f'Test : <strong>{test.shape[0]:,} lignes</strong> &nbsp;·&nbsp; '
                f'{msg_train}</div>',
                unsafe_allow_html=True
            )
            with st.expander("Aperçu — 10 premières lignes (train)"):
                st.dataframe(train.head(10), use_container_width=True)

    except ValueError as e:
        st.markdown(
            f'<div class="banner banner-err"><span class="b-badge">Erreur</span> {e}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.markdown(
            f'<div class="banner banner-err"><span class="b-badge">Erreur</span> {e}</div>',
            unsafe_allow_html=True
        )
else:
    st.markdown(
        '<div class="banner banner-info"><span class="b-badge">Requis</span> '
        'Chargez les deux fichiers pour continuer.</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# BLOC 2 — PRÉTRAITEMENT
# ═══════════════════════════════════════════════

preproc_done = False
top_features = means = stds = None
X_train = y_train = X_test = y_test = None
feat_imp = None

st.markdown("""
<div class="step-block">
  <div class="step-header">
    <div><div class="step-num{}">02</div></div>
    <div>
      <div class="step-title">Prétraitement automatique</div>
      <div class="step-desc">Encodage · Normalisation · Sélection des meilleures features</div>
    </div>
  </div>
""".format("" if data_ready else " warn"), unsafe_allow_html=True)

if not data_ready:
    st.markdown(
        '<div class="banner banner-warn"><span class="b-badge">En attente</span> '
        'Chargez les données à l\'étape 01 pour débloquer cette section.</div>',
        unsafe_allow_html=True
    )
else:
    with st.spinner("Traitement en cours…"):
        le = LabelEncoder()
        for col in CATEGORICAL_COLS:
            train[col] = le.fit_transform(train[col])
            test[col]  = le.transform(test[col])

        train['attack'] = np.where(train['attack'] == 'normal', 0, 1)
        test['attack']  = np.where(test['attack']  == 'normal', 0, 1)

        # Supprimer num_outbound_cmds (toujours 0 dans NSL-KDD) si présente
        for col_drop in ['num_outbound_cmds', 'last_flag']:
            if col_drop in train.columns:
                train = train.drop(columns=[col_drop])
            if col_drop in test.columns:
                test = test.drop(columns=[col_drop])

        numerical_cols = [c for c in train.columns if c not in ['attack'] + CATEGORICAL_COLS]
        means = train[numerical_cols].mean()
        stds  = train[numerical_cols].std()
        train[numerical_cols] = (train[numerical_cols] - means) / stds
        test[numerical_cols]  = (test[numerical_cols]  - means) / stds

        X_all = train.drop('attack', axis=1)
        y_all = train['attack']
        selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
        selector.fit(X_all, y_all)
        feat_imp     = pd.Series(selector.feature_importances_, index=X_all.columns)
        top_features = feat_imp.nlargest(11).index.tolist()

        X_train = train[top_features].values
        y_train = train['attack'].values
        X_test  = test[top_features].values
        y_test  = test['attack'].values
        preproc_done = True

    st.markdown(
        '<div class="banner banner-ok"><span class="b-badge">OK</span> '
        'Prétraitement terminé — 11 features retenues.</div>',
        unsafe_allow_html=True
    )

    with st.expander("Importance des features (top 11)"):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        vals = feat_imp.nlargest(11).sort_values()
        bar_colors = ['#0ea5b0' if i >= len(vals) - 3 else '#b8dde2' for i in range(len(vals))]
        vals.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='none')
        ax.set_xlabel("Score d'importance", fontsize=8)
        ax.set_title("Top 11 features — ExtraTreesClassifier", fontsize=9, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# BLOC 3 — ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════

st.markdown("""
<div class="step-block">
  <div class="step-header">
    <div><div class="step-num{}">03</div></div>
    <div>
      <div class="step-title">Entraînement du modèle</div>
      <div class="step-desc">Choisissez un algorithme puis lancez l'entraînement</div>
    </div>
  </div>
""".format("" if preproc_done else " warn"), unsafe_allow_html=True)

if not preproc_done:
    st.markdown(
        '<div class="banner banner-warn"><span class="b-badge">En attente</span> '
        'Le prétraitement doit être complété pour débloquer cette section.</div>',
        unsafe_allow_html=True
    )
else:
    col_sel, col_btn = st.columns([3, 1], gap="medium")
    with col_sel:
        model_choice = st.selectbox(
            "Algorithme",
            ["Random Forest", "Naive Bayes", "SVM", "KNN", "Régression Logistique"]
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        train_btn = st.button("Lancer l'entraînement")

    if train_btn:
        with st.spinner(f"Entraînement — {model_choice}…"):
            models_map = {
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Naive Bayes":         GaussianNB(),
                "SVM":                 SVC(),
                "KNN":                 KNeighborsClassifier(n_neighbors=5),
                "Régression Logistique": LogisticRegression(max_iter=1000),
            }
            model = models_map[model_choice]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            st.session_state['model']        = model
            st.session_state['top_features'] = top_features
            st.session_state['means']        = means
            st.session_state['stds']         = stds
            st.session_state['model_name']   = model_choice

        st.markdown(
            f'<div class="banner banner-ok"><span class="b-badge">Succès</span> '
            f'Modèle <strong>{model_choice}</strong> entraîné.</div>',
            unsafe_allow_html=True
        )

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class="metric-row">
            <div class="mtile"><div class="mlabel">Accuracy</div><div class="mvalue teal">{acc:.4f}</div></div>
            <div class="mtile"><div class="mlabel">Vrais positifs</div><div class="mvalue">{tp:,}</div></div>
            <div class="mtile"><div class="mlabel">Faux négatifs</div><div class="mvalue red">{fn:,}</div></div>
        </div>""", unsafe_allow_html=True)

        col_cm, col_rep = st.columns(2, gap="medium")
        with col_cm:
            st.markdown('<div class="ctitle"><span class="dot"></span>Matrice de confusion</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                        cmap=sns.light_palette("#0ea5b0", as_cmap=True),
                        linewidths=1.5, linecolor='#f0f4f8',
                        xticklabels=["Normal", "Attaque"],
                        yticklabels=["Normal", "Attaque"],
                        annot_kws={"size": 12, "weight": "bold", "color": "#1b2f4e"})
            ax.set_xlabel("Prédit", fontsize=8, labelpad=6)
            ax.set_ylabel("Réel", fontsize=8, labelpad=6)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_rep:
            st.markdown('<div class="ctitle"><span class="dot"></span>Rapport de classification</div>', unsafe_allow_html=True)
            report = classification_report(
                y_test, y_pred,
                target_names=["Normal", "Attaque"],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    elif 'model' in st.session_state:
        st.markdown(
            f'<div class="banner banner-info"><span class="b-badge">Actif</span> '
            f'Modèle chargé : <strong>{st.session_state["model_name"]}</strong></div>',
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# BLOC 4 — PRÉDICTION D'UNE CONNEXION
# ═══════════════════════════════════════════════

model_ready = 'model' in st.session_state

st.markdown("""
<div class="step-block">
  <div class="step-header">
    <div><div class="step-num{}">04</div></div>
    <div>
      <div class="step-title">Analyser une connexion</div>
      <div class="step-desc">Renseignez les paramètres de la connexion à classifier</div>
    </div>
  </div>
""".format("" if model_ready else " warn"), unsafe_allow_html=True)

if not model_ready:
    st.markdown(
        '<div class="banner banner-warn"><span class="b-badge">En attente</span> '
        'Entraînez un modèle à l\'étape 03 pour débloquer cette section.</div>',
        unsafe_allow_html=True
    )
else:
    top_f  = st.session_state['top_features']
    means_ = st.session_state['means']
    stds_  = st.session_state['stds']

    with st.form("pred_form"):
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(top_f):
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    label=feat,
                    help=DESCRIPTIONS.get(feat, feat),
                    value=0.0, format="%.4f",
                    key=f"p_{feat}"
                )
        submitted = st.form_submit_button("Analyser la connexion")

    if submitted:
        raw = pd.Series(input_vals)
        norm_vals = {}
        for feat in top_f:
            if feat in means_.index and stds_[feat] != 0:
                norm_vals[feat] = (raw[feat] - means_[feat]) / stds_[feat]
            else:
                norm_vals[feat] = raw[feat]

        X_input    = np.array([list(norm_vals.values())])
        prediction = st.session_state['model'].predict(X_input)[0]

        if prediction == 1:
            st.markdown("""<div class="result-card err">
                <div class="r-label">Résultat de l'analyse</div>
                <div class="r-title">Attaque détectée</div>
                <div style="font-size:.83rem;color:#9e1f30;margin-top:.4rem;font-weight:300;">
                    Cette connexion est classifiée comme malveillante.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-card ok">
                <div class="r-label">Résultat de l'analyse</div>
                <div class="r-title">Trafic normal</div>
                <div style="font-size:.83rem;color:#0b6b4e;margin-top:.4rem;font-weight:300;">
                    Aucune menace détectée sur cette connexion.
                </div>
            </div>""", unsafe_allow_html=True)

        if hasattr(st.session_state['model'], "predict_proba"):
            proba = st.session_state['model'].predict_proba(X_input)[0]
            st.markdown(f"""<div class="proba-row">
                <div class="proba-item">
                    <div class="p-label">P — Normal</div>
                    <div class="p-val">{proba[0]:.4f}</div>
                </div>
                <div class="proba-item">
                    <div class="p-label">P — Attaque</div>
                    <div class="p-val">{proba[1]:.4f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════

st.markdown(
    '<div class="nids-footer">NIDS v2.1 &nbsp;·&nbsp; NSL-KDD Dataset &nbsp;·&nbsp; RF · NB · SVM · KNN · LR</div>',
    unsafe_allow_html=True
)
