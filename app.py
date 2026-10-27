import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

DEFAULT_FEATURES = [
    "gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s",
    "degree_p", "degree_t", "workex", "etest_p", "specialisation", "mba_p",
]
FEATURES = list(getattr(model, "feature_names_in_", DEFAULT_FEATURES))

# ----------------------------------------------------------------------------
# Feature definitions
# NOTE: the numeric codes below assume standard alphabetical LabelEncoding.
# If you encoded differently during training, change the numbers here.
# ----------------------------------------------------------------------------
NUMERIC = {
    "ssc_p": ("10th Grade %", "Secondary school (SSC) percentage"),
    "hsc_p": ("12th Grade %", "Higher secondary (HSC) percentage"),
    "degree_p": ("Degree %", "Undergraduate degree percentage"),
    "etest_p": ("Employability Test %", "Score in the employability / aptitude test"),
    "mba_p": ("MBA %", "Current MBA percentage"),
}

CATEGORICAL = {
    "gender": ("Gender", [("Female", 0), ("Male", 1)]),
    "ssc_b": ("10th Board", [("Central", 0), ("Others", 1)]),
    "hsc_b": ("12th Board", [("Central", 0), ("Others", 1)]),
    "hsc_s": ("12th Stream", [("Arts", 0), ("Commerce", 1), ("Science", 2)]),
    "degree_t": ("Degree Type", [("Comm & Mgmt", 0), ("Others", 1), ("Sci & Tech", 2)]),
    "workex": ("Work Experience", [("No", 0), ("Yes", 1)]),
    "specialisation": ("MBA Specialisation", [("Mkt & Fin", 0), ("Mkt & HR", 1)]),
}

SECTIONS = [
    {"title": "Personal", "icon": "👤", "fields": ["gender"]},
    {"title": "School Education", "icon": "🏫", "fields": ["ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s"]},
    {"title": "Degree", "icon": "🎓", "fields": ["degree_p", "degree_t"]},
    {"title": "MBA & Experience", "icon": "💼", "fields": ["etest_p", "mba_p", "specialisation", "workex"]},
]

ALL_LABELS = {k: v[0] for k, v in NUMERIC.items()}
ALL_LABELS.update({k: v[0] for k, v in CATEGORICAL.items()})

META = {
    "numeric": [{"key": k, "label": v[0], "hint": v[1]} for k, v in NUMERIC.items()],
    "categorical": {
        k: {"label": v[0], "options": [{"label": o, "value": val} for o, val in v[1]]}
        for k, v in CATEGORICAL.items()
    },
    "sections": SECTIONS,
}


# ----------------------------------------------------------------------------
# Prediction helpers (pandas)
# ----------------------------------------------------------------------------
def to_frame(values):
    """Build a one-row DataFrame with the exact column order the model was trained on."""
    return pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)


def predict_frame(df):
    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    classes = list(model.classes_)
    p_placed = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
    return pred, p_placed


def validate(data):
    clean = {}
    for f in FEATURES:
        if f not in data:
            raise ValueError(f"Missing field: {ALL_LABELS.get(f, f)}")
        try:
            val = float(data[f])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for {ALL_LABELS.get(f, f)}")
        if not np.isfinite(val):
            raise ValueError(f"Invalid value for {ALL_LABELS.get(f, f)}")
        if f in NUMERIC and not (0 <= val <= 100):
            raise ValueError(f"{ALL_LABELS[f]} must be between 0 and 100")
        if f in CATEGORICAL and int(val) not in [v for _, v in CATEGORICAL[f][1]]:
            raise ValueError(f"Invalid option for {ALL_LABELS[f]}")
        clean[f] = val
    return clean


def describe_step(idx, thr, went_left, value):
    name = FEATURES[idx]
    if name in CATEGORICAL:
        label, options = CATEGORICAL[name]
        opt = next((o for o, v in options if v == int(round(value))), str(value))
        return f"Your {label.lower()} is “{opt}”."
    label = NUMERIC[name][0]
    rel = "at or below" if went_left else "above"
    return f"Your {label} ({value:.0f}%) is {rel} the model’s checkpoint of {thr:.1f}%."


def explain_path(df):
    tree = model.tree_
    steps = []
    for nid in model.decision_path(df).indices:
        if tree.children_left[nid] == tree.children_right[nid]:
            continue  # leaf node
        idx = int(tree.feature[nid])
        thr = float(tree.threshold[nid])
        value = float(df.iloc[0, idx])
        steps.append(describe_step(idx, thr, value <= thr, value))
    return steps


def strengths_and_gaps(values):
    strengths, gaps = [], []
    for k, (label, _) in NUMERIC.items():
        v = values[k]
        if v >= 70:
            strengths.append(f"Your {label} of {v:.0f}% is strong.")
        elif v < 60:
            gaps.append(f"Your {label} of {v:.0f}% is on the lower side; raising it would help.")
    if int(values["workex"]) == 1:
        strengths.append("You have prior work experience, which recruiters value.")
    else:
        gaps.append("You have no work experience yet. Internships or projects can help.")
    return strengths, gaps


def what_if(values, base_pred):
    """Find the smallest change in a single score that flips the prediction."""
    out = []
    target = 1 - base_pred
    for k, (label, _) in NUMERIC.items():
        for delta in range(5, 45, 5):
            new_val = values[k] + delta if base_pred == 0 else values[k] - delta
            if new_val > 100 or new_val < 0:
                break
            test = dict(values)
            test[k] = new_val
            p, _ = predict_frame(to_frame(test))
            if p == target:
                if base_pred == 0:
                    msg = (f"If your {label} rose from {values[k]:.0f}% to {new_val:.0f}%, "
                           f"the model would predict you are likely to be placed.")
                else:
                    msg = (f"Your result is sensitive to {label}: if it dropped from "
                           f"{values[k]:.0f}% to {new_val:.0f}%, the prediction would flip to not placed.")
                out.append((delta, msg))
                break
    if base_pred == 0 and int(values["workex"]) == 0:
        test = dict(values)
        test["workex"] = 1
        p, _ = predict_frame(to_frame(test))
        if p == 1:
            out.append((0, "Gaining work experience alone would change the prediction to likely placed."))
    out.sort(key=lambda t: t[0])
    return [m for _, m in out[:4]]


def importance():
    imps = getattr(model, "feature_importances_", None)
    if imps is None:
        return []
    s = pd.Series(imps, index=FEATURES).sort_values(ascending=False)
    s = s[s > 0.005].head(6)
    return [{"label": ALL_LABELS.get(name, name), "value": float(v)} for name, v in s.items()]


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return Response(HTML.replace("__META__", json.dumps(META)), mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(status="ok", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = validate(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400

    df = to_frame(values)
    pred, p = predict_frame(df)
    conf = p if pred == 1 else 1 - p
    strengths, gaps = strengths_and_gaps(values)

    if pred == 1:
        headline = "Likely to be placed 🎉"
        summary = (f"Based on the profile you entered, the model predicts that you are likely to get "
                   f"placed. It is {conf * 100:.0f}% confident in this result.")
    else:
        headline = "Placement is unlikely right now"
        summary = (f"Based on the profile you entered, the model predicts that you may not get placed "
                   f"at the moment. It is {conf * 100:.0f}% confident. The suggestions below show what "
                   f"could improve your chances.")

    return jsonify(
        prediction=pred,
        probability=p,
        confidence=conf,
        headline=headline,
        summary=summary,
        reasons=explain_path(df),
        strengths=strengths,
        gaps=gaps,
        whatif=what_if(values, pred),
        importance=importance(),
    )


# ----------------------------------------------------------------------------
# Frontend (single-file)
# ----------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Placement Predictor · AI Career Insights</title>
<script>
  (function () {
    var t = localStorage.getItem("theme");
    if (!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
  })();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#f4f6fb; --card:#ffffff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0;
    --accent:#6366f1; --accent2:#8b5cf6; --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
    --soft:#eef2ff; --shadow:0 10px 30px -12px rgba(15,23,42,.18);
    color-scheme: light;
  }
  [data-theme="dark"] {
    --bg:#0a0f1f; --card:#121932; --text:#e8ebf7; --muted:#94a3c4; --border:#222c4b;
    --soft:#1a2242; --shadow:0 10px 30px -12px rgba(0,0,0,.6);
    color-scheme: dark;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html { scroll-behavior:smooth; }
  body {
    font-family:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--text); min-height:100vh; line-height:1.5;
    transition:background .3s,color .3s;
  }
  body::before, body::after {
    content:""; position:fixed; z-index:-1; border-radius:50%; filter:blur(90px); opacity:.35; pointer-events:none;
  }
  body::before { width:420px; height:420px; background:var(--accent); top:-120px; left:-100px; }
  body::after { width:380px; height:380px; background:var(--accent2); bottom:-120px; right:-80px; }
  [data-theme="dark"] body::before, [data-theme="dark"] body::after { opacity:.22; }

  header {
    position:sticky; top:0; z-index:20; backdrop-filter:blur(14px);
    background:color-mix(in srgb, var(--bg) 75%, transparent);
    border-bottom:1px solid var(--border);
  }
  .nav { max-width:1200px; margin:auto; padding:14px 20px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .brand { display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:-.02em; font-size:1.1rem; }
  .logo { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; color:#fff; font-size:1.2rem;
          background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 8px 20px -6px var(--accent); }
  .brand small { display:block; font-weight:500; color:var(--muted); font-size:.72rem; letter-spacing:0; }
  .icon-btn { border:1px solid var(--border); background:var(--card); color:var(--text); width:42px; height:42px; border-radius:12px;
              cursor:pointer; font-size:1.1rem; transition:transform .2s, border-color .2s; }
  .icon-btn:hover { transform:translateY(-2px); border-color:var(--accent); }

  main { max-width:1200px; margin:auto; padding:28px 20px 60px; }
  .hero { margin-bottom:26px; }
  .hero h1 { font-size:clamp(1.7rem,4vw,2.5rem); font-weight:800; letter-spacing:-.03em; line-height:1.15; }
  .hero h1 span { background:linear-gradient(135deg,var(--accent),var(--accent2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .hero p { color:var(--muted); margin-top:8px; max-width:640px; }

  .grid { display:grid; grid-template-columns:1.25fr 1fr; gap:22px; align-items:start; }
  @media (max-width:960px) { .grid { grid-template-columns:1fr; } }

  .card { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:22px; box-shadow:var(--shadow); transition:background .3s,border-color .3s; }
  .card + .card { margin-top:18px; }
  .card h2 { font-size:1rem; font-weight:700; display:flex; align-items:center; gap:10px; margin-bottom:16px; }
  .card h2 .ico { width:32px; height:32px; border-radius:10px; background:var(--soft); display:grid; place-items:center; }

  .samples { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px; align-items:center; }
  .samples span { font-size:.8rem; color:var(--muted); margin-right:4px; }
  .chip { border:1px solid var(--border); background:var(--card); color:var(--text); padding:7px 13px; border-radius:999px;
          font-size:.8rem; font-weight:600; cursor:pointer; transition:.2s; }
  .chip:hover { border-color:var(--accent); color:var(--accent); transform:translateY(-1px); }

  .field { margin-bottom:18px; }
  .field:last-child { margin-bottom:0; }
  .field-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; gap:10px; }
  label.lbl { font-weight:600; font-size:.88rem; }
  .num { display:flex; align-items:center; gap:4px; background:var(--soft); border:1px solid var(--border); border-radius:10px; padding:4px 10px; }
  .num input { width:52px; border:none; outline:none; background:transparent; color:var(--text); font-weight:700; font-size:.95rem; text-align:right; font-family:inherit; }
  .num input::-webkit-inner-spin-button { display:none; }
  .num span { color:var(--muted); font-size:.85rem; }
  .hint { font-size:.75rem; color:var(--muted); margin-top:5px; }

  input[type=range] { -webkit-appearance:none; appearance:none; width:100%; height:8px; border-radius:99px; outline:none; cursor:pointer;
    background:linear-gradient(90deg,var(--accent) var(--p,50%),var(--border) var(--p,50%)); }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:22px; height:22px; border-radius:50%; background:#fff;
    border:4px solid var(--accent); box-shadow:0 4px 10px rgba(99,102,241,.4); transition:transform .15s; }
  input[type=range]::-webkit-slider-thumb:hover { transform:scale(1.15); }
  input[type=range]::-moz-range-thumb { width:16px; height:16px; border-radius:50%; background:#fff; border:4px solid var(--accent); }

  .seg { display:flex; gap:6px; background:var(--soft); padding:5px; border-radius:14px; border:1px solid var(--border); flex-wrap:wrap; }
  .seg button { flex:1; min-width:80px; border:none; background:transparent; color:var(--muted); padding:9px 12px; border-radius:10px;
                font-weight:600; font-size:.85rem; cursor:pointer; font-family:inherit; transition:.2s; }
  .seg button:hover { color:var(--text); }
  .seg button.on { background:var(--card); color:var(--accent); box-shadow:0 4px 12px -4px rgba(15,23,42,.25); }

  .actions { display:flex; gap:10px; margin-top:20px; }
  .btn { border:none; cursor:pointer; font-family:inherit; font-weight:700; font-size:.95rem; border-radius:14px; padding:14px 20px; transition:.2s; }
  .btn.primary { flex:1; color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 10px 24px -8px var(--accent); }
  .btn.primary:hover { transform:translateY(-2px); box-shadow:0 14px 28px -8px var(--accent); }
  .btn.primary:disabled { opacity:.7; cursor:wait; transform:none; }
  .btn.ghost { background:var(--card); color:var(--text); border:1px solid var(--border); }
  .btn.ghost:hover { border-color:var(--accent); }

  .sticky { position:sticky; top:86px; }
  @media (max-width:960px) { .sticky { position:static; } }

  .strength { display:flex; align-items:center; gap:14px; }
  .radar { width:150px; height:150px; flex-shrink:0; }
  .radar .grid-line { fill:none; stroke:var(--border); stroke-width:1; }
  .radar .poly { fill:color-mix(in srgb, var(--accent) 28%, transparent); stroke:var(--accent); stroke-width:2; transition:all .25s; }
  .radar text { fill:var(--muted); font-size:9px; font-weight:600; font-family:inherit; }
  .meter { flex:1; }
  .meter b { font-size:1.9rem; letter-spacing:-.03em; }
  .meter small { color:var(--muted); display:block; font-size:.78rem; }
  .bar { height:8px; border-radius:99px; background:var(--border); margin-top:10px; overflow:hidden; }
  .bar > i { display:block; height:100%; width:0; border-radius:99px; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .4s; }

  .empty { text-align:center; padding:26px 10px; color:var(--muted); }
  .empty .big { font-size:2.6rem; margin-bottom:6px; }
  .skeleton div { height:14px; border-radius:8px; margin:12px 0; background:linear-gradient(90deg,var(--soft),var(--border),var(--soft)); background-size:200% 100%; animation:sh 1.2s infinite; }
  @keyframes sh { to { background-position:-200% 0; } }

  .verdict { display:flex; gap:18px; align-items:center; animation:pop .5s ease; }
  @keyframes pop { from { opacity:0; transform:translateY(10px) scale(.98); } to { opacity:1; transform:none; } }
  .gauge { position:relative; width:128px; height:128px; flex-shrink:0; }
  .gauge svg { transform:rotate(-90deg); }
  .gauge circle { fill:none; stroke-width:11; }
  .gauge .bg { stroke:var(--border); }
  .gauge .fg { stroke-linecap:round; stroke-dasharray:339.29; stroke-dashoffset:339.29; transition:stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1); }
  .gauge .val { position:absolute; inset:0; display:grid; place-content:center; text-align:center; }
  .gauge .val b { font-size:1.7rem; letter-spacing:-.03em; }
  .gauge .val small { font-size:.65rem; color:var(--muted); }
  .verdict h3 { font-size:1.3rem; letter-spacing:-.02em; line-height:1.25; }
  .verdict p { color:var(--muted); font-size:.9rem; margin-top:6px; }
  .tag { display:inline-block; font-size:.7rem; font-weight:700; padding:4px 10px; border-radius:99px; margin-bottom:8px; }
  .tag.good { background:color-mix(in srgb,var(--good) 18%,transparent); color:var(--good); }
  .tag.warn { background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn); }

  details { border:1px solid var(--border); border-radius:14px; margin-top:12px; background:var(--card); overflow:hidden; }
  details summary { cursor:pointer; padding:12px 14px; font-weight:700; font-size:.88rem; list-style:none; display:flex; justify-content:space-between; align-items:center; }
  details summary::-webkit-details-marker { display:none; }
  details summary::after { content:"＋"; color:var(--muted); }
  details[open] summary::after { content:"－"; }
  details ul { padding:0 14px 14px 32px; font-size:.86rem; color:var(--muted); }
  details li { margin:6px 0; }
  details li::marker { color:var(--accent); }

  .imp { margin:10px 0; font-size:.82rem; }
  .imp div { display:flex; justify-content:space-between; margin-bottom:4px; }
  .imp .bar { margin-top:0; height:7px; }
  .res-actions { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }
  .res-actions .btn { padding:10px 14px; font-size:.82rem; }
  .disclaimer { font-size:.72rem; color:var(--muted); margin-top:14px; }

  .hist { list-style:none; }
  .hist li { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:10px 12px; border:1px solid var(--border); border-radius:12px; margin-bottom:8px; cursor:pointer; font-size:.82rem; transition:.2s; }
  .hist li:hover { border-color:var(--accent); transform:translateX(3px); }
  .hist .pill { font-weight:700; font-size:.72rem; padding:3px 9px; border-radius:99px; }
  .hist .pill.g { background:color-mix(in srgb,var(--good) 18%,transparent); color:var(--good); }
  .hist .pill.w { background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn); }
  .hist small { color:var(--muted); }

  .toast { position:fixed; bottom:24px; left:50%; transform:translate(-50%,80px); background:var(--text); color:var(--bg); padding:11px 18px;
           border-radius:12px; font-size:.85rem; font-weight:600; transition:transform .3s; z-index:50; }
  .toast.show { transform:translate(-50%,0); }
  footer { text-align:center; color:var(--muted); font-size:.78rem; padding:20px; }
  @media print { header, .actions, .samples, .res-actions, .toast { display:none; } body::before, body::after { display:none; } }
</style>
</head>
<body>
<header>
  <div class="nav">
    <div class="brand">
      <div class="logo">🎯</div>
      <div>Placement Predictor<small>AI-powered career insights</small></div>
    </div>
    <button class="icon-btn" id="themeBtn" aria-label="Toggle theme" title="Toggle light / dark">🌙</button>
  </div>
</header>

<main>
  <section class="hero">
    <h1>Will you get <span>placed</span>? Find out in seconds.</h1>
    <p>Enter your academic profile and our decision-tree model will predict your placement outcome, explain why, and show what you can improve.</p>
  </section>

  <div class="grid">
    <!-- LEFT: FORM -->
    <div>
      <div class="samples">
        <span>Try a sample:</span>
        <button class="chip" data-sample="strong">🚀 Strong profile</button>
        <button class="chip" data-sample="average">📘 Average profile</button>
        <button class="chip" data-sample="weak">🌱 Needs work</button>
      </div>
      <div id="form"></div>
      <div class="actions">
        <button class="btn ghost" id="resetBtn" type="button">Reset</button>
        <button class="btn primary" id="predictBtn" type="button">Predict my placement →</button>
      </div>
    </div>

    <!-- RIGHT: RESULT -->
    <div class="sticky">
      <div class="card">
        <h2><span class="ico">📊</span> Live profile strength</h2>
        <div class="strength">
          <svg class="radar" viewBox="0 0 200 200" id="radar"></svg>
          <div class="meter">
            <b id="avgVal">0%</b>
            <small>Average academic score</small>
            <div class="bar"><i id="avgBar"></i></div>
          </div>
        </div>
      </div>

      <div class="card" id="resultCard">
        <h2><span class="ico">✨</span> Your prediction</h2>
        <div id="result">
          <div class="empty"><div class="big">🔮</div>Fill in your details and press <b>Predict</b> to see your result here.</div>
        </div>
      </div>

      <div class="card" id="histCard" style="display:none">
        <h2><span class="ico">🕘</span> Recent predictions</h2>
        <ul class="hist" id="hist"></ul>
      </div>
    </div>
  </div>
</main>
<footer>Built with Flask, pandas &amp; scikit-learn · For guidance only, not a guarantee of placement.</footer>
<div class="toast" id="toast"></div>

<script>
const META = __META__;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const numMeta = Object.fromEntries(META.numeric.map((n) => [n.key, n]));
const vals = {};
let lastResult = null;

/* ---------- Theme ---------- */
function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  $("#themeBtn").textContent = t === "dark" ? "☀️" : "🌙";
}
setTheme(document.documentElement.getAttribute("data-theme"));
$("#themeBtn").onclick = () => setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");

/* ---------- Toast ---------- */
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- Form builder ---------- */
function numericField(key) {
  const m = numMeta[key];
  return `<div class="field">
    <div class="field-head">
      <label class="lbl" for="r_${key}">${esc(m.label)}</label>
      <div class="num"><input type="number" id="n_${key}" min="0" max="100" step="0.1"><span>%</span></div>
    </div>
    <input type="range" id="r_${key}" min="0" max="100" step="0.5">
    <div class="hint">${esc(m.hint)}</div>
  </div>`;
}
function catField(key) {
  const c = META.categorical[key];
  return `<div class="field">
    <div class="field-head"><label class="lbl">${esc(c.label)}</label></div>
    <div class="seg" data-key="${key}">
      ${c.options.map((o) => `<button type="button" data-v="${o.value}">${esc(o.label)}</button>`).join("")}
    </div>
  </div>`;
}
function buildForm() {
  $("#form").innerHTML = META.sections.map((s) => `
    <div class="card">
      <h2><span class="ico">${s.icon}</span> ${esc(s.title)}</h2>
      ${s.fields.map((f) => (numMeta[f] ? numericField(f) : catField(f))).join("")}
    </div>`).join("");

  META.numeric.forEach(({ key }) => {
    const r = $("#r_" + key), n = $("#n_" + key);
    r.addEventListener("input", () => setNum(key, r.value));
    n.addEventListener("input", () => setNum(key, n.value === "" ? 0 : Math.min(100, Math.max(0, +n.value))));
  });
  document.querySelectorAll(".seg").forEach((seg) => {
    seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setCat(seg.dataset.key, +b.dataset.v)));
  });
}
function setNum(key, v) {
  v = +v; vals[key] = v;
  const r = $("#r_" + key), n = $("#n_" + key);
  r.value = v; if (document.activeElement !== n) n.value = v;
  r.style.setProperty("--p", v + "%");
  updateStrength();
}
function setCat(key, v) {
  vals[key] = v;
  document.querySelectorAll(`.seg[data-key="${key}"] button`).forEach((b) => b.classList.toggle("on", +b.dataset.v === v));
}
function fill(data) {
  Object.entries(data).forEach(([k, v]) => (numMeta[k] ? setNum(k, v) : setCat(k, +v)));
}

const SAMPLES = {
  strong:  { gender:1, ssc_p:82, ssc_b:0, hsc_p:78, hsc_b:0, hsc_s:1, degree_p:76, degree_t:0, workex:1, etest_p:85, specialisation:0, mba_p:73 },
  average: { gender:0, ssc_p:66, ssc_b:1, hsc_p:63, hsc_b:1, hsc_s:1, degree_p:62, degree_t:0, workex:0, etest_p:68, specialisation:1, mba_p:62 },
  weak:    { gender:0, ssc_p:48, ssc_b:1, hsc_p:46, hsc_b:1, hsc_s:2, degree_p:50, degree_t:2, workex:0, etest_p:42, specialisation:1, mba_p:52 },
};
function defaults() {
  const d = {};
  META.numeric.forEach((n) => (d[n.key] = 65));
  Object.entries(META.categorical).forEach(([k, c]) => (d[k] = c.options[0].value));
  return d;
}
document.querySelectorAll("[data-sample]").forEach((b) => (b.onclick = () => { fill(SAMPLES[b.dataset.sample]); toast("Sample loaded"); }));
$("#resetBtn").onclick = () => { fill(defaults()); resetResult(); };

/* ---------- Radar + strength ---------- */
const AXES = [["ssc_p","10th"],["hsc_p","12th"],["degree_p","Degree"],["etest_p","E-test"],["mba_p","MBA"]];
function pt(i, r) {
  const a = (-90 + i * (360 / AXES.length)) * Math.PI / 180;
  return [100 + r * Math.cos(a), 100 + r * Math.sin(a)];
}
function updateStrength() {
  let svg = [25, 50, 75, 100].map((lv) =>
    `<polygon class="grid-line" points="${AXES.map((_, i) => pt(i, 62 * lv / 100).join(",")).join(" ")}"/>`).join("");
  svg += AXES.map((_, i) => { const [x, y] = pt(i, 62); return `<line class="grid-line" x1="100" y1="100" x2="${x}" y2="${y}"/>`; }).join("");
  const poly = AXES.map(([k], i) => pt(i, 62 * (vals[k] || 0) / 100).join(",")).join(" ");
  svg += `<polygon class="poly" points="${poly}"/>`;
  svg += AXES.map(([, l], i) => { const [x, y] = pt(i, 78); return `<text x="${x}" y="${y + 3}" text-anchor="middle">${l}</text>`; }).join("");
  $("#radar").innerHTML = svg;
  const avg = AXES.reduce((s, [k]) => s + (vals[k] || 0), 0) / AXES.length;
  $("#avgVal").textContent = avg.toFixed(0) + "%";
  $("#avgBar").style.width = avg + "%";
}

/* ---------- Prediction ---------- */
function resetResult() {
  lastResult = null;
  $("#result").innerHTML = `<div class="empty"><div class="big">🔮</div>Fill in your details and press <b>Predict</b> to see your result here.</div>`;
}
const list = (items) => `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`;

function renderResult(r) {
  const good = r.prediction === 1;
  const color = good ? "var(--good)" : "var(--warn)";
  const conf = Math.round(r.confidence * 100);
  let html = `
    <div class="verdict">
      <div class="gauge">
        <svg width="128" height="128" viewBox="0 0 128 128">
          <circle class="bg" cx="64" cy="64" r="54"/>
          <circle class="fg" id="gfg" cx="64" cy="64" r="54" style="stroke:${color}"/>
        </svg>
        <div class="val"><b>${conf}%</b><small>confidence</small></div>
      </div>
      <div>
        <span class="tag ${good ? "good" : "warn"}">${good ? "PLACED" : "NOT PLACED"}</span>
        <h3>${esc(r.headline)}</h3>
        <p>${esc(r.summary)}</p>
      </div>
    </div>`;
  if (r.strengths.length) html += `<details open><summary>💪 What is working in your favour</summary>${list(r.strengths)}</details>`;
  if (r.gaps.length) html += `<details open><summary>🎯 Where you can improve</summary>${list(r.gaps)}</details>`;
  if (r.whatif.length) html += `<details open><summary>🔁 What could change the outcome</summary>${list(r.whatif)}</details>`;
  if (r.reasons.length) html += `<details><summary>🧠 How the model reached this decision</summary>${list(r.reasons)}</details>`;
  if (r.importance.length) {
    const max = r.importance[0].value;
    html += `<details><summary>📈 What matters most to this model</summary><div style="padding:0 14px 14px">` +
      r.importance.map((i) => `<div class="imp"><div><span>${esc(i.label)}</span><b>${(i.value * 100).toFixed(0)}%</b></div><div class="bar"><i style="width:${(i.value / max) * 100}%"></i></div></div>`).join("") +
      `</div></details>`;
  }
  html += `<div class="res-actions">
      <button class="btn ghost" id="copyBtn">📋 Copy summary</button>
      <button class="btn ghost" id="printBtn">🖨️ Print</button>
    </div>
    <p class="disclaimer">This is a statistical prediction from a machine-learning model trained on past data. It is meant for guidance only.</p>`;
  $("#result").innerHTML = html;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("#gfg").style.strokeDashoffset = 339.29 * (1 - r.confidence);
  }));
  $("#copyBtn").onclick = copySummary;
  $("#printBtn").onclick = () => window.print();
}

function copySummary() {
  const r = lastResult; if (!r) return;
  const lines = [r.headline, r.summary, ""];
  if (r.strengths.length) lines.push("Strengths:", ...r.strengths.map((s) => "- " + s), "");
  if (r.gaps.length) lines.push("Improve:", ...r.gaps.map((s) => "- " + s), "");
  if (r.whatif.length) lines.push("What could change the outcome:", ...r.whatif.map((s) => "- " + s));
  navigator.clipboard.writeText(lines.join("\n")).then(() => toast("Summary copied ✓"), () => toast("Could not copy"));
}

async function predict() {
  const btn = $("#predictBtn");
  btn.disabled = true; btn.textContent = "Analysing…";
  $("#result").innerHTML = `<div class="skeleton"><div style="width:70%"></div><div></div><div style="width:85%"></div><div style="width:55%"></div></div>`;
  if (window.innerWidth <= 960) $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(vals) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong");
    lastResult = data; renderResult(data); saveHistory(data);
  } catch (e) {
    $("#result").innerHTML = `<div class="empty"><div class="big">⚠️</div>${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Predict my placement →";
  }
}
$("#predictBtn").onclick = predict;

/* ---------- History ---------- */
function loadHistory() { try { return JSON.parse(localStorage.getItem("ph") || "[]"); } catch (e) { return []; } }
function saveHistory(r) {
  try {
    const h = loadHistory();
    h.unshift({ t: Date.now(), vals: { ...vals }, ok: r.prediction === 1, c: Math.round(r.confidence * 100) });
    localStorage.setItem("ph", JSON.stringify(h.slice(0, 5)));
  } catch (e) {}
  renderHistory();
}
function renderHistory() {
  const h = loadHistory();
  $("#histCard").style.display = h.length ? "block" : "none";
  $("#hist").innerHTML = h.map((x, i) => `
    <li data-i="${i}">
      <div><span class="pill ${x.ok ? "g" : "w"}">${x.ok ? "Placed" : "Not placed"}</span> <small>${x.c}% confidence</small></div>
      <small>${new Date(x.t).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</small>
    </li>`).join("");
  document.querySelectorAll("#hist li").forEach((li) => (li.onclick = () => { fill(h[+li.dataset.i].vals); toast("Inputs restored"); }));
}

/* ---------- Init ---------- */
buildForm();
fill(defaults());
renderHistory();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)