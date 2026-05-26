"""
Language Detector — Render-ready single file
© 2024 Aravind Ugge. All rights reserved.

Deploy to Render:
  Build command : pip install -r requirements.txt
  Start command : python app.py
  Environment   : (none needed — everything is auto-detected)
"""

import os, re, csv, pickle, subprocess, sys

# ── Auto-install gunicorn and boot it when running on Render ─────────────────
# Render sets the PORT env var. When present we launch ourselves under gunicorn
# so the process is production-grade without a separate Procfile or render.yaml.
_PORT = os.environ.get("PORT")
if _PORT and "gunicorn" not in sys.argv[0]:
    try:
        import gunicorn  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn", "-q"])
    os.execv(
        sys.executable,
        [sys.executable, "-m", "gunicorn",
         "app:app",
         "--bind", f"0.0.0.0:{_PORT}",
         "--workers", "1",
         "--timeout", "120"]
    )

# ── Normal imports (after gunicorn re-exec guard) ─────────────────────────────
import numpy as np
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Flatten, Embedding, Dropout, LSTM
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(BASE_DIR, "Language Detection.csv")
_CACHE         = os.environ.get("MODEL_CACHE_DIR", "/tmp")
MODEL_PATH     = os.path.join(_CACHE, "lang_model.keras")
TOKENIZER_PATH = os.path.join(_CACHE, "tokenizer.pkl")
ENCODER_PATH   = os.path.join(_CACHE, "label_encoder.pkl")
MAX_WORDS      = 10000
MAX_LEN        = 150
EMBEDDING_DIM  = 128

# ── NLTK ──────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ── Text helpers ──────────────────────────────────────────────────────────────
def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def extract_keywords(tokens):
    seen, result = set(), []
    for t in tokens:
        if t not in STOP_WORDS and len(t) > 1 and t not in seen:
            seen.add(t); result.append(t)
    return result[:5]

def load_csv(path):
    texts, langs = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t, l = row.get("Text", "").strip(), row.get("Language", "").strip()
            if t and l:
                texts.append(t); langs.append(l)
    return texts, langs

# ── Model definition ──────────────────────────────────────────────────────────
def create_model(num_classes):
    m = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation="relu", padding="same"),
        MaxPooling1D(2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

# ── Load or train ─────────────────────────────────────────────────────────────
if all(os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH]):
    print("✅ Loading saved model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f: tokenizer     = pickle.load(f)
    with open(ENCODER_PATH,   "rb") as f: label_encoder = pickle.load(f)
else:
    print("🔄 Training model from scratch …")
    raw_texts, raw_langs = load_csv(CSV_PATH)
    label_encoder        = LabelEncoder()
    labels               = label_encoder.fit_transform(raw_langs)
    clean_texts          = [clean_text(t) for t in raw_texts]
    tokenizer            = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(clean_texts)
    X = pad_sequences(tokenizer.texts_to_sequences(clean_texts), maxlen=MAX_LEN)
    y = tf.keras.utils.to_categorical(labels, len(label_encoder.classes_))
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(len(label_encoder.classes_))
    model.fit(X_tr, y_tr, epochs=5, batch_size=32, validation_data=(X_v, y_v), verbose=1)
    os.makedirs(_CACHE, exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f: pickle.dump(tokenizer,     f)
    with open(ENCODER_PATH,   "wb") as f: pickle.dump(label_encoder, f)
    print("✅ Model saved.")

LANGUAGES = list(label_encoder.classes_)

# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess(text):
    cleaned = clean_text(text)
    if not cleaned:
        return None, 0, 0, 0, 0, []
    tokens = simple_tokenize(cleaned)
    padded = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LEN)
    return (padded, len(tokens), len(cleaned),
            len(set(tokens)), len(re.findall(r"[^a-zA-Z0-9\s]", cleaned)),
            extract_keywords(tokens))

def run_detect(text):
    padded, wc, cc, uw, sc, kw = preprocess(text)
    if padded is None:
        return {"error": "Empty text after preprocessing"}
    pred = model.predict(padded, verbose=0)[0]
    idx  = int(np.argmax(pred))
    return {
        "language":      LANGUAGES[idx],
        "confidence":    float(pred[idx]),
        "word_count":    wc,
        "char_count":    cc,
        "unique_words":  uw,
        "special_chars": sc,
        "keywords":      kw,
        "probabilities": [float(p) for p in pred],
        "all_languages": LANGUAGES,
    }

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Language Detector</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#0d0d14;--surface:#16161f;--card:#1c1c2b;--border:#2a2a3d;--accent:#7c6aff;--accent2:#ff6a9a;--green:#2affa8;--text:#e8e8f0;--muted:#6b6b8a;--radius:14px}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:40px 20px 60px}
    body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(124,106,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(124,106,255,.04) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
    .wrapper{position:relative;z-index:1;max-width:780px;margin:auto}
    header{text-align:center;margin-bottom:36px}
    header .badge{display:inline-block;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:3px;text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);border-radius:100px;padding:4px 14px;margin-bottom:14px}
    header h1{font-family:'Space Mono',monospace;font-size:clamp(26px,5vw,42px);font-weight:700;background:linear-gradient(135deg,#fff 30%,var(--accent) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2}
    header p{margin-top:10px;color:var(--muted);font-size:15px}
    .input-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-bottom:20px}
    textarea{width:100%;height:140px;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:14px 16px;font-family:'DM Sans',sans-serif;font-size:15px;resize:vertical;outline:none;transition:border-color .2s}
    textarea:focus{border-color:var(--accent)}
    textarea::placeholder{color:var(--muted)}
    .char-count{font-size:12px;color:var(--muted);text-align:right;margin-top:6px}
    .btn-row{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
    button{display:flex;align-items:center;gap:7px;padding:10px 20px;border:none;border-radius:8px;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:600;cursor:pointer;transition:transform .15s,opacity .15s}
    button:hover{opacity:.85;transform:translateY(-1px)}
    button:active{transform:translateY(0)}
    .btn-detect{background:var(--accent);color:#fff}
    .btn-mic{background:var(--accent2);color:#fff}
    .btn-clear{background:var(--border);color:var(--text)}
    #spinner{display:none;text-align:center;padding:20px;color:var(--muted);font-size:13px;letter-spacing:2px}
    .dot-anim{animation:pulse 1s infinite}
    @keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
    .result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-bottom:16px;display:none;animation:slideUp .35s ease}
    @keyframes slideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
    .lang-result{display:flex;align-items:center;gap:14px;margin-bottom:18px}
    .lang-flag{width:48px;height:36px;border-radius:6px;object-fit:cover;border:1px solid var(--border)}
    .lang-name{font-family:'Space Mono',monospace;font-size:22px;font-weight:700}
    .confidence-pill{display:inline-block;background:rgba(42,255,168,.12);color:var(--green);border:1px solid rgba(42,255,168,.3);border-radius:100px;padding:3px 12px;font-size:13px;font-weight:600;font-family:'Space Mono',monospace}
    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:18px}
    .stat-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center}
    .stat-box .val{font-family:'Space Mono',monospace;font-size:22px;font-weight:700;color:var(--accent)}
    .stat-box .lbl{font-size:11px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:1px}
    .kw-label{font-size:11px;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:8px}
    .kw-cloud{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:20px}
    .kw-chip{background:rgba(124,106,255,.12);color:var(--accent);border:1px solid rgba(124,106,255,.25);border-radius:100px;padding:4px 12px;font-size:13px}
    .chart-wrap{position:relative;height:200px}
    .history-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;display:none}
    .history-card h3{font-family:'Space Mono',monospace;font-size:13px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
    .history-item{display:flex;justify-content:space-between;align-items:flex-start;padding:10px 0;border-bottom:1px solid var(--border);gap:12px}
    .history-item:last-child{border-bottom:none}
    .hi-text{font-size:13px;color:var(--muted);flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
    .hi-lang{font-family:'Space Mono',monospace;font-size:13px;color:var(--green);white-space:nowrap}
    .error-box{background:rgba(255,106,154,.08);border:1px solid rgba(255,106,154,.3);color:var(--accent2);border-radius:10px;padding:14px 16px;font-size:14px;display:none;margin-bottom:16px}
    .site-footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);text-align:center}
    .site-footer p{font-family:'Space Mono',monospace;font-size:12px;color:var(--muted);letter-spacing:1px}
    .site-footer span{color:var(--accent);font-weight:700}
  </style>
</head>
<body>
<div class="wrapper">
  <header>
    <div class="badge">AI · NLP · Deep Learning</div>
    <h1>Language Detector</h1>
    <p>CNN + LSTM model trained on multilingual text</p>
  </header>

  <div class="input-card">
    <textarea id="inputText" placeholder="Type or paste text in any language…"></textarea>
    <div class="char-count"><span id="charCount">0</span> characters</div>
    <div class="btn-row">
      <button class="btn-detect" onclick="detectLanguage()">⚡ Detect Language</button>
      <button class="btn-mic"    onclick="startVoiceInput()">🎤 Speak</button>
      <button class="btn-clear"  onclick="clearAll()">✕ Clear</button>
    </div>
  </div>

  <div id="spinner"><span class="dot-anim">Analyzing</span> ···</div>
  <div class="error-box" id="errorBox"></div>

  <div class="result-card" id="resultCard">
    <div class="lang-result">
      <img class="lang-flag" id="flagImg" src="" alt="" onerror="this.style.display='none'"/>
      <div>
        <div class="lang-name" id="langName"></div>
        <span class="confidence-pill" id="confPill"></span>
      </div>
    </div>
    <div class="stats-grid" id="statsGrid"></div>
    <div class="kw-label">Top Keywords</div>
    <div class="kw-cloud" id="kwCloud"></div>
    <div class="chart-wrap"><canvas id="probChart"></canvas></div>
  </div>

  <div class="history-card" id="historyCard">
    <h3>📋 Recent Detections</h3>
    <div id="historyList"></div>
  </div>

  <footer class="site-footer">
    <p>© 2024 <span>Aravind Ugge</span> · All rights reserved</p>
  </footer>
</div>

<script>
  let chartInstance=null,detectionHistory=[],debounceTimer;

  document.getElementById('inputText').addEventListener('input',()=>{
    const len=document.getElementById('inputText').value.length;
    document.getElementById('charCount').textContent=len;
    clearTimeout(debounceTimer);
    if(len>15)debounceTimer=setTimeout(detectLanguage,1200);
  });

  async function detectLanguage(){
    const text=document.getElementById('inputText').value.trim();
    if(!text)return;
    showSpinner(true);hideError();
    document.getElementById('resultCard').style.display='none';
    try{
      const res=await fetch('/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      const data=await res.json();
      showSpinner(false);
      if(data.error){showError(data.error);return;}
      renderResult(data,text);
    }catch(err){showSpinner(false);showError('Network error — is the server running?');}
  }

  function renderResult(data,inputText){
    const code=data.language.slice(0,2).toLowerCase();
    const flag=document.getElementById('flagImg');
    flag.src=`https://flagcdn.com/48x36/${code}.png`;flag.style.display='block';
    document.getElementById('langName').textContent=data.language;
    document.getElementById('confPill').textContent=(data.confidence*100).toFixed(1)+'% confidence';
    document.getElementById('statsGrid').innerHTML=[
      {val:data.word_count,lbl:'Words'},{val:data.char_count,lbl:'Characters'},
      {val:data.unique_words,lbl:'Unique Words'},{val:data.special_chars,lbl:'Special Chars'},
    ].map(s=>`<div class="stat-box"><div class="val">${s.val}</div><div class="lbl">${s.lbl}</div></div>`).join('');
    document.getElementById('kwCloud').innerHTML=data.keywords.length
      ?data.keywords.map(k=>`<span class="kw-chip">${k}</span>`).join('')
      :'<span style="color:var(--muted);font-size:13px">No keywords extracted</span>';
    renderChart(data.probabilities,data.all_languages);
    document.getElementById('resultCard').style.display='block';
    addHistory(data,inputText);
  }

  function renderChart(probs,labels){
    const paired=probs.map((p,i)=>({p,l:labels?labels[i]:'Lang'+(i+1)}));
    paired.sort((a,b)=>b.p-a.p);
    const top=paired.slice(0,10);
    if(chartInstance)chartInstance.destroy();
    chartInstance=new Chart(document.getElementById('probChart'),{
      type:'bar',
      data:{
        labels:top.map(x=>x.l),
        datasets:[{
          label:'Confidence %',
          data:top.map(x=>+(x.p*100).toFixed(2)),
          backgroundColor:top.map((_,i)=>i===0?'rgba(124,106,255,.8)':'rgba(124,106,255,.2)'),
          borderColor:'rgba(124,106,255,.9)',borderWidth:1,borderRadius:4
        }]
      },
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#6b6b8a',font:{size:11}},grid:{color:'rgba(255,255,255,.04)'}},
          y:{beginAtZero:true,ticks:{color:'#6b6b8a',font:{size:11},callback:v=>v+'%'},grid:{color:'rgba(255,255,255,.04)'}}
        }
      }
    });
  }

  function addHistory(data,text){
    detectionHistory.unshift({lang:data.language,conf:data.confidence,text});
    if(detectionHistory.length>8)detectionHistory.pop();
    document.getElementById('historyCard').style.display='block';
    document.getElementById('historyList').innerHTML=detectionHistory.map(h=>
      `<div class="history-item">
        <span class="hi-text">${escHtml(h.text)}</span>
        <span class="hi-lang">${h.lang} · ${(h.conf*100).toFixed(0)}%</span>
       </div>`).join('');
  }

  function clearAll(){
    document.getElementById('inputText').value='';
    document.getElementById('charCount').textContent='0';
    document.getElementById('resultCard').style.display='none';
    hideError();
  }
  function showSpinner(on){document.getElementById('spinner').style.display=on?'block':'none';}
  function showError(msg){const el=document.getElementById('errorBox');el.textContent='⚠ '+msg;el.style.display='block';}
  function hideError(){document.getElementById('errorBox').style.display='none';}
  function escHtml(str){return str.replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}

  function startVoiceInput(){
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){showError('Voice input not supported in this browser.');return;}
    const r=new SR();r.lang='en-US';r.start();
    r.onresult=e=>{document.getElementById('inputText').value=e.results[0][0].transcript;detectLanguage();};
    r.onerror=()=>showError('Microphone error. Please allow microphone access.');
  }
</script>
</body>
</html>"""

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = run_detect(text)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Local dev entry-point ─────────────────────────────────────────────────────
# On Render, the gunicorn re-exec above fires before we reach here.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
