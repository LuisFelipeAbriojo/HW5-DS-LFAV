"""App de Streamlit. Solo LEE resultados precomputados (results/, cache/); nunca transcribe el corpus al iniciar.
La única llamada al motor es en la pestaña "Probador", y es la misma función que usa todo lo demás.

    streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from vozregistro.config import load_config, path

cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")
RES = path("results_dir")


@st.cache_data
def load_json(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_script(task, clip):
    return yaml.safe_load((path("scripts_dir") / task / f"{clip}.yaml").read_text(encoding="utf-8"))


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def ci(v):
    return f"[{100 * v[0]:.0f}–{100 * v[1]:.0f}%]"


st.sidebar.title(cfg["app"]["title"])
group = st.sidebar.radio("Grupo de audio", ["real", "synthetic"],
                         format_func=lambda g: "Grabaciones reales" if g == "real" else "Sintético (TTS, aparte)")
suffix = "" if group == "real" else "_synthetic"
if group == "synthetic":
    st.sidebar.warning("Voces sintéticas Edge TTS. No se mezclan con los números principales.")
st.sidebar.caption("Toda la información de los audios es inventada. Ver README.")

tab_prob, tab_corpus, tab_res, tab_order, tab_test = st.tabs(
    ["1 · Problema y método", "2 · Corpus", "3 · Resultados (entrevistas)", "4 · Pedidos y escalera", "5 · Probador"])

# ---------------------------------------------------------------- 1
with tab_prob:
    st.header("El audio que nadie procesa")
    st.markdown("""
Transcribir es donde empieza el trabajo, no donde termina. Una organización no necesita texto:
necesita **campos verificados, en una tabla, con una medida de cuánto confiar en ellos.**

| Ya existe | Qué hace | Qué falta |
|---|---|---|
| Dictado del teclado | Frases cortas | No se guarda nada |
| Transcripción de notas de voz | Muestra el texto | Ninguna fila en una base de datos |
| Subtítulos de Zoom/Meet | Subtítulos | El audio sale de la organización |
| Otter / Fireflies | Notas de reunión | Pago por minuto, datos en el extranjero |
""")
    st.subheader("Pipeline")
    st.markdown("""
**Una vez:** guion → grabación → verdad de campo (gratis: el guion ES la referencia) → transcripción en GPU → caché versionada → escalera de degradación.

**Por cada audio nuevo:** `process_audio()` → ASR (faster-whisper, dispositivo leído del modelo) → normalización numérica →
extracción (reglas o LLM con JSON) → **verificación literal** (si el valor no está en la transcripción: `sin_verificar`) →
en pedidos, **validación aritmética** y diagnóstico de la causa.
""")
    st.info("Una sola función, dos esquemas: `vozregistro.engine.process_audio(ruta, schema='interview'|'order')`.")

# ---------------------------------------------------------------- 2
with tab_corpus:
    task = st.radio("Tarea", ["interview", "order"], horizontal=True,
                    format_func=lambda t: "Entrevistas" if t == "interview" else "Llamadas de venta")
    ev = load_json(f"{task}_eval{suffix}.json")
    clip_ids = sorted(p.stem for p in (path("scripts_dir") / task).glob("*.yaml"))
    clip = st.selectbox("Clip", clip_ids)
    sc = load_script(task, clip)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Audio")
        audio_base = path("audio_dir" if group == "real" else "synthetic_dir") / task
        found = [audio_base / f"{clip}{e}" for e in cfg["audio_ext"] if (audio_base / f"{clip}{e}").exists()]
        if found:
            st.audio(str(found[0]))
        else:
            st.caption("El audio no se versiona (ver README para solicitarlo).")
        st.markdown(f"**Variación:** {sc.get('variacion', '')}")
        if task == "order":
            st.json(sc["traps"])
        st.markdown("**Verdad de campo**")
        st.json(sc["ground_truth"])
    with c2:
        st.subheader("Transcripción del modelo")
        hyp = None
        if ev and "L0" in ev["levels"]:
            rows = {r["clip"]: r for r in ev["levels"]["L0"]["clips"]}
            from vozregistro import cache
            tg = ev["levels"]["L0"]["fuente"]
            data = cache.load_tag(task, tg, group)
            if clip in data:
                hyp = data[clip]["text"]
                st.caption(f"{tg} · WER {rows[clip]['wer']:.3f} · device={data[clip]['device']}")
        st.text_area("Hipótesis", hyp or "(sin transcripción cacheada todavía)", height=260)
        st.text_area("Referencia (guion)", " ".join(next(iter(t.values())) for t in sc["turns"]), height=200)

# ---------------------------------------------------------------- 3
with tab_res:
    bench = load_json("asr_benchmark.json")
    ev = load_json(f"interview_eval{suffix}.json")
    st.header("Capa 1 · Transcripción")
    if bench and bench["por_config"]:
        rows = []
        for tg, m in bench["por_config"].items():
            if not m:
                continue
            a = m["agg"]
            _, model, ctype, dev, glos = tg.split("__")
            rows.append({"modelo": model, "precisión": ctype, "dispositivo (medido)": ", ".join(a["devices"]),
                         "glosario": glos, "WER": a["wer"]["media"], "WER IC95": f"{a['wer']['ic95'][0]:.3f}–{a['wer']['ic95'][1]:.3f}",
                         "CER": a["cer"]["media"], "s / min audio": a["s_per_min"]["media"],
                         "VRAM pico (MB)": a["peak_vram_mb"], "Δ VRAM (MB)": a["delta_vram_mb_max"],
                         "recall términos": a["term_recall"]["media"]})
        st.dataframe(pd.DataFrame(rows).sort_values(["modelo", "precisión"]), use_container_width=True, hide_index=True)
        st.caption(f"Soporte de glosario en la versión instalada: {bench['glossary_support_instalado']}")
        for k, v in bench["comparaciones"].items():
            with st.expander(k):
                st.json(v)
    else:
        st.info("Aún no hay benchmark de GPU. Córrelo en Colab: notebooks/colab_gpu.ipynb")

    st.header("Capa 2 · Registro")
    if ev:
        for lvl, e in ev["levels"].items():
            st.subheader("Transcripción de referencia (ASR perfecto)" if lvl == "REF" else f"Audio {lvl} · {e['fuente']}")
            if "wer" in e:
                st.caption(f"WER medio {e['wer']['media']:.3f} (IC95 {e['wer']['ic95'][0]:.3f}–{e['wer']['ic95'][1]:.3f})")
            cols = st.columns(len(e["por_metodo"]))
            for col, (m, agg) in zip(cols, e["por_metodo"].items()):
                with col:
                    st.markdown(f"**{m}**")
                    if not agg:
                        st.caption("sin resultados (¿falta caché/API key?)")
                        continue
                    st.dataframe(pd.DataFrame([{"campo": f, "exactitud": pct(v["exactitud"]), "IC95 Wilson": ci(v["ic95_wilson"]),
                                                "n": v["n"]} for f, v in agg["por_campo"].items()]),
                                 hide_index=True, use_container_width=True)
                    st.caption(f"F1 macro nivel_ingles: {agg.get('macro_f1_nivel_ingles', float('nan')):.2f} · "
                               f"errores absolutos: {json.dumps({k: round(v, 2) for k, v in agg['error_absoluto_medio'].items()})}")
                    st.caption(f"Verificación: {agg['verificacion']}")
                    if "ablacion_normalizacion_numerica" in agg:
                        st.caption(f"Sin la capa numérica: {agg['ablacion_normalizacion_numerica']}")
        wv = ev.get("wer_vs_registro")
        if wv and wv["tabla"]:
            st.subheader("¿El WER predice si el registro sirve?")
            df = pd.DataFrame(wv["tabla"])
            fig = go.Figure()
            for m in df["metodo"].unique():
                d = df[df["metodo"] == m]
                fig.add_trace(go.Scatter(x=d["wer"], y=d["campos_ok"], mode="markers+text", text=d["clip"],
                                         textposition="top center", name=m))
            fig.update_layout(xaxis_title="WER", yaxis_title="campos correctos (de 6)", height=380)
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            for col, key, title in ((c1, "wer_alto_registro_perfecto", "WER alto, registro perfecto"),
                                    (c2, "wer_bajo_registro_roto", "WER bajo, registro roto")):
                col.markdown(f"**{title}**")
                if wv[key]:
                    col.json(wv[key])
                else:
                    col.caption("No hay un clip así en estos datos.")
    else:
        st.info("Corre `python scripts/evaluate.py` para generar los resultados.")

# ---------------------------------------------------------------- 4
with tab_order:
    ev = load_json(f"order_eval{suffix}.json")
    st.caption(f"Dominio: {cfg['order']['domain']}")
    if not ev:
        st.info("Corre `python scripts/evaluate.py`.")
    else:
        levels = [l for l in ev["levels"] if l != "REF"]
        methods = sorted({m for e in ev["levels"].values() for m, a in e["por_metodo"].items() if a})
        st.subheader("Escalera de degradación: WER y exactitud de campos")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=levels, y=[ev["levels"][l]["wer"]["media"] for l in levels], name="WER", yaxis="y2",
                                 line=dict(dash="dot")))
        for m in methods:
            fig.add_trace(go.Scatter(x=levels, y=[(ev["levels"][l]["por_metodo"].get(m) or {}).get("global", {}).get("exactitud")
                                                  for l in levels], name=f"exactitud campos ({m})"))
        fig.update_layout(yaxis=dict(title="exactitud de campos", range=[0, 1.05]),
                          yaxis2=dict(title="WER", overlaying="y", side="right"), height=380)
        st.plotly_chart(fig, use_container_width=True)
        build = load_json(f"ladder_build_{group}_order.json")
        if build:
            with st.expander("Cómo se construyó cada nivel (SNR medida)"):
                st.json(build)
        st.subheader("Validación aritmética: atrapados vs perdidos")
        rows = []
        for l, e in ev["levels"].items():
            for m, agg in e["por_metodo"].items():
                if agg and agg.get("validacion"):
                    v = agg["validacion"]
                    rows.append({"nivel": l, "método": m, "atrapados": v["atrapados"], "perdidos": v["perdidos"],
                                 "falsas alarmas": v["falsas_alarmas"], "correctos sin alarma": v["correctos_sin_alarma"],
                                 "no chequeables": v["no_chequeables"],
                                 "diagnóstico acierta": f"{v['diagnostico_acierta']}/{v['diagnostico_n']}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        lv = st.selectbox("Detalle por llamada en el nivel", list(ev["levels"]))
        m = st.selectbox("Método", methods)
        agg = ev["levels"][lv]["por_metodo"].get(m)
        if agg:
            st.dataframe(pd.DataFrame(agg["validacion"]["detalle"]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 5
with tab_test:
    st.header("Probador")
    st.caption("Sube un clip: se llama a `process_audio()`, la misma función del pipeline. Sin GPU irá lento (CPU).")
    schema = st.radio("Esquema", ["interview", "order"], horizontal=True, key="t_schema")
    method = st.radio("Extracción", cfg["extraction"]["methods"], horizontal=True)
    model = st.selectbox("Modelo ASR", [cfg["asr"]["model"], "medium", "small", "base"])
    up = st.file_uploader("Audio", type=[e.strip(".") for e in cfg["audio_ext"]])
    if up and st.button("Procesar"):
        if up.size > cfg["app"]["max_upload_mb"] * 2**20:
            st.error("Archivo demasiado grande.")
        else:
            from vozregistro.engine import process_audio  # import perezoso: la app arranca sin cargar ASR
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as fh:
                fh.write(up.getbuffer())
            with st.spinner("Transcribiendo y extrayendo…"):
                r = process_audio(fh.name, schema=schema, method=method, model=model)
            if r["error"]:
                st.error(r["error"])
            c1, c2 = st.columns(2)
            c1.metric("Dispositivo (leído del modelo)", r["device"] or "—")
            c2.metric("Tiempo total (s)", r["elapsed_s"])
            st.text_area("Transcripción", r["transcript"] or "", height=200)
            if r["fields"]:
                st.dataframe(pd.DataFrame([{"campo": k, "valor": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v,
                                            "estado": (r["verified"] or {}).get(k, "—")} for k, v in r["fields"].items()]),
                             hide_index=True, use_container_width=True)
            if r.get("validation"):
                v = r["validation"]
                (st.success if v["cuadra"] else st.error)(
                    f"Esperado {v.get('esperado')} · dicho {v.get('dicho')} · diagnóstico: {v['diagnostico']}")
                st.json(v)
            with st.expander("Resultado completo"):
                st.json({k: v for k, v in r.items() if k not in ("words",)})
