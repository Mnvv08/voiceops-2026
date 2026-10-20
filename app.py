"""Live demo UI: record your voice, hear the agent reply.

    streamlit run app.py
"""
import tempfile

import streamlit as st

from src.agent import Agent
from src.kb import KnowledgeBase
from src.voice import Voice

st.set_page_config(page_title="ZentroMart Voice Support", page_icon="🎧")
st.title("ZentroMart Voice Support")
st.caption("Speak to the agent. Speech is handled by Gnani STT/TTS; answers come from ZentroMart's documents.")


@st.cache_resource
def load():
    return KnowledgeBase(), Voice()


kb, voice = load()
if "agent" not in st.session_state:
    st.session_state.agent = Agent("chat", kb)
    st.session_state.log = []
    st.session_state.n = 0

if st.button("New call"):
    st.session_state.agent = Agent("chat", kb)
    st.session_state.log, st.session_state.n = [], 0

audio = st.audio_input(f"Record your question (turn {st.session_state.n + 1})", key=f"rec{st.session_state.n}")
if audio is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio.getvalue())
        in_path = f.name
    out_path = in_path.replace(".wav", "_reply.wav")
    with st.spinner("Listening and thinking..."):
        heard, reply = voice.turn(in_path, st.session_state.agent, out_path)
    st.session_state.log.append((heard, reply, out_path, st.session_state.agent.last_sources))
    st.session_state.n += 1
    st.rerun()

for i, (heard, reply, wav, sources) in enumerate(st.session_state.log):
    st.chat_message("user").write(heard)
    with st.chat_message("assistant"):
        st.write(reply)
        st.audio(wav, autoplay=(i == len(st.session_state.log) - 1))
        with st.expander("Sources used"):
            for c in sources:
                st.markdown(f"**{c['doc']} · {c['section_id']}**")
                st.text(c["text"][:400])
    if i == len(st.session_state.log) - 1 and st.session_state.agent.tools:
        with st.expander("Tool calls"):
            st.json(st.session_state.agent.tools.log)
