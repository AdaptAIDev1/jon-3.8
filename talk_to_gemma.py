#!/usr/bin/env python3
# JON voice chat: mic → faster-whisper → Ollama → pico2wave (+ RAG “dictionary” context)
# Features: VAD end-on-pause, echo guard, barge-in, markdown cleanup for TTS, FAISS retrieval
# Created by Jeffrey Chery - AdaptAI
# Edited and enhancements by Carlos Jofre
#
import io, wave, os, sys, time, json, queue, threading, subprocess, re
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
from faster_whisper import WhisperModel

# ===== RAG imports =====
# If you're on newer LangChain splits:
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ---------- CONFIG ----------
# Ollama
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2b-gpu")   # chat model

# Whisper (Jetson: CPU int8 is stable)
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "tiny.en")
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"

# Audio I/O
SAMPLE_RATE       = 16000
FRAME_MS          = 20                  # 10/20/30 valid for webrtcvad
MIC_DEVICE_INDEX  = None                # None = default input; set a number to force a device
OUTPUT_SINK       = os.getenv("OUTPUT_SINK", None)  # e.g. "alsa_output.platform-3510000.hda.hdmi-stereo"

# VAD / utterance control
VAD_MODE          = 3                   # 0..3 (3 = most aggressive)
SILENCE_END_MS    = 800                 # end utterance after ~0.8s silence
MIN_UTTER_MS      = 1200                # require >= 1.2s speech
MAX_UTTER_MS      = 20000               # cap at 20s

# Start gates (reduce false triggers)
START_SPEECH_FRAMES = 12                # need N consecutive speech frames (~240ms at 20ms)
RMS_START_THRESH    = 0.025             # energy floor to start (0.015–0.035 typical)

# Echo / barge-in control
BARGE_RMS        = 0.060                # must speak clearly to interrupt TTS
BARGE_FRAMES     = 4                    # ~80ms at 20ms frames
COOLDOWN_MS      = 200                  # ignore residual ring for 200ms after TTS stops

# Transcript filters
STOPWORDS        = {"you", "ya", "yup", "uh", "um", "hmm"}
MIN_TOKENS       = 4                    # ignore transcripts shorter than this

# ---------- RAG CONFIG ----------
DATA_DIR     = os.getenv("RAG_DATA_DIR", "./data")            # where your .txt files live
INDEX_DIR    = os.getenv("RAG_INDEX_DIR", "./faiss_index")    # where FAISS persists
EMBED_MODEL  = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")  # embedding model in Ollama
TOP_K        = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE    = float(os.getenv("RAG_MIN_SCORE", "0.30"))
CHUNK_SIZE   = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# System guidance: “dictionary” behavior
SYSTEM_PROMPT = (
    "Respond in plain text only (no Markdown, no asterisks or formatting). "
    "Be concise, friendly, and direct. You are running on a Jetson."
)
RAG_SYSTEM_GUARD = (
    "You must answer ONLY using the provided context. "
    "If the answer is not fully contained in the context, say exactly: "
    "\"I don't have that in my knowledge base.\" Do not guess."
)

# ----------------------------

print("Loading Whisper…")
whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ----- Globals for TTS/echo handling -----
tts_stop_flag = threading.Event()
tts_proc_gen = None
tts_proc_play = None
tts_active = False
last_tts_stop_ms = 0
barge_count = 0

# ----- Audio queue -----
audio_q = queue.Queue()

def sd_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))

def rms16(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(arr))) / 32768.0)

def pcm_to_wav_bytes(pcm: bytes) -> io.BytesIO:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    bio.seek(0)
    return bio

def transcribe_pcm(pcm: bytes) -> str:
    wav_io = pcm_to_wav_bytes(pcm)
    segments, info = whisper_model.transcribe(
        wav_io,
        beam_size=3,
        vad_filter=False,                 # we do our own VAD
        no_speech_threshold=0.6,          # be conservative
        condition_on_previous_text=False,
        temperature=0.0
    )
    return " ".join(s.text.strip() for s in segments if s.text).strip()

def valid_transcript(text: str) -> bool:
    if not text:
        return False
    toks = text.lower().split()
    if len(toks) < MIN_TOKENS:
        return False
    if len(toks) == 1 and toks[0] in STOPWORDS:
        return False
    return True

def clean_text_for_tts(text: str) -> str:
    # Strip common Markdown and weird formatting so TTS doesn't say "asterisk asterisk"
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===== RAG: index load/build =====
def load_text_documents(data_dir: str):
    docs = []
    if not os.path.isdir(data_dir):
        return docs
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".txt"):
            path = os.path.join(data_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(Document(page_content=f.read(), metadata={"source": fname}))
            except Exception as e:
                print(f"[RAG] Skipping {fname}: {e}")
    return docs

def build_faiss_index(docs, embeddings):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Chunked {len(docs)} docs into {len(chunks)} chunks.")
    return FAISS.from_documents(chunks, embedding=embeddings)

def load_or_create_index():
    print("[RAG] Initializing embeddings:", EMBED_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    if os.path.isdir(INDEX_DIR) and os.path.isfile(os.path.join(INDEX_DIR, "index.faiss")):
        print("[RAG] Loading FAISS index from", INDEX_DIR)
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return db
    print("[RAG] Building FAISS index from", DATA_DIR)
    docs = load_text_documents(DATA_DIR)
    if not docs:
        print("[RAG] No .txt files found in ./data; dictionary will be empty.")
        # Create an empty index (rare) by seeding a dummy doc; or return None to disable RAG
        return None
    db = build_faiss_index(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    print("[RAG] Saved FAISS index to", INDEX_DIR)
    return db

RAG_DB = load_or_create_index()

def rag_retrieve(query: str):
    """Return list[(doc, score)] filtered by MIN_SCORE."""
    if RAG_DB is None:
        return []
    try:
        results = RAG_DB.similarity_search_with_score(query, k=TOP_K)
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return []
    filtered = [(doc, score) for doc, score in results if (score is None or score >= MIN_SCORE)]
    return filtered

def rag_context_block(hits):
    if not hits:
        return ""
    lines = []
    for i, (doc, score) in enumerate(hits, 1):
        src = doc.metadata.get("source", "unknown")
        sc = f"{score:.3f}" if isinstance(score, (float, int)) else "n/a"
        lines.append(f"[{i}] (score={sc}, source={src})\n{doc.page_content}")
    return "\n\n".join(lines)

# ===== Chat to Ollama (with RAG) =====
def ask_ollama_with_rag(user_prompt: str) -> str:
    """
    - Retrieve RAG context.
    - If none, respond with strict dictionary message.
    - Else, inject guard + context and ask the chat model.
    """
    hits = rag_retrieve(user_prompt)
    if not hits:
        return "I don't have that in my knowledge base."

    context = rag_context_block(hits)

    # Build a per-turn system message that enforces dictionary behavior
    turn_system = {"role": "system", "content": RAG_SYSTEM_GUARD}
    turn_context = {
        "role": "system",
        "content": f"Context:\n{context}"
    }

    messages = history + [turn_system, turn_context, {"role": "user", "content": user_prompt}]
    url = f"{OLLAMA_URL}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    reply = r.json()["message"]["content"].strip()

    # Keep a minimal history (without dumping long context into running history)
    history.extend([
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": reply}
    ])
    return reply

def kill_tts_subprocs():
    global tts_proc_gen, tts_proc_play
    for p in (tts_proc_gen, tts_proc_play):
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    tts_proc_gen = None
    tts_proc_play = None

def speak_tts(text: str):
    global tts_proc_gen, tts_proc_play, tts_active, last_tts_stop_ms
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " ").strip())
    for s in [x for x in sentences if x]:
        if tts_stop_flag.is_set():
            break
        wav = "/tmp/jon_tts.wav"
        tts_proc_gen = subprocess.Popen(["pico2wave", "-w", wav, s],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        tts_proc_gen.wait()
        if tts_stop_flag.is_set():
            break
        cmd = ["paplay", wav] if not OUTPUT_SINK else ["paplay", "--device", OUTPUT_SINK, wav]
        tts_proc_play = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        while tts_proc_play.poll() is None:
            if tts_stop_flag.is_set():
                break
            time.sleep(0.02)
    kill_tts_subprocs()
    tts_active = False
    last_tts_stop_ms = int(time.time() * 1000)

def start_tts(text: str):
    global tts_active
    stop_tts()
    tts_stop_flag.clear()
    tts_active = True
    threading.Thread(target=speak_tts, args=(clean_text_for_tts(text),), daemon=True).start()

def stop_tts():
    global tts_active, last_tts_stop_ms
    tts_stop_flag.set()
    kill_tts_subprocs()
    tts_active = False
    last_tts_stop_ms = int(time.time() * 1000)

def main():
    global barge_count, tts_active
    frame_bytes = int(SAMPLE_RATE * (FRAME_MS / 1000.0)) * 2
    vad = webrtcvad.Vad(VAD_MODE)

    buf = bytearray()
    prestart_buf = bytearray()
    speaking = False
    silence_ms = 0
    spoken_ms = 0

    print("🎤 JON is listening. Speak naturally; I’ll stop on pause. You can interrupt me while I'm talking.")
    with sd.RawInputStream(samplerate=SAMPLE_RATE,
                           blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
                           dtype="int16", channels=1,
                           device=MIC_DEVICE_INDEX, callback=sd_callback):
        try:
            while True:
                chunk = audio_q.get()
                for i in range(0, len(chunk), frame_bytes):
                    fr = chunk[i:i + frame_bytes]
                    if len(fr) < frame_bytes:
                        continue

                    # keep a small lookback (so we don't clip initial consonants)
                    prestart_buf.extend(fr)
                    if len(prestart_buf) > frame_bytes * 10:  # ~200ms lookback
                        prestart_buf = prestart_buf[-frame_bytes * 10:]

                    energy = rms16(fr)
                    try:
                        is_speech = vad.is_speech(fr, SAMPLE_RATE)
                    except Exception:
                        is_speech = False

                    now_ms = int(time.time() * 1000)

                    # --- Echo guard: while TTS is active, ignore mic unless strong barge-in ---
                    if tts_active:
                        if is_speech and energy >= BARGE_RMS:
                            barge_count += 1
                            if barge_count >= BARGE_FRAMES:
                                stop_tts()  # kill TTS immediately
                                barge_count = 0
                                prestart_buf.clear()
                                buf.clear()
                                speaking = False
                                silence_ms = 0
                                spoken_ms = 0
                        else:
                            barge_count = 0
                        continue  # do not buffer AI's own voice

                    # brief cooldown right after TTS stops (ignore speaker ring)
                    if now_ms - last_tts_stop_ms < COOLDOWN_MS:
                        continue

                    # --- Start / continue / end utterance ---
                    if not speaking:
                        # require consecutive speech frames + energy gate
                        if is_speech and energy >= RMS_START_THRESH:
                            barge_count += 1  # reuse counter for start gate
                        else:
                            barge_count = 0
                        if barge_count >= START_SPEECH_FRAMES:
                            speaking = True
                            silence_ms = 0
                            spoken_ms = 0
                            buf.extend(prestart_buf)
                            prestart_buf.clear()
                    else:
                        buf.extend(fr)
                        spoken_ms += FRAME_MS
                        if is_speech:
                            silence_ms = 0
                        else:
                            silence_ms += FRAME_MS

                        should_end = ((silence_ms >= SILENCE_END_MS and spoken_ms >= MIN_UTTER_MS) or
                                      (spoken_ms >= MAX_UTTER_MS))
                        if should_end:
                            pcm = bytes(buf)
                            buf.clear()
                            speaking = False
                            silence_ms = 0
                            spoken_ms = 0
                            barge_count = 0

                            text = transcribe_pcm(pcm)
                            if valid_transcript(text):
                                print(f"\n🗣️ You: {text}")
                                try:
                                    reply = ask_ollama_with_rag(text)
                                    print(f"🤖 JON: {reply}\n")
                                    start_tts(reply)
                                except Exception as e:
                                    print(f"[Ollama error] {e}")
                            # else: ignore tiny/noisy transcript
        except KeyboardInterrupt:
            pass
        finally:
            stop_tts()

if __name__ == "__main__":
    main()
