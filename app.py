import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# CONFIGURACIÓN CENTRALIZADA
VECTORSTORE_DIR = "vectorstore"

API_KEY = os.getenv("GITHUB_TOKEN")
LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.1,  # Pequeño margen para evitar respuestas muy rígidas
    "base_url": "https://models.inference.ai.azure.com",
    "api_key": API_KEY
}
EMBEDDINGS_CONFIG = {
    "model": "text-embedding-3-small",
    "base_url": "https://models.inference.ai.azure.com",
    "api_key": API_KEY
}

# Validar que existe la clave API
if not API_KEY:
    st.error("❌ Error: GITHUB_TOKEN no encontrado en variables de entorno.")
    st.stop()

st.set_page_config(
    page_title="Asistente Tributario - Ruiz Salazar",
    page_icon="⚖️",
    layout="wide"
)

# ── PROMPT ──────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente de consultas tributarias del estudio Ruiz Salazar Tributaria.
Respondes preguntas sobre normativa tributaria chilena basándote en los fragmentos normativos entregados.

Reglas:
- Analiza los fragmentos entregados y extrae la información relevante para responder.
- Cita el artículo, circular o resolución exacta cuando esté disponible en el contexto.
- Si puedes inferir una respuesta parcial, entrégala indicando sus limitaciones.
- Solo si el contexto es completamente irrelevante para la pregunta, responde:
  "La normativa disponible no cubre esta consulta. Consulte directamente con un profesional."
- No inventes normas ni artículos que no aparezcan en el contexto.
- Responde en español formal.

Contexto normativo recuperado:
{context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

# FUNCIONES AUXILIARES
def validar_vectorstore() -> bool:
    """Verifica que el directorio del vectorstore existe."""
    if not os.path.exists(VECTORSTORE_DIR):
        st.error(f"❌ Error: El directorio '{VECTORSTORE_DIR}' no existe.")
        return False
    return True

# ── CARGA DEL MODELO (cached) ────────────────────────────────
@st.cache_resource
def cargar_chain():
    """
    Carga la cadena de procesamiento con embeddings, vectorstore y LLM.
    Se cachea para evitar recargas innecesarias.
    """
    if not validar_vectorstore():
        st.stop()
        
    try:
        embeddings = OpenAIEmbeddings(**EMBEDDINGS_CONFIG)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        llm = ChatOpenAI(**LLM_CONFIG)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Valor por defecto, se sobrescribe dinámicamente
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"❌ Error al cargar la cadena de procesamiento: {str(e)}")
        st.stop()

# ── INTERFAZ ─────────────────────────────────────────────────
st.title("⚖️ Asistente Tributario")
st.caption("Ruiz Salazar Tributaria — consulta normativa con citación de fuente")

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

k_docs = st.sidebar.slider(
    "Fragmentos a recuperar (k):",
    min_value=1,
    max_value=8,
    value=4,
    help="Más fragmentos = respuestas más completas pero más largas"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Corpus disponible**")
st.sidebar.markdown("""
- DL 824 — Ley de Renta
- DL 825 — Ley de IVA
- Circulares SII
- Código Tributario
""")

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Este asistente no reemplaza asesoría tributaria profesional. "
    "Las respuestas son orientativas y deben ser validadas por un contador o abogado tributario."
)

# INICIALIZAR SESSION STATE
# Historial en session state
if "historial" not in st.session_state:
    st.session_state.historial = []

# Layout principal - DOS COLUMNAS
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("💬 Consulta")

    consulta = st.text_area(
        "Ingresa tu consulta tributaria:",
        height=120,
        placeholder="Ej: ¿Qué actividades están exentas de IVA según el artículo 12 del DL 825?"
    )

    consultar_btn = st.button("🔍 Consultar", type="primary", use_container_width=True)

    # Procesar consulta
    if consultar_btn and consulta.strip():
        with st.spinner("Buscando en la normativa..."):
            try:
                chain = cargar_chain()
                # Actualizar k dinámicamente
                chain.retriever.search_kwargs["k"] = k_docs
                resultado = chain.invoke({"query": consulta})

                respuesta = resultado["result"]
                fuentes = resultado["source_documents"]

                # Guardar en historial
                st.session_state.historial.append({
                    "consulta": consulta,
                    "respuesta": respuesta,
                    "fuentes": fuentes,
                    "k_usado": k_docs  # Registrar cuántos fragmentos se usaron
                })

                st.success(f"✅ Procesada con {len(fuentes)} fragmentos relevantes")

            except Exception as e:
                st.error(f"❌ Error al procesar la consulta: {str(e)}")

    elif consultar_btn:
        st.warning("⚠️ Ingresa una consulta antes de continuar.")

    # MOSTRAR ÚLTIMA RESPUESTA
    if st.session_state.historial:
        ultimo = st.session_state.historial[-1]

        st.subheader("📋 Respuesta")
        st.markdown(ultimo["respuesta"])

        st.subheader(f"📎 Fuentes ({len(ultimo['fuentes'])} fragmentos)")
        
        for i, doc in enumerate(ultimo["fuentes"], 1):
            fuente = doc.metadata.get("source", "desconocida")
            pagina = doc.metadata.get("page", "?")
            nombre = os.path.basename(fuente)
            
            with st.expander(f"Fragmento {i} — {nombre}, pág. {pagina}"):
                st.caption(doc.page_content)


# COLUMNA 2 - HISTORIAL
with col2:
    st.subheader("🕓 Historial")
    
    if not st.session_state.historial:
        st.info("Las consultas aparecerán aquí.")
    else:
        # Mostrar contador de consultas
        st.metric("Total de consultas", len(st.session_state.historial))
        st.markdown("---")
        
        # Mostrar historial en orden inverso (más reciente primero)
        for i, item in enumerate(reversed(st.session_state.historial), 1):
            numero_consulta = len(st.session_state.historial) - i + 1
            titulo = f"**{numero_consulta}.** {item['consulta'][:50]}..."
            
            with st.expander(titulo):
                st.markdown(f"**Pregunta:** {item['consulta']}")
                st.markdown(f"**Respuesta:** {item['respuesta']}")
                st.caption(f"📊 Fragmentos usados: {item.get('k_usado', 4)}")
    
    # Botón para limpiar historial
    st.markdown("---")
    if st.session_state.historial:
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.historial = []
                st.rerun()
        
        with col_export:
            if st.button("📥 Exportar", use_container_width=True, disabled=True):
                st.info("Función en desarrollo")
