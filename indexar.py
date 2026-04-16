import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Directorios para almacenar documentos y vectorstore
DOCS_DIR = "docs"
VECTORSTORE_DIR = "vectorstore"

def cargar_pdfs(directorio: str):
    """
    Carga todos los archivos PDF de un directorio especificado.
    
    Args:
        directorio (str): Ruta del directorio que contiene los PDFs
        
    Returns:
        list: Lista de documentos cargados desde los PDFs
    """
    docs = []
    # Buscar todos los archivos .pdf en el directorio
    pdfs = list(Path(directorio).glob("*.pdf"))

    # Validar que existan PDFs en el directorio
    if not pdfs:
        print(f"No se encontraron PDFs en '{directorio}/'")
        return []
    
    # Iterar sobre cada PDF y cargarlo
    for pdf_path in pdfs:
        print(f"  Cargando: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    # Informar cantidad de páginas cargadas
    print(f"  Total páginas cargadas: {len(docs)}")
    return docs

def contar_tokens(text: str) -> int:
    """
    Cuenta la cantidad de tokens en un texto usando el tokenizador de OpenAI.
    
    Args:
        text (str): Texto a tokenizar
        
    Returns:
        int: Cantidad de tokens en el texto
    """
    # Obtener el tokenizador estándar de OpenAI
    enc = tiktoken.get_encoding("cl100k_base")
    # Codificar el texto y contar los tokens generados
    return len(enc.encode(text))

def crear_chunks(docs):
    """
    Divide los documentos en fragmentos (chunks) de tamaño manejable.
    Utiliza conteo de tokens real en lugar de caracteres.
    
    Args:
        docs (list): Lista de documentos a dividir
        
    Returns:
        list: Lista de fragmentos generados
    """
    # Crear un divisor recursivo que respeta la estructura del texto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,                # Tamaño máximo de cada chunk en tokens # ahora en tokens reales, no caracteres
        chunk_overlap=80,              # Solapamiento entre chunks para continuidad
        length_function=contar_tokens, # Usar función personalizada para contar tokens
        separators=["\n\n", "\n", ".", " "] # Intentar dividir por estos separadores (en orden de preferencia)
    )
    # Aplicar la división a los documentos y retornar los chunks
    return splitter.split_documents(docs)

def crear_vectorstore_en_batches(chunks, embeddings, batch_size=50):
    """
    Crea un vectorstore procesando los chunks en lotes para evitar
    sobrecargar la API de embeddings.
    
    Args:
        chunks (list): Lista de fragmentos a vectorizar
        embeddings: Modelo de embeddings a utilizar
        batch_size (int): Cantidad de chunks a procesar por lote (default: 50)
        
    Returns:
        FAISS: Vectorstore con todos los embeddings almacenados
    """
    print(f"  Procesando {len(chunks)} chunks en batches de {batch_size}...")
    
    vectorstore = None
    # Procesar los chunks en lotes para no saturar la API
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Calcular número de lote actual y total
        num_batch = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"  Batch {num_batch}/{total_batches} ({len(batch)} chunks)")

        # Crear vectorstore con el primer lote o agregar nuevos vectores a los existentes
        if vectorstore is None:
            # Primera iteración: crear el vectorstore inicial
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Iteraciones posteriores: crear vectorstore del lote y fusionarlo con el existente
            batch_vs = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_vs)
    
    return vectorstore

def indexar():
    """
    Orquesta todo el proceso de indexación: carga PDFs, crea chunks,
    genera embeddings y almacena el vectorstore.
    """
    print("=== Indexando corpus normativo ===\n")

    # Paso 1: Cargar todos los PDFs del directorio
    print("1. Cargando PDFs...")
    docs = cargar_pdfs(DOCS_DIR)
    if not docs:
        return

    # Paso 2: Dividir documentos en chunks manejables
    print("\n2. Generando chunks...")
    chunks = crear_chunks(docs)

    # Paso 3: Crear embeddings y construir el vectorstore
    print("\n3. Generando embeddings y construyendo vectorstore...")
    # Configurar el modelo de embeddings de OpenAI (accesible a través de Azure)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",          # Modelo más eficiente
        base_url="https://models.inference.ai.azure.com", # Usar API de Azure
        api_key=os.getenv("GITHUB_TOKEN")        # Token desde variables de entorno
    )

    # Crear vectorstore procesando en lotes para eficiencia
    vectorstore = crear_vectorstore_en_batches(chunks, embeddings, batch_size=50)
    # Guardar el vectorstore en disco para uso posterior
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"  Vectorstore guardado en '{VECTORSTORE_DIR}/'")
    print("\n=== Indexación completada ===")

# Ejecutar el programa principal
if __name__ == "__main__":
    indexar()
