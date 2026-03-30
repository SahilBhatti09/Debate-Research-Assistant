import os
import requests
import tempfile
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma as ChromaUpdated 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# LOAD ENV
# =========================
load_dotenv()
os.environ["USER_AGENT"] = "DebateAssistant/1.0"

# =========================
# CONFIG
# =========================
DB_PATH = "debate_db"

GOOGLE_DRIVE_PDF_LINKS = [
    "https://drive.google.com/file/d/1KJDPUoa4VV9v3HzkaO1ct49Mx7PukMBb/view?usp=drive_link",
    "https://drive.google.com/file/d/1QHPPwU71k3Vjv6fbqSeEjiMdt_HWcukQ/view?usp=drive_link",
    "https://drive.google.com/file/d/1f-54fLLXfX72Hi-eotgnxgLDv37AFtfS/view?usp=drive_link"
]

JSONL_FILE_LINK = "https://drive.google.com/file/d/12Dk93fVAbwhBGP0I5cFIW3gu5m7w7O3G/view?usp=drive_link"

PREDEFINED_LINKS = [
    "https://debateexperts.com/debate-structure-key-components-and-formats/",
    "https://debateproject.eu/chapter-4-1-british-parliamentary-bp-debate-format/",
    "https://cimsa.fk.ugm.ac.id/medebate/asian-parliamentary-debate/"
]

LOCAL_JSONL_FILE = "speeches.jsonl"

# =========================
# UTILITIES
# =========================
def download_file_from_drive(drive_link, local_file):
    try:
        file_id = drive_link.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        r = requests.get(download_url)
        with open(local_file, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded: {local_file}")
        return local_file
    except Exception as e:
        print(f"Failed to download {drive_link}: {e}")
        return None

# =========================
# LOAD DOCUMENTS
# =========================
def load_pdf_from_drive(link):
    try:
        local_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        download_file_from_drive(link, local_file)
        loader = PyMuPDFLoader(local_file)
        return loader.load()
    except Exception as e:
        print(f"PDF Load Error ({link}): {e}")
        return []

def load_jsonl_speeches(file_path):
    if not os.path.exists(file_path):
        download_file_from_drive(JSONL_FILE_LINK, file_path)
    docs = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                content = f"Motion: {item['motion']}\nRole: {item['speaker_role']}\nSpeech: {item['speech_text']}"
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "debate_id": item.get("debate_id"),
                        "speech_id": item.get("speech_id"),
                        "team_label": item.get("team_label"),
                        "speaker_role": item.get("speaker_role")
                    }
                ))
            except json.JSONDecodeError as e:
                print("JSONL line error:", e)
    return docs

def load_predefined_links(links):
    docs = []
    for link in links:
        try:
            loader = WebBaseLoader(link)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Link Load Error: {link} -> {e}")
    return docs

# =========================
# VECTORSTORE
# =========================
def create_vectorstore(documents, persist_dir=DB_PATH):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = ChromaUpdated.from_documents(split_docs, embedding, persist_directory=persist_dir)
    return vectordb

def get_vectorstore(rebuild=False):
    if rebuild or not os.path.exists(DB_PATH):
        print("Building vector database...")
        pdf_docs = []
        for link in GOOGLE_DRIVE_PDF_LINKS:
            pdf_docs.extend(load_pdf_from_drive(link))
        web_docs = load_predefined_links(PREDEFINED_LINKS)
        json_docs = load_jsonl_speeches(LOCAL_JSONL_FILE)
        all_docs = json_docs + pdf_docs + web_docs  # JSONL prioritized
        vectordb = create_vectorstore(all_docs)
        print("✅ Vector database built.")
        return vectordb
    else:
        print("Loading existing vector database...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = ChromaUpdated(persist_directory=DB_PATH, embedding_function=embedding)
        return vectordb

# =========================
# LLM + SEARCH + PROMPT
# =========================
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # more chunks

llm = ChatMistralAI(model="mistral-small-2506", temperature=0.3)
web_search = TavilySearch(max_results=3)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a Debate & Research Assistant trained for ANY topic.
- PRIORITY: JSONL speeches > PDF > Links > Web
- RESPONSE STRUCTURE: DIRECT → ARGUMENTS → EXAMPLES → COUNTER → CONCLUSION
- Tone: Sharp, Strategic, Persuasive
- Avoid generic or repeated arguments
- If context is weak: use web search and reason logically
- Include ethical, current, or abstract debate issues
- Avoid repeating standard openings like " Ladies and Gentlemen"    
- These rules can be overriden if user wants different structure.
"""),
    ("human",
     """Context:
{context}

Question:
{question}
""")
])

# =========================
# PIPELINE
# =========================
def generate_answer(question):
    try:
        docs = retriever.invoke(question)  
        context = "\n\n".join(doc.page_content for doc in docs)
        context = "[DOCUMENT + LINKS]\n" + context
    except Exception as e:
        print("Retriever Error:", e)
        context = ""

    # Fallback web search if context weak
    if len(context.strip()) < 300:
        try:
            web_results = web_search.invoke(question)
            # Extract text from results
            if isinstance(web_results, list):
                web_text = "\n".join([r.get('content', '') for r in web_results])
            elif isinstance(web_results, dict):
                web_text = web_results.get('content', str(web_results))
            else:
                web_text = str(web_results)
            context += "\n\n[WEB SEARCH]\n" + web_text
                        
        except Exception as e:
            print("Web Search Error:", e)

    if not context.strip():
        context = "No strong context found."

    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)
    return response.content

# =========================
# CHAT LOOP
# =========================
if __name__ == "__main__":
    print("\nDebate & Research Assistant (Generic)")
    print("Type '0' to exit")
    print("--------------------------------")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["0", "exit", "quit"]:
            print("\nGoodbye!")
            break
        answer = generate_answer(query)
        print(f"\nAI:\n{answer}\n")