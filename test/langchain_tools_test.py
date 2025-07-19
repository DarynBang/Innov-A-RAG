import logging
import sys

# === Create a fresh log file every time ===
LOG_FILE_PATH = "assistant.log"

# Remove any existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up new logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),  # Overwrite each run
        logging.StreamHandler(sys.stdout),  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logging.info("=== Starting Company & Patent Assistant ===")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logging.info("Loaded HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2")

# Create directories for Chroma DB
os.makedirs("chroma_company", exist_ok=True)
os.makedirs("chroma_patent", exist_ok=True)
logging.info("Created Chroma directories")

company_data = [
    {"name": "TechNova", "summary": "TechNova specializes in AI-driven robotics, autonomous systems, and industrial automation for manufacturing, logistics, and smart cities."},
    {"name": "BioHealth", "summary": "BioHealth develops genomics-based precision medicine solutions using AI-powered DNA sequencing and rare disease diagnostics."},
    {"name": "EcoEnergy", "summary": "EcoEnergy provides green energy technology, including AI-optimized solar farms, smart grids, and energy storage."},
    {"name": "FinSecure", "summary": "FinSecure focuses on AI-based financial fraud detection, real-time risk analytics, and secure payment processing systems."},
    {"name": "AgriSmart", "summary": "AgriSmart integrates AI and IoT for smart agriculture, including crop monitoring, yield prediction, and automated irrigation."},
    {"name": "MedVision", "summary": "MedVision creates AI-assisted medical imaging software for diagnostics, surgical guidance, and remote healthcare solutions."},
    {"name": "EduNext", "summary": "EduNext provides AI-driven adaptive learning platforms, personalized curriculum design, and student performance analytics."},
    {"name": "AutoDrive", "summary": "AutoDrive builds autonomous vehicle software stacks, including AI-based perception, route planning, and safety systems."},
    {"name": "CyberShield", "summary": "CyberShield develops advanced cybersecurity systems using AI for threat detection, anomaly monitoring, and response automation."},
    {"name": "RetailX", "summary": "RetailX applies AI to optimize supply chains, customer behavior analysis, dynamic pricing, and personalized marketing in retail."}
]

patent_data = [
    # TechNova patents
    {"id": "P001", "company": "TechNova", "abstract": "AI-based robotic arm control system using computer vision and reinforcement learning."},
    {"id": "P002", "company": "TechNova", "abstract": "Autonomous warehouse navigation for industrial robots with real-time sensor fusion."},
    {"id": "P003", "company": "TechNova", "abstract": "Edge computing framework for decentralized robot learning in smart factories."},

    # BioHealth patents
    {"id": "P004", "company": "BioHealth", "abstract": "AI-powered genomic sequencing technique for faster DNA analysis."},
    {"id": "P005", "company": "BioHealth", "abstract": "Precision medicine pipeline using machine learning for rare genetic disorders."},
    {"id": "P006", "company": "BioHealth", "abstract": "Deep learning-based anomaly detection in medical genomics data."},

    # EcoEnergy patents
    {"id": "P007", "company": "EcoEnergy", "abstract": "Smart grid load balancing with AI-based energy consumption forecasting."},
    {"id": "P008", "company": "EcoEnergy", "abstract": "Battery storage optimization algorithm for renewable energy systems."},
    {"id": "P009", "company": "EcoEnergy", "abstract": "Real-time solar farm energy output prediction using neural networks."},

    # FinSecure patents
    {"id": "P010", "company": "FinSecure", "abstract": "AI fraud detection system using transactional anomaly pattern recognition."},
    {"id": "P011", "company": "FinSecure", "abstract": "Blockchain-integrated secure payment authorization method with AI risk scoring."},
    {"id": "P012", "company": "FinSecure", "abstract": "Machine learning models for predictive financial risk analytics."},

    # AgriSmart patents
    {"id": "P013", "company": "AgriSmart", "abstract": "Drone-based AI crop monitoring system with multispectral imaging."},
    {"id": "P014", "company": "AgriSmart", "abstract": "Automated irrigation control using reinforcement learning from weather and soil data."},
    {"id": "P015", "company": "AgriSmart", "abstract": "Yield prediction model combining satellite imagery with machine learning."},

    # MedVision patents
    {"id": "P016", "company": "MedVision", "abstract": "AI-assisted MRI image segmentation for tumor detection."},
    {"id": "P017", "company": "MedVision", "abstract": "Real-time AI-guided surgical navigation using augmented reality."},
    {"id": "P018", "company": "MedVision", "abstract": "Remote health diagnostics system with AI-enabled wearable sensors."},

    # EduNext patents
    {"id": "P019", "company": "EduNext", "abstract": "Adaptive learning platform with real-time student performance feedback."},
    {"id": "P020", "company": "EduNext", "abstract": "AI curriculum generator for personalized educational pathways."},
    {"id": "P021", "company": "EduNext", "abstract": "Predictive analytics for early detection of student learning difficulties."},

    # AutoDrive patents
    {"id": "P022", "company": "AutoDrive", "abstract": "Autonomous vehicle lane detection using deep convolutional networks."},
    {"id": "P023", "company": "AutoDrive", "abstract": "AI-based route planning considering real-time traffic and weather data."},
    {"id": "P024", "company": "AutoDrive", "abstract": "Safety-critical anomaly detection in autonomous driving systems."},

    # CyberShield patents
    {"id": "P025", "company": "CyberShield", "abstract": "Real-time threat detection using AI-driven network behavior analysis."},
    {"id": "P026", "company": "CyberShield", "abstract": "Malware classification system with adversarial machine learning."},
    {"id": "P027", "company": "CyberShield", "abstract": "Automated cybersecurity incident response engine with reinforcement learning."},

    # RetailX patents
    {"id": "P028", "company": "RetailX", "abstract": "AI-based customer behavior prediction for retail optimization."},
    {"id": "P029", "company": "RetailX", "abstract": "Dynamic pricing engine using real-time sales and demand data."},
    {"id": "P030", "company": "RetailX", "abstract": "Supply chain optimization through machine learning and IoT sensors."}
]

# Exact lookup dictionaries
company_exact_info = {c['name']: c for c in company_data}
patent_exact_info = {p['id']: p for p in patent_data}

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
logging.info("Initialized RecursiveCharacterTextSplitter")

# Create Documents with splitting
company_docs = []
for c in company_data:
    chunks = splitter.split_text(c['summary'])
    for idx, chunk in enumerate(chunks):
        company_docs.append(Document(page_content=chunk, metadata={"name": c['name'], "chunk_id": idx}))
    logging.info(f"Company {c['name']} split into {len(chunks)} chunks")

patent_docs = []
for p in patent_data:
    chunks = splitter.split_text(p['abstract'])
    for idx, chunk in enumerate(chunks):
        patent_docs.append(Document(page_content=chunk, metadata={"id": p['id'], "company": p['company'], "chunk_id": idx}))
    logging.info(f"Patent {p['id']} split into {len(chunks)} chunks")

# Save to Chroma vectorstores
company_db = Chroma.from_documents(company_docs, embeddings, persist_directory="chroma_company")
logging.info("Saved company data to Chroma vectorstore")

patent_db = Chroma.from_documents(patent_docs, embeddings, persist_directory="chroma_patent")
logging.info("Saved patent data to Chroma vectorstore")

# ====== Tools =======
from langchain.tools import Tool

# RAG Retrieval Tools
def company_rag_tool(query: str):
    retriever = Chroma(persist_directory="chroma_company", embedding_function=embeddings).as_retriever()
    return retriever.invoke(query)

def patent_rag_tool(query: str):
    retriever = Chroma(persist_directory="chroma_patent", embedding_function=embeddings).as_retriever()
    return retriever.invoke(query)

# Exact Lookup Tools
def company_exact_lookup(name: str):
    return company_exact_info.get(name, f"No exact info found for {name}")

def patent_exact_lookup(pid: str):
    return patent_exact_info.get(pid, f"No exact info found for {pid}")

# Wrap tools into LangChain Tool objects
company_rag = Tool.from_function(
    func=company_rag_tool,
    name="CompanyRAGTool",
    description="Use this to retrieve company-related information from the RAG system."
)

patent_rag = Tool.from_function(
    func=patent_rag_tool,
    name="PatentRAGTool",
    description="Use this to retrieve patent-related information from the RAG system."
)

company_exact = Tool.from_function(
    func=company_exact_lookup,
    name="CompanyExactInfoTool",
    description="Use this to retrieve exact company information by name."
)

patent_exact = Tool.from_function(
    func=patent_exact_lookup,
    name="PatentExactInfoTool",
    description="Use this to retrieve exact patent information by patent ID."
)

# ====== Agent Setup =======
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Load model (can be GPT-4o-mini or gpt-3.5-turbo)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# All 4 tools
tools = [company_rag, patent_rag, company_exact, patent_exact]

# System prompt to guide multi-role behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a multi-role assistant with 4 tools: CompanyRAGTool, PatentRAGTool, CompanyExactInfoTool, and PatentExactInfoTool.\n"
     "For any user question about companies or patents, you must first use a relevant tool before responding directly.\n"
     "If a question is about a company's business focus, use CompanyRAGTool.\n"
     "If a question is about patent information, use PatentRAGTool or PatentExactInfoTool.\n"
     "If the tools return no result, then you can reply directly.")
    ,
    MessagesPlaceholder("messages"),
    MessagesPlaceholder("agent_scratchpad")
])

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt)

# With memory (optional)
memory = ConversationBufferMemory(
    memory_key="messages",
    input_key="input",   # <-- Tell memory what the input is
    output_key="output",   # <-- Fix the warning here
    return_messages=True
)

# Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

logging.info("Agent executor initialized with 4 tools")

# Example 1: RAG Retrieval about TechNova
agent_executor.invoke({"input": "Tell me about TechNova's business focus."})

# Example 2: Exact info of patent
agent_executor.invoke({"input": "Give me the exact information of patent P003."})

# Example 3: Patent RAG retrieval
agent_executor.invoke({"input": "What are the AI-related patents available?"})

# Example 4: Company Exact info
agent_executor.invoke({"input": "Show me the full record of BioHealth."})

# ====== CLI Chat =======
# print("=== Welcome to the Company & Patent Assistant ===")
# print("Type 'exit' to quit.")
# print("-----------------------------------------------")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         logging.info("User exited the session.")
#         break

#     for chunk in agent_executor.stream({"input": user_input}):
#         if "output" in chunk:
#             print(chunk["output"], end="", flush=True)
#     print("\n")  # Spacing after each response

# Category	Example Queries
# Company RAG Retrieval	
# - "What does AgriSmart do in the agriculture sector?"
# - "Tell me about CyberShield's main business area."
# - "What is RetailX focusing on?"
# Patent RAG Retrieval	
# - "Are there any patents related to smart grids or energy management?"
# - "Show me the AI applications in medical imaging patents."
# - "List AutoDrive's work on autonomous vehicle safety."
# Company Exact Info	
# - "Give me the full profile of FinSecure."
# - "What is the detailed record for EcoEnergy?"
# Patent Exact Info	
# - "Show me the exact information of patent P017."
# - "I need the details of patent P026."
# - "What is patent P014 about?"
# Mixed / Open Query	
# - "Who is working on adaptive learning platforms?"
# - "Which company is developing AI for crop monitoring?"
# - "Tell me about AI in cybersecurity threats."




