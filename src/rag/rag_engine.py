"""
Retrieval-Augmented Generation (RAG) Module
Enhances document understanding with contextual knowledge retrieval
"""
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
import chromadb
from chromadb.config import Settings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import TextLoader
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. RAG features will be limited.")


class RAGEngine:
    """Retrieval-Augmented Generation for Document Understanding"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get('chunk_size', 500)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.top_k = config.get('top_k_results', 3)
        
        # Initialize embeddings
        if LANGCHAIN_AVAILABLE:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            
            # Initialize vector store
            self.vector_store = None
            self._init_vector_store()
        else:
            self.embeddings = None
            self.text_splitter = None
            self.vector_store = None
    
    def _init_vector_store(self):
        """Initialize vector store"""
        try:
            persist_directory = "data/chroma_db"
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def add_document(self, text: str, metadata: Dict):
        """Add document to vector store"""
        if not LANGCHAIN_AVAILABLE or not self.vector_store:
            logger.warning("RAG not available")
            return
        
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Add to vector store
            metadatas = [metadata] * len(chunks)
            self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
            
            logger.info(f"Added document with {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for relevant documents"""
        if not LANGCHAIN_AVAILABLE or not self.vector_store:
            return []
        
        try:
            k = top_k or self.top_k
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(score)
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def enhance_extraction(self, document_text: str, document_type: str) -> Dict:
        """
        Enhance field extraction with retrieved context
        """
        if not LANGCHAIN_AVAILABLE:
            return {'enhanced_fields': {}, 'context': []}
        
        # Search for similar documents
        query = f"Extract fields from {document_type}: {document_text[:200]}"
        context = self.search(query)
        
        enhanced_fields = {}
        
        # Use context to improve extraction
        if context:
            logger.info(f"Found {len(context)} relevant documents for context")
            
            # Extract patterns from similar documents
            for ctx in context:
                if 'extracted_fields' in ctx.get('metadata', {}):
                    # Learn from similar documents
                    for field, value in ctx['metadata']['extracted_fields'].items():
                        if field not in enhanced_fields:
                            enhanced_fields[field] = []
                        enhanced_fields[field].append(value)
        
        return {
            'enhanced_fields': enhanced_fields,
            'context': context
        }
    
    def build_knowledge_base(self, documents: List[Dict]):
        """Build knowledge base from training documents"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Cannot build knowledge base.")
            return
        
        logger.info(f"Building knowledge base from {len(documents)} documents...")
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = {
                'document_type': doc.get('type', 'unknown'),
                'extracted_fields': doc.get('fields', {}),
                'source': doc.get('source', 'unknown')
            }
            
            self.add_document(text, metadata)
        
        logger.info("Knowledge base built successfully")


class KnowledgeGraph:
    """Simple knowledge graph for entity relationships"""
    
    def __init__(self):
        self.graph = {
            'nodes': {},  # entity_id -> entity_data
            'edges': []    # [(source_id, target_id, relationship)]
        }
    
    def add_entity(self, entity_id: str, entity_data: Dict):
        """Add entity to knowledge graph"""
        self.graph['nodes'][entity_id] = entity_data
    
    def add_relationship(self, source_id: str, target_id: str, relationship: str):
        """Add relationship between entities"""
        self.graph['edges'].append((source_id, target_id, relationship))
    
    def build_from_document(self, entities: Dict, document_type: str):
        """Build graph from extracted entities"""
        # Add document node
        doc_id = f"doc_{document_type}_{id(entities)}"
        self.add_entity(doc_id, {'type': 'document', 'doc_type': document_type})
        
        # Add entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if isinstance(entity, dict) and 'text' in entity:
                    entity_id = f"{entity_type}_{entity['text']}"
                    self.add_entity(entity_id, {
                        'type': entity_type,
                        'text': entity['text'],
                        'label': entity.get('label', entity_type)
                    })
                    
                    # Connect to document
                    self.add_relationship(doc_id, entity_id, 'contains')
    
    def get_related_entities(self, entity_id: str) -> List[str]:
        """Get entities related to given entity"""
        related = []
        
        for source, target, rel in self.graph['edges']:
            if source == entity_id:
                related.append(target)
            elif target == entity_id:
                related.append(source)
        
        return related
    
    def visualize(self, output_path: str):
        """Visualize knowledge graph"""
        try:
            from pyvis.network import Network
            
            net = Network(height="750px", width="100%", directed=True)
            
            # Add nodes
            for node_id, node_data in self.graph['nodes'].items():
                net.add_node(
                    node_id,
                    label=node_data.get('text', node_id),
                    title=str(node_data)
                )
            
            # Add edges
            for source, target, rel in self.graph['edges']:
                net.add_edge(source, target, label=rel)
            
            # Save
            net.save_graph(output_path)
            logger.info(f"Knowledge graph saved to: {output_path}")
            
        except ImportError:
            logger.warning("pyvis not available. Cannot visualize graph.")
    
    def export_json(self, output_path: str):
        """Export graph as JSON"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.graph, f, indent=2)
        
        logger.info(f"Knowledge graph exported to: {output_path}")


def create_rag_pipeline(config: Dict) -> RAGEngine:
    """Factory function to create RAG engine"""
    return RAGEngine(config)
