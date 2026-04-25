from sqlalchemy.orm import Session
from sqlalchemy import select
from models.document import Document, DocumentChunk

class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_document_with_chunks(self, user_id: str, pdf_source: str, chunks_data: list[dict]) -> Document:
        """Saves metadata and embeddings in a single transaction."""
        
        # 1. Create the parent metadata record
        new_doc = Document(user_id=user_id, pdf_source=pdf_source)
        
        # 2. Attach the vector chunks
        for chunk in chunks_data:
            new_chunk = DocumentChunk(
                text_content=chunk["text"],
                embedding=chunk["embedding"]
            )
            new_doc.chunks.append(new_chunk)
            
        self.session.add(new_doc)
        self.session.commit()
        self.session.refresh(new_doc)
        
        return new_doc

    def search_similar_chunks(self, query_embedding: list[float], limit: int = 5):
        """Performs a vector similarity search (Cosine Distance) and fetches metadata."""
        
        stmt = (
            select(DocumentChunk, Document)
            .join(DocumentChunk.document)
            # .cosine_distance is provided by pgvector
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        
        results = self.session.execute(stmt).all()
        return results