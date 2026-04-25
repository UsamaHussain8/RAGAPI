from datetime import datetime
from sqlalchemy import ForeignKey, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    pdf_source: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship linking metadata to the vector chunks
    # cascade="all, delete-orphan" ensures chunks are deleted if the document is deleted
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Connects the embedding back to the metadata
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    text_content: Mapped[str] = mapped_column(Text)
    
    # The pgvector column. You MUST match the dimension to your AI model.
    # e.g., 1536 for OpenAI ada-002, 768 for many HuggingFace open-source models
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))

    document: Mapped["Document"] = relationship(back_populates="chunks")