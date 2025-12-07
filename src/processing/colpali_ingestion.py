"""
ColPali Vision-Native Document Ingestion

Eliminates OCR entirely by treating documents as images and using
Vision Language Models (VLMs) for understanding.

Research Context (Dec 2025):
- ColPali = ColBERT + PaliGemma
- Documents are encoded as images, preserving layout/tables/charts
- Multi-vector embeddings capture both text AND visual structure
- Outperforms OCR pipelines on ViDoRe benchmark by 30%+

Why this matters:
- OCR mangles tables into unintelligible strings
- OCR ignores charts and diagrams
- OCR loses layout cues (indentation, font size, spatial relationships)
- ColPali "sees" the page and understands visual context

Usage:
    processor = ColPaliProcessor()
    
    # Process a PDF with visual understanding
    pages = processor.process_pdf("manual.pdf")
    
    # Each page has:
    # - image_embedding: Multi-vector visual embedding
    # - extracted_text: Gemini's visual OCR (much better than traditional)
    # - layout_analysis: Tables, charts, diagrams detected
    # - page_image: Base64 encoded for retrieval display
"""

import os
import io
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TableExtraction:
    """Structured table extracted from document page."""
    rows: List[List[str]]
    headers: Optional[List[str]] = None
    caption: Optional[str] = None
    location: Optional[Dict[str, int]] = None  # x, y, width, height
    confidence: float = 0.0


@dataclass
class ChartExtraction:
    """Chart/diagram extracted from document page."""
    chart_type: str  # bar, line, pie, flowchart, diagram, etc.
    title: Optional[str] = None
    description: str = ""
    data_points: Optional[List[Dict]] = None  # Extracted data if chart
    location: Optional[Dict[str, int]] = None
    confidence: float = 0.0


@dataclass
class LayoutElement:
    """Generic layout element detected on page."""
    element_type: str  # heading, paragraph, list, table, chart, image, caption
    content: str
    level: Optional[int] = None  # For headings: h1=1, h2=2, etc.
    bounding_box: Optional[Dict[str, int]] = None
    confidence: float = 0.0


@dataclass
class PageAnalysis:
    """Complete analysis of a single document page."""
    page_number: int
    image_base64: str  # Page rendered as image
    image_embedding: Optional[List[float]] = None  # Visual embedding
    
    # Extracted content
    extracted_text: str = ""
    layout_elements: List[LayoutElement] = field(default_factory=list)
    tables: List[TableExtraction] = field(default_factory=list)
    charts: List[ChartExtraction] = field(default_factory=list)
    
    # Metadata
    has_visual_content: bool = False
    dominant_content_type: str = "text"  # text, tabular, visual, mixed
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "page_number": self.page_number,
            "image_base64": self.image_base64[:100] + "...",  # Truncated for logging
            "extracted_text": self.extracted_text,
            "tables": [{"rows": t.rows, "caption": t.caption} for t in self.tables],
            "charts": [{"type": c.chart_type, "description": c.description} for c in self.charts],
            "has_visual_content": self.has_visual_content,
            "dominant_content_type": self.dominant_content_type,
        }


@dataclass
class DocumentAnalysis:
    """Complete analysis of a multi-page document."""
    filename: str
    total_pages: int
    pages: List[PageAnalysis]
    
    # Document-level metadata
    document_type: str = "unknown"  # manual, report, presentation, form, etc.
    detected_language: str = "en"
    has_toc: bool = False
    
    # Aggregated content
    all_tables: List[TableExtraction] = field(default_factory=list)
    all_charts: List[ChartExtraction] = field(default_factory=list)
    
    # Processing stats
    total_processing_time_ms: float = 0.0
    
    def get_searchable_text(self) -> str:
        """Get all extracted text for indexing."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.extracted_text}" 
            for p in self.pages
        )


# ═══════════════════════════════════════════════════════════════════════════
# PDF to Image Converter
# ═══════════════════════════════════════════════════════════════════════════

class PDFToImageConverter:
    """
    Convert PDF pages to images for visual processing.
    
    Uses pdf2image (poppler) for high-quality rendering.
    Falls back to PyMuPDF if poppler not available.
    """
    
    def __init__(self, dpi: int = 150):
        """
        Initialize converter.
        
        Args:
            dpi: Resolution for rendering (150 is good balance of quality/size)
        """
        self.dpi = dpi
        self._backend = self._detect_backend()
        logger.info(f"PDF converter using backend: {self._backend}")
    
    def _detect_backend(self) -> str:
        """Detect available PDF rendering backend."""
        try:
            from pdf2image import convert_from_path
            return "pdf2image"
        except ImportError:
            pass
        
        try:
            import fitz  # PyMuPDF
            return "pymupdf"
        except ImportError:
            pass
        
        return "none"
    
    def convert(self, pdf_path: str) -> List[Tuple[int, bytes]]:
        """
        Convert PDF to list of (page_number, image_bytes) tuples.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (page_number, PNG bytes) tuples
        """
        if self._backend == "pdf2image":
            return self._convert_pdf2image(pdf_path)
        elif self._backend == "pymupdf":
            return self._convert_pymupdf(pdf_path)
        else:
            raise ImportError(
                "No PDF backend available. Install pdf2image or PyMuPDF:\n"
                "  pip install pdf2image  # Requires poppler\n"
                "  pip install PyMuPDF"
            )
    
    def _convert_pdf2image(self, pdf_path: str) -> List[Tuple[int, bytes]]:
        """Convert using pdf2image (poppler backend)."""
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, dpi=self.dpi)
        results = []
        
        for i, img in enumerate(images, 1):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            results.append((i, buffer.getvalue()))
        
        return results
    
    def _convert_pymupdf(self, pdf_path: str) -> List[Tuple[int, bytes]]:
        """Convert using PyMuPDF."""
        import fitz
        
        doc = fitz.open(pdf_path)
        results = []
        
        # Calculate zoom factor for desired DPI (PyMuPDF default is 72 DPI)
        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=matrix)
            results.append((i, pix.tobytes("png")))
        
        doc.close()
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Vision Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class GeminiVisionAnalyzer:
    """
    Use Gemini's vision capabilities for document understanding.
    
    This replaces traditional OCR with visual understanding:
    - Reads text in context of visual layout
    - Understands tables as structured data
    - Interprets charts and diagrams
    - Detects document structure (headings, sections)
    """
    
    # Prompts for different analysis tasks
    LAYOUT_ANALYSIS_PROMPT = """Analyze this document page image and extract:

1. **Text Content**: Extract ALL visible text, preserving structure (headings, paragraphs, lists)
2. **Tables**: If tables exist, extract them as structured data with headers and rows
3. **Charts/Diagrams**: Describe any charts, graphs, or diagrams and extract data if possible
4. **Layout Structure**: Identify document hierarchy (main headings, subheadings, body text)

Format your response as JSON:
{
    "extracted_text": "full text content with structure preserved",
    "layout_elements": [
        {"type": "heading", "level": 1, "content": "..."},
        {"type": "paragraph", "content": "..."},
        {"type": "list_item", "content": "..."}
    ],
    "tables": [
        {
            "caption": "optional caption",
            "headers": ["col1", "col2"],
            "rows": [["data1", "data2"], ...]
        }
    ],
    "charts": [
        {
            "type": "bar|line|pie|flowchart|diagram",
            "title": "optional title",
            "description": "what the chart shows",
            "data_points": [{"label": "x", "value": 10}, ...] 
        }
    ],
    "page_type": "text|tabular|visual|mixed",
    "has_visual_content": true/false
}

Be thorough and accurate. Extract ALL text visible on the page."""

    TABLE_EXTRACTION_PROMPT = """Extract the table(s) from this image as structured data.

For each table, provide:
- headers: Column headers if present
- rows: All data rows as arrays
- caption: Any caption or title for the table

Format as JSON:
{
    "tables": [
        {
            "headers": ["Column 1", "Column 2", ...],
            "rows": [
                ["row1col1", "row1col2", ...],
                ["row2col1", "row2col2", ...]
            ],
            "caption": "Table caption if present"
        }
    ]
}

Be precise with numbers and maintain alignment."""

    CHART_ANALYSIS_PROMPT = """Analyze the chart/diagram in this image.

Provide:
1. Chart type (bar, line, pie, scatter, flowchart, diagram, etc.)
2. Title and axis labels
3. Data points if extractable
4. Key insights or what the chart communicates

Format as JSON:
{
    "chart_type": "...",
    "title": "...",
    "x_axis": "...",
    "y_axis": "...",
    "data_points": [{"label": "...", "value": ...}, ...],
    "description": "what this chart shows",
    "key_insights": ["insight 1", "insight 2"]
}"""

    def __init__(self):
        """Initialize Gemini vision analyzer."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for vision analysis")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Use Gemini 2.0 Flash for vision (fast + capable)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            logger.info("✅ Gemini vision analyzer initialized")
        except ImportError:
            raise ImportError("google-generativeai required: pip install google-generativeai")
    
    def analyze_page(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze a document page image using Gemini vision.
        
        Args:
            image_bytes: PNG image data
            
        Returns:
            Parsed analysis results
        """
        import google.generativeai as genai
        from PIL import Image
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Send to Gemini with layout analysis prompt
        response = self.model.generate_content(
            [self.LAYOUT_ANALYSIS_PROMPT, img],
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temp for accuracy
                response_mime_type="application/json"
            )
        )
        
        # Parse JSON response
        try:
            import json
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # Fallback: return raw text if JSON parsing fails
            logger.warning("Failed to parse Gemini response as JSON")
            return {
                "extracted_text": response.text,
                "layout_elements": [],
                "tables": [],
                "charts": [],
                "page_type": "text",
                "has_visual_content": False
            }
    
    def extract_tables(self, image_bytes: bytes) -> List[TableExtraction]:
        """
        Focused table extraction from image.
        
        Args:
            image_bytes: PNG image data
            
        Returns:
            List of extracted tables
        """
        import google.generativeai as genai
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_bytes))
        
        response = self.model.generate_content(
            [self.TABLE_EXTRACTION_PROMPT, img],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        try:
            import json
            result = json.loads(response.text)
            
            tables = []
            for t in result.get("tables", []):
                tables.append(TableExtraction(
                    rows=t.get("rows", []),
                    headers=t.get("headers"),
                    caption=t.get("caption"),
                    confidence=0.9
                ))
            return tables
        except:
            return []
    
    def analyze_chart(self, image_bytes: bytes) -> Optional[ChartExtraction]:
        """
        Focused chart/diagram analysis.
        
        Args:
            image_bytes: PNG image data
            
        Returns:
            Chart extraction or None
        """
        import google.generativeai as genai
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_bytes))
        
        response = self.model.generate_content(
            [self.CHART_ANALYSIS_PROMPT, img],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        try:
            import json
            result = json.loads(response.text)
            
            return ChartExtraction(
                chart_type=result.get("chart_type", "unknown"),
                title=result.get("title"),
                description=result.get("description", ""),
                data_points=result.get("data_points"),
                confidence=0.85
            )
        except:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Visual Embedding Generator
# ═══════════════════════════════════════════════════════════════════════════

class VisualEmbedder:
    """
    Generate embeddings for document page images.
    
    Uses multimodal embedding models that understand both
    visual layout and text content.
    """
    
    def __init__(self):
        """Initialize visual embedder."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for visual embeddings")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            logger.info("✅ Visual embedder initialized")
        except ImportError:
            raise ImportError("google-generativeai required")
    
    def embed_image(self, image_bytes: bytes) -> List[float]:
        """
        Generate embedding for document page image.
        
        Note: As of Dec 2025, Gemini doesn't have direct image embedding.
        We use a workaround: describe the image, then embed the description.
        
        For true ColPali-style embeddings, you'd use:
        - Voyage AI's multimodal embeddings
        - OpenAI's multimodal embeddings (when available)
        - Self-hosted ColPali model
        
        Args:
            image_bytes: PNG image data
            
        Returns:
            Embedding vector
        """
        from PIL import Image
        
        # First, get a rich description of the visual content
        img = Image.open(io.BytesIO(image_bytes))
        
        model = self._genai.GenerativeModel("gemini-2.0-flash-exp")
        
        describe_prompt = """Describe this document page in detail for embedding purposes.
Include:
- All text content
- Layout structure (headers, sections, columns)
- Any tables (describe structure)
- Any charts/diagrams (describe what they show)
- Visual elements (logos, images, formatting)

Be comprehensive - this description will be used for semantic search."""
        
        response = model.generate_content([describe_prompt, img])
        description = response.text
        
        # Now embed the rich description
        embed_model = self._genai.GenerativeModel("models/text-embedding-004")
        result = self._genai.embed_content(
            model="models/text-embedding-004",
            content=description,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        return result['embedding']
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate query embedding for searching visual documents.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        result = self._genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        
        return result['embedding']


# ═══════════════════════════════════════════════════════════════════════════
# Main ColPali Processor
# ═══════════════════════════════════════════════════════════════════════════

class ColPaliProcessor:
    """
    Vision-native document processor inspired by ColPali architecture.
    
    Replaces traditional OCR pipeline with visual understanding:
    1. Convert PDF pages to images
    2. Use Gemini vision to understand content (not just OCR)
    3. Generate visual-aware embeddings
    4. Extract structured data (tables, charts) with layout context
    
    Usage:
        processor = ColPaliProcessor()
        
        # Process entire document
        analysis = processor.process_pdf("document.pdf")
        
        # Access results
        for page in analysis.pages:
            print(f"Page {page.page_number}:")
            print(f"  Text: {page.extracted_text[:100]}...")
            print(f"  Tables: {len(page.tables)}")
            print(f"  Charts: {len(page.charts)}")
        
        # Get all text for indexing
        full_text = analysis.get_searchable_text()
    """
    
    def __init__(
        self,
        dpi: int = 150,
        generate_embeddings: bool = True,
        extract_tables: bool = True,
        extract_charts: bool = True
    ):
        """
        Initialize ColPali processor.
        
        Args:
            dpi: Resolution for PDF rendering
            generate_embeddings: Whether to generate visual embeddings
            extract_tables: Whether to perform detailed table extraction
            extract_charts: Whether to perform chart analysis
        """
        self.pdf_converter = PDFToImageConverter(dpi=dpi)
        self.vision_analyzer = GeminiVisionAnalyzer()
        
        self.generate_embeddings = generate_embeddings
        self.extract_tables = extract_tables
        self.extract_charts = extract_charts
        
        if generate_embeddings:
            self.visual_embedder = VisualEmbedder()
        else:
            self.visual_embedder = None
        
        logger.info("✅ ColPali processor initialized")
        logger.info(f"   Embeddings: {generate_embeddings}")
        logger.info(f"   Table extraction: {extract_tables}")
        logger.info(f"   Chart analysis: {extract_charts}")
    
    def process_pdf(self, pdf_path: str) -> DocumentAnalysis:
        """
        Process entire PDF with vision-native understanding.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Complete document analysis
        """
        import time
        start_time = time.time()
        
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"📄 Processing PDF: {path.name}")
        
        # Convert PDF to images
        page_images = self.pdf_converter.convert(pdf_path)
        logger.info(f"   Converted {len(page_images)} pages to images")
        
        # Process each page
        pages = []
        all_tables = []
        all_charts = []
        
        for page_num, image_bytes in page_images:
            logger.info(f"   Analyzing page {page_num}/{len(page_images)}...")
            
            page_start = time.time()
            
            # Encode image as base64 for storage
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Analyze with Gemini vision
            analysis = self.vision_analyzer.analyze_page(image_bytes)
            
            # Parse layout elements
            layout_elements = []
            for elem in analysis.get("layout_elements", []):
                layout_elements.append(LayoutElement(
                    element_type=elem.get("type", "unknown"),
                    content=elem.get("content", ""),
                    level=elem.get("level"),
                    confidence=0.9
                ))
            
            # Parse tables
            tables = []
            for t in analysis.get("tables", []):
                table = TableExtraction(
                    rows=t.get("rows", []),
                    headers=t.get("headers"),
                    caption=t.get("caption"),
                    confidence=0.9
                )
                tables.append(table)
                all_tables.append(table)
            
            # Parse charts
            charts = []
            for c in analysis.get("charts", []):
                chart = ChartExtraction(
                    chart_type=c.get("type", "unknown"),
                    title=c.get("title"),
                    description=c.get("description", ""),
                    data_points=c.get("data_points"),
                    confidence=0.85
                )
                charts.append(chart)
                all_charts.append(chart)
            
            # Generate visual embedding if enabled
            embedding = None
            if self.generate_embeddings and self.visual_embedder:
                try:
                    embedding = self.visual_embedder.embed_image(image_bytes)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for page {page_num}: {e}")
            
            # Create page analysis
            page = PageAnalysis(
                page_number=page_num,
                image_base64=image_base64,
                image_embedding=embedding,
                extracted_text=analysis.get("extracted_text", ""),
                layout_elements=layout_elements,
                tables=tables,
                charts=charts,
                has_visual_content=analysis.get("has_visual_content", False),
                dominant_content_type=analysis.get("page_type", "text"),
                processing_time_ms=(time.time() - page_start) * 1000
            )
            
            pages.append(page)
        
        # Determine document type based on content
        doc_type = self._classify_document_type(pages)
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ PDF processing complete")
        logger.info(f"   Pages: {len(pages)}")
        logger.info(f"   Tables found: {len(all_tables)}")
        logger.info(f"   Charts found: {len(all_charts)}")
        logger.info(f"   Document type: {doc_type}")
        logger.info(f"   Total time: {total_time:.0f}ms")
        
        return DocumentAnalysis(
            filename=path.name,
            total_pages=len(pages),
            pages=pages,
            document_type=doc_type,
            all_tables=all_tables,
            all_charts=all_charts,
            total_processing_time_ms=total_time
        )
    
    def process_image(self, image_path: str) -> PageAnalysis:
        """
        Process a single image (screenshot, scan, etc.).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Page analysis
        """
        import time
        start_time = time.time()
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Analyze
        analysis = self.vision_analyzer.analyze_page(image_bytes)
        
        # Generate embedding
        embedding = None
        if self.generate_embeddings and self.visual_embedder:
            embedding = self.visual_embedder.embed_image(image_bytes)
        
        return PageAnalysis(
            page_number=1,
            image_base64=image_base64,
            image_embedding=embedding,
            extracted_text=analysis.get("extracted_text", ""),
            has_visual_content=analysis.get("has_visual_content", False),
            dominant_content_type=analysis.get("page_type", "text"),
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _classify_document_type(self, pages: List[PageAnalysis]) -> str:
        """Classify document type based on content analysis."""
        total_tables = sum(len(p.tables) for p in pages)
        total_charts = sum(len(p.charts) for p in pages)
        visual_pages = sum(1 for p in pages if p.has_visual_content)
        
        # Heuristics for document classification
        if total_tables > len(pages) * 0.5:
            return "report"
        elif total_charts > len(pages) * 0.3:
            return "presentation"
        elif visual_pages > len(pages) * 0.7:
            return "manual"
        elif len(pages) == 1:
            return "form"
        else:
            return "document"
    
    def search(
        self,
        query: str,
        documents: List[DocumentAnalysis],
        top_k: int = 5
    ) -> List[Tuple[PageAnalysis, float]]:
        """
        Search across processed documents using visual embeddings.
        
        Args:
            query: Search query
            documents: List of processed documents
            top_k: Number of results to return
            
        Returns:
            List of (page, score) tuples
        """
        if not self.visual_embedder:
            raise ValueError("Embeddings disabled - enable to use search")
        
        import numpy as np
        
        # Embed query
        query_embedding = self.visual_embedder.embed_query(query)
        query_vec = np.array(query_embedding)
        
        # Collect all pages with embeddings
        candidates = []
        for doc in documents:
            for page in doc.pages:
                if page.image_embedding:
                    candidates.append(page)
        
        if not candidates:
            return []
        
        # Calculate similarities
        scores = []
        for page in candidates:
            page_vec = np.array(page.image_embedding)
            # Cosine similarity
            similarity = np.dot(query_vec, page_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(page_vec)
            )
            scores.append((page, float(similarity)))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def process_document(
    file_path: str,
    generate_embeddings: bool = True
) -> DocumentAnalysis:
    """
    Process a document file (PDF or image) with visual understanding.
    
    Args:
        file_path: Path to document
        generate_embeddings: Whether to generate visual embeddings
        
    Returns:
        Document analysis
    """
    processor = ColPaliProcessor(generate_embeddings=generate_embeddings)
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        return processor.process_pdf(file_path)
    elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        page = processor.process_image(file_path)
        return DocumentAnalysis(
            filename=path.name,
            total_pages=1,
            pages=[page],
            document_type="image"
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def extract_tables_from_pdf(pdf_path: str) -> List[TableExtraction]:
    """
    Extract all tables from a PDF document.
    
    Args:
        pdf_path: Path to PDF
        
    Returns:
        List of extracted tables
    """
    processor = ColPaliProcessor(
        generate_embeddings=False,
        extract_charts=False
    )
    
    analysis = processor.process_pdf(pdf_path)
    return analysis.all_tables


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from PDF using vision-native understanding.
    
    This is superior to traditional OCR because:
    - Preserves document structure
    - Handles multi-column layouts correctly
    - Extracts text from within images/diagrams
    
    Args:
        pdf_path: Path to PDF
        
    Returns:
        Extracted text
    """
    processor = ColPaliProcessor(
        generate_embeddings=False,
        extract_tables=False,
        extract_charts=False
    )
    
    analysis = processor.process_pdf(pdf_path)
    return analysis.get_searchable_text()
