import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

_log = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self, vision_model: Any):
        self.vision_model = vision_model

        # Initialize Docling converter
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseDocumentBackend,
                ),
            }
        )

    def _get_page_numbers(self, element: Any) -> List[int]:
        page_numbers: List[int] = []
        if not hasattr(element, "prov") or not element.prov:
            return page_numbers

        for provenance in element.prov:
            page_no = getattr(provenance, "page_no", None)
            if isinstance(page_no, int) and page_no not in page_numbers:
                page_numbers.append(page_no)

        return page_numbers

    def _get_primary_page_number(self, element: Any) -> Optional[int]:
        page_numbers = self._get_page_numbers(element)
        return min(page_numbers) if page_numbers else None

    def _build_visual_basename(
        self, prefix: str, page_number: Optional[int], index: int
    ) -> str:
        page_token = f"p{page_number}" if page_number is not None else "punknown"
        return f"{prefix}_{page_token}_{index:03d}"

    def _get_provenance_metadata(self, element: Any) -> List[Dict[str, Any]]:
        if not hasattr(element, "prov") or not element.prov:
            return []
        return [provenance.to_dict() for provenance in element.prov]

    def _table_to_text(self, table_df):
        if table_df is None or table_df.empty:
            return "Empty table"

        columns = ", ".join(str(col) for col in table_df.columns)

        rows = []
        for _, row in table_df.iterrows():
            row_text = ", ".join(f"{col} = {row[col]}" for col in table_df.columns)
            rows.append(row_text)

        table_text = f"Table with columns: {columns}.\nRows:\n" + "\n".join(rows[:20])
        return table_text

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _caption_image(self, image_path):
        try:
            base64_image = self._encode_image(image_path)
            response = self.vision_model.invoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, charts, or visual elements.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    )
                ]
            )
            return response.content
        except Exception as e:
            _log.error(f"Failed to caption image {image_path}: {e}")
            return "Image captioning failed"

    def extract_visual_elements(
        self, document, doc_name: str
    ) -> Dict[str, List[Document]]:
        output_dir = Path("./extracted_elements") / doc_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_data = []
        table_documents = []
        image_documents = []
        table_index = 0
        image_index = 0

        for element, _level in document.iterate_items():
            page_numbers = self._get_page_numbers(element)
            page_number = self._get_primary_page_number(element)

            if isinstance(element, TableItem):
                table_index += 1
                table_basename = self._build_visual_basename(
                    "table", page_number, table_index
                )
                try:
                    table_img = element.get_image(document)
                    if table_img:
                        img_path = output_dir / f"{table_basename}.png"
                        table_img.save(img_path)
                except Exception as e:
                    _log.debug(f"Could not save table image: {e}")

                try:
                    table_df = element.export_to_dataframe(doc=document)
                    if table_df is not None and not table_df.empty:
                        csv_filename = f"{table_basename}.csv"
                        table_df.to_csv(output_dir / csv_filename)
                        table_text = self._table_to_text(table_df)

                        table_doc = Document(
                            page_content=table_text,
                            metadata={
                                "document": doc_name,
                                "type": "table",
                                "table_label": str(element.label),
                                "table_id": table_basename,
                                "content_type": "table",
                                "source_file": csv_filename,
                                "page_number": page_number,
                                "page_numbers": json.dumps(page_numbers)
                                if page_numbers
                                else "[]",
                            },
                        )
                        table_documents.append(table_doc)

                        tables_data.append(
                            {
                                "label": str(element.label),
                                "table_id": table_basename,
                                "page_number": page_number,
                                "page_numbers": page_numbers,
                                "prov": self._get_provenance_metadata(element),
                                "preview": table_df.head(5).to_dict(orient="records"),
                            }
                        )
                except Exception as e:
                    _log.debug(f"Could not export table data: {e}")

            elif isinstance(element, PictureItem):
                image_index += 1
                image_basename = self._build_visual_basename(
                    "image", page_number, image_index
                )
                try:
                    img = element.get_image(document)
                    if img:
                        img_path = output_dir / f"{image_basename}.png"
                        img.save(img_path)

                        caption = self._caption_image(img_path)

                        image_doc = Document(
                            page_content=f"Image description: {caption}",
                            metadata={
                                "document": doc_name,
                                "type": "image",
                                "image_label": str(element.label),
                                "image_id": image_basename,
                                "content_type": "image",
                                "image_path": str(img_path),
                                "page_number": page_number,
                                "page_numbers": json.dumps(page_numbers)
                                if page_numbers
                                else "[]",
                            },
                        )

                        image_documents.append(image_doc)
                except Exception as e:
                    _log.debug(f"Could not process image: {e}")

        if tables_data:
            with open(output_dir / "tables_metadata.json", "w") as f:
                json.dump(tables_data, f, indent=2)
            _log.info(f"Extracted {len(tables_data)} tables to {output_dir}")

        return {"tables": table_documents, "images": image_documents}

    def convert_all(self, doc_paths: List[Path]):
        return self.doc_converter.convert_all(doc_paths, raises_on_error=False)

    async def _caption_image_async(self, image_path) -> str:
        try:
            base64_image = self._encode_image(image_path)
            response = await self.vision_model.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, charts, or visual elements.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    )
                ]
            )
            return response.content
        except Exception as e:
            _log.error(f"Failed to caption image {image_path}: {e}")
            return "Image captioning failed"

    async def extract_visual_elements_async(
        self, document, doc_name: str
    ) -> Dict[str, List[Document]]:
        """Async version of extract_visual_elements: all images are captioned concurrently."""
        output_dir = Path("./extracted_elements") / doc_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_data = []
        table_documents = []
        pending_images: List[tuple] = []
        table_index = 0
        image_index = 0

        for element, _level in document.iterate_items():
            page_numbers = self._get_page_numbers(element)
            page_number = self._get_primary_page_number(element)

            if isinstance(element, TableItem):
                table_index += 1
                table_basename = self._build_visual_basename(
                    "table", page_number, table_index
                )
                try:
                    table_img = element.get_image(document)
                    if table_img:
                        img_path = output_dir / f"{table_basename}.png"
                        table_img.save(img_path)
                except Exception as e:
                    _log.debug(f"Could not save table image: {e}")

                try:
                    table_df = element.export_to_dataframe(doc=document)
                    if table_df is not None and not table_df.empty:
                        csv_filename = f"{table_basename}.csv"
                        table_df.to_csv(output_dir / csv_filename)
                        table_text = self._table_to_text(table_df)

                        table_doc = Document(
                            page_content=table_text,
                            metadata={
                                "document": doc_name,
                                "type": "table",
                                "table_label": str(element.label),
                                "table_id": table_basename,
                                "content_type": "table",
                                "source_file": csv_filename,
                                "page_number": page_number,
                                "page_numbers": json.dumps(page_numbers)
                                if page_numbers
                                else "[]",
                            },
                        )
                        table_documents.append(table_doc)

                        tables_data.append(
                            {
                                "label": str(element.label),
                                "table_id": table_basename,
                                "page_number": page_number,
                                "page_numbers": page_numbers,
                                "prov": self._get_provenance_metadata(element),
                                "preview": table_df.head(5).to_dict(orient="records"),
                            }
                        )
                except Exception as e:
                    _log.debug(f"Could not export table data: {e}")

            elif isinstance(element, PictureItem):
                image_index += 1
                image_basename = self._build_visual_basename(
                    "image", page_number, image_index
                )
                try:
                    img = element.get_image(document)
                    if img:
                        img_path = output_dir / f"{image_basename}.png"
                        img.save(img_path)
                        pending_images.append(
                            (
                                element,
                                img_path,
                                page_number,
                                page_numbers,
                                image_basename,
                            )
                        )
                except Exception as e:
                    _log.debug(f"Could not process image: {e}")

        # Caption all images in this document concurrently
        image_documents = []
        if pending_images:
            captions = await asyncio.gather(
                *[
                    self._caption_image_async(img_path)
                    for _, img_path, _, _, _ in pending_images
                ]
            )
            for (
                element,
                img_path,
                page_number,
                page_numbers,
                image_basename,
            ), caption in zip(pending_images, captions):
                image_doc = Document(
                    page_content=f"Image description: {caption}",
                    metadata={
                        "document": doc_name,
                        "type": "image",
                        "image_label": str(element.label),
                        "image_id": image_basename,
                        "content_type": "image",
                        "image_path": str(img_path),
                        "page_number": page_number,
                        "page_numbers": json.dumps(page_numbers)
                        if page_numbers
                        else "[]",
                    },
                )
                image_documents.append(image_doc)

        if tables_data:
            with open(output_dir / "tables_metadata.json", "w") as f:
                json.dump(tables_data, f, indent=2)
            _log.info(f"Extracted {len(tables_data)} tables to {output_dir}")

        return {"tables": table_documents, "images": image_documents}

    async def convert_all_async(self, doc_paths: List[Path]):
        """Run the blocking Docling conversion in a thread so the event loop stays free."""
        loop = asyncio.get_running_loop()

        def _convert():
            return list(
                self.doc_converter.convert_all(doc_paths, raises_on_error=False)
            )

        return await loop.run_in_executor(None, _convert)
