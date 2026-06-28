import asyncio
import httpx
import tempfile
import os
from pathlib import Path
from loguru import logger
from arxiv.fetcher import ArxivPaper


class PDFDownloader:
    """
    Downloads arXiv PDFs to a temporary directory.
    Returns file paths ready for ingestion into ARIA.
    """

    def __init__(self, download_dir: str = None):
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="aria_arxiv_")
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"PDF download dir: {self.download_dir}")

    async def download_paper(self, paper: ArxivPaper) -> str | None:
        """
        Download a single paper PDF.
        Returns local file path, or None if download failed.
        """
        # Sanitize filename
        safe_title = "".join(
            c if c.isalnum() or c in "._- " else "_"
            for c in paper.title[:60]
        ).strip()
        filename = f"{paper.arxiv_id.replace('/', '_')}_{safe_title}.pdf"
        filepath = os.path.join(self.download_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            logger.debug(f"Already downloaded: {filename}")
            return filepath

        try:
            async with httpx.AsyncClient(
                timeout=60,
                follow_redirects=True,
                headers={"User-Agent": "ARIA-Research-System/1.0"}
            ) as client:
                response = await client.get(paper.pdf_url)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

            size_kb = os.path.getsize(filepath) / 1024
            logger.info(f"Downloaded: {filename} ({size_kb:.0f} KB)")
            return filepath

        except Exception as e:
            logger.error(f"Failed to download {paper.arxiv_id}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    async def download_papers(
        self,
        papers: list[ArxivPaper],
        max_concurrent: int = 3,
    ) -> list[tuple[ArxivPaper, str]]:
        """
        Download multiple papers with concurrency control.
        Returns list of (paper, filepath) tuples for successful downloads.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(paper: ArxivPaper):
            async with semaphore:
                filepath = await self.download_paper(paper)
                await asyncio.sleep(1)  # Rate limit
                return (paper, filepath)

        results = await asyncio.gather(
            *[download_with_limit(p) for p in papers],
            return_exceptions=True,
        )

        successful = [
            (paper, path)
            for paper, path in results
            if not isinstance((paper, path), Exception) and path is not None
        ]

        logger.info(f"Downloaded {len(successful)}/{len(papers)} papers successfully")
        return successful


# Singleton
pdf_downloader = PDFDownloader()