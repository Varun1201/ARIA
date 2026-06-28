import httpx
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


ARXIV_API = "https://export.arxiv.org/api/query"

# Specific arXiv paper IDs for foundational AI research papers
# These are verified, high-quality papers directly relevant to ARIA's domain
CURATED_PAPERS = {
    "rag": [
        "2005.11401",  # RAG original paper (Lewis et al.)
        "2004.04906",  # Dense Passage Retrieval (DPR)
        "2112.09118",  # Improving RAG with REALM
        "2302.00083",  # Benchmarking RAG
        "2312.10997",  # RAG survey
        "2401.15884",  # CRAG - Corrective RAG
        "2404.16130",  # RAG vs Fine-tuning
    ],
    "llm_foundation": [
        "1706.03762",  # Attention is All You Need
        "2005.14165",  # GPT-3
        "2302.13971",  # LLaMA
        "2307.09288",  # LLaMA 2
        "2310.06825",  # Mistral 7B
        "2203.02155",  # InstructGPT / RLHF
        "2210.11610",  # Scaling instruction finetuning
    ],
    "agents": [
        "2210.03629",  # ReAct
        "2302.04761",  # Toolformer
        "2303.17580",  # HuggingGPT
        "2308.11432",  # AgentBench
        "2309.07864",  # OpenAgents
    ],
    "multimodal": [
        "2103.00020",  # CLIP
        "2304.08485",  # LLaVA
        "2301.12597",  # BLIP-2
        "2204.14198",  # Flamingo
    ],
    "evaluation": [
        "2109.07958",  # TruthfulQA
        "2211.09110",  # HELM
        "2307.03025",  # FActScore hallucination
        "2309.01431",  # RAGAS - RAG evaluation
        "2306.05685",  # Hallucination survey
    ],
}

# Topic clusters still used for dynamic fetching
TOPIC_CLUSTERS = {
    "rag": ["retrieval augmented generation"],
    "llm_foundation": ["large language models"],
    "agents": ["LLM agents"],
    "multimodal": ["vision language models"],
    "evaluation": ["LLM evaluation hallucination"],
}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    pdf_url: str
    categories: list[str]
    topic_cluster: str
    extra_metadata: dict = field(default_factory=dict)


class ArxivFetcher:
    """
    Fetches papers from arXiv API.
    Uses curated paper IDs for reliable, relevant results.
    """

    async def fetch_by_ids(
        self,
        arxiv_ids: list[str],
        topic_cluster: str = "general",
    ) -> list[ArxivPaper]:
        """Fetch specific papers by arXiv ID — most reliable method."""
        id_list = ",".join(arxiv_ids)
        params = {
            "id_list": id_list,
            "max_results": len(arxiv_ids),
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(ARXIV_API, params=params)
            response.raise_for_status()

        papers = self._parse_atom(response.text, topic_cluster)
        logger.info(f"Fetched {len(papers)} papers by ID for cluster: {topic_cluster}")
        return papers

    async def fetch_by_query(
        self,
        query: str,
        max_results: int = 10,
        topic_cluster: str = "general",
    ) -> list[ArxivPaper]:
        """Fetch papers by search query with cs.CL/cs.AI category filter."""
        # Add category filter to get only CS/AI papers
        filtered_query = f"({query}) AND (cat:cs.CL OR cat:cs.AI OR cat:cs.IR OR cat:cs.LG)"
        params = {
            "search_query": filtered_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(ARXIV_API, params=params)
            response.raise_for_status()

        papers = self._parse_atom(response.text, topic_cluster)
        logger.info(f"Fetched {len(papers)} papers for query: '{query}'")
        return papers

    async def fetch_cluster(
        self,
        cluster_name: str,
        max_per_query: int = 5,
    ) -> list[ArxivPaper]:
        """
        Fetch papers for a topic cluster.
        Uses curated IDs first (reliable), then dynamic query as supplement.
        """
        papers = []
        seen_ids = set()

        # 1. Fetch curated papers by ID (always reliable)
        curated_ids = CURATED_PAPERS.get(cluster_name, [])
        if curated_ids:
            # Fetch in batches of 10
            for i in range(0, len(curated_ids), 10):
                batch = curated_ids[i:i+10]
                batch_papers = await self.fetch_by_ids(batch, cluster_name)
                for p in batch_papers:
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        papers.append(p)
                await asyncio.sleep(1)

        # 2. Supplement with dynamic query if needed
        if len(papers) < max_per_query * 2:
            queries = TOPIC_CLUSTERS.get(cluster_name, [])
            for query in queries[:1]:
                dynamic = await self.fetch_by_query(
                    query=query,
                    max_results=max_per_query,
                    topic_cluster=cluster_name,
                )
                for p in dynamic:
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        papers.append(p)
                await asyncio.sleep(1)

        logger.info(f"Cluster '{cluster_name}': {len(papers)} unique papers")
        return papers

    async def fetch_all_clusters(self, max_per_query: int = 3) -> list[ArxivPaper]:
        """Fetch papers across all topic clusters."""
        all_papers = []
        for cluster_name in CURATED_PAPERS:
            papers = await self.fetch_cluster(cluster_name, max_per_query)
            all_papers.extend(papers)
            await asyncio.sleep(2)
        logger.info(f"Total papers fetched: {len(all_papers)}")
        return all_papers

    async def fetch_recent(
        self,
        days: int = 7,
        max_results: int = 20,
    ) -> list[ArxivPaper]:
        """Fetch recent papers across AI/ML categories."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        all_recent = []

        for cluster_name, queries in TOPIC_CLUSTERS.items():
            for query in queries[:1]:
                papers = await self.fetch_by_query(
                    query=query,
                    max_results=max_results,
                    topic_cluster=cluster_name,
                )
                recent = [p for p in papers if p.published >= cutoff]
                all_recent.extend(recent)
                await asyncio.sleep(1)

        seen = set()
        unique = []
        for p in all_recent:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                unique.append(p)

        logger.info(f"Found {len(unique)} papers published in last {days} days")
        return unique

    def _parse_atom(self, xml_text: str, topic_cluster: str) -> list[ArxivPaper]:
        """Parse arXiv Atom XML response into ArxivPaper objects."""
        import xml.etree.ElementTree as ET

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            try:
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                ]

                published_str = entry.find("atom:published", ns).text
                updated_str = entry.find("atom:updated", ns).text
                published = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                updated = datetime.fromisoformat(
                    updated_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)

                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                for link in entry.findall("atom:link", ns):
                    if link.get("type") == "application/pdf":
                        pdf_url = link.get("href")
                        break

                categories = [
                    c.get("term")
                    for c in entry.findall("atom:category", ns)
                ]

                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors[:5],
                    abstract=abstract,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    categories=categories,
                    topic_cluster=topic_cluster,
                    extra_metadata={
                        "source": "arxiv",
                        "arxiv_id": arxiv_id,
                        "topic_cluster": topic_cluster,
                    },
                ))

            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
                continue

        return papers


# Singleton
arxiv_fetcher = ArxivFetcher()