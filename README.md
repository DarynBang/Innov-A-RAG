## 🤖 Innov-A-RAG: Innovation Discovery Multi-Agent RAG Pipeline.

Run code to save firm summary and patents first -> Process query, either with patent or not.

### Things still need to add.consider:
- QueryNormalizingAgent/PlanningAgent: -> direct workflow
- Hybrid Retrieval (Dense + Sparse): Augment vector search with a traditional sparse index (e.g. BM25) over raw text or keywords. Blend scores to maintain recall in case embeddings miss niche terms.
- ConsistencyAgent/FactCheckingAgent: add between Opportunity/Risk and Manager to flag glaring contradictions or hallucinations. (Considering)
- Perhaps combine both the MarketOpportunityAgent and MarketRiskAgent into a single agent? (Considering)

---

## 🧠 Agents Used


---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## Author

This project was co-authored by [Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804)

---
