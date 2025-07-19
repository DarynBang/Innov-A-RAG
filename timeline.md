# 19/07/2025 

- Currently, the **retrieval folders** contains just the code that maybe would later used for Hybrid Search. **BUT currently maybe it is not used yet**. 

- We set all the **LLMs with temperature 0.1** currently. If you want the **fixed response** so maybe we should fix this.

- The **Planning Agent**: 
  - Should check if it nicely or correctly splitting the user question into multi subquestions later 
  - 

- The **Normalized Query Agent**
  - Currently the tools are manually defined in the PROMPTS, which is not good for extended used. should fix.
  - TOOLS / FUNCTIONS are currently registered to the Multi Agent Runner 
  - Normalized Agent workflow: normalize the query --> get tools and execute the tools
    ```json
    Query: "What are the latest trends in AI patents?"
    Output: {
        "query_type": "general",
        "identifiers": {
            "companies": [],
            "patents": []
        },
        "recommended_tools": ["patent_rag_retrieval", "hybrid_rag_retrieval"],
        "reasoning": "General industry trend query requiring broad patent analysis"
    }
    ```
    => Currently this in the Normalized query agent is not good for handling yet. What i would like is that, maybe the ```patent_rag_retrieval``` would give some patents. But to get the full information of each patent, we should also need to execute the ```exact_patent_lookup```
    => Also in the examples given in the PROMPTS for the normalized agent, we have missed the case of existing both the patents and the firms in the query. 

- The **Generalized Agent**
  - Code of ```_format_contexts``` not understand yet --> do not know if it is correct or not 
  - More specifically, not know why we need to check ```# Handle structured results if 'chunks' in result: ... ```

- 2 agents **Market Opportunity** and **Market Risks** is somewhat nearly the same. Just difference at the PROMPTs

- The **Market Manager Agent** contains lots of information in its prompts.
  - Maybe should define and confim which is needed which is not 
  - Currently we did not ask it to suggest about the technology or products based on the current market analysis yet 

- The **Fact Checking Agent** 
  - currently need to perform 2nd check on many things including sources. But it currently does not read the sources or any thing yet. It just read something like "Source: Company A, Patent B" but does not actually have accessed or reading the data source to check if exactly the sources there ? 
  - does not see ```is_response_reliable```, ```get_validation_summary``` and ```get_validation_summary``` use anywhere but already defined 

- Later need to change all the **printing stmts** now into the ```logger.info()```

--- 


