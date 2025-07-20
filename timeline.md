# 19/07/2025 

## 15:00 

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

## 17:00

- The hybrid retrieval is now in used but yeh still need to check 
- Fix LLM temp to 0 to have fixed response 
- Now the tools are dynamically added to the prompt, not manually anymore 
- Need to check the Normalize, Planning and Fact Checking Agent again. 

--- 

# 20/07/2025

- `utils/data_mapping.py`: logic works quite nice and might be needed for letting user try this "enchance searching" when on the streamlit application. 
- `tools/company_tools.py`: this file defines 2 tools which is `exact_company_lookup_tool` and `company_rag_retrieval_tool_wrapper` for retrieving exact company information or company context from RAG.
- `tools/patent_tools.py`: has the same logic as `company_tools.py`
- `tools/hybrid_rag_tools.py`: this file defines one tools that is `hybrid_rag_retrieval_tool_wrapper` function. The tool is used to retrieve the RAG context from both the firms and the patents.
- `tools/enhanced_hybrid_rag_tools.py`: this is expected to called the Hybrid Retriever (which is of sparse and dense retrieval) BUT not works as expected. 
  - Currently, this is somewhere like the `hybrid_rag_tools.py` but with more detailed information when we perform retrieving information. This is just returned many results than the previous versions. 
  - This file provides **tools** called `enhanced_hybrid_rag_retrieval_tool` for hybrid rag retrieval with enhanced info + `company_data_with_mapping_tool` for getting company data with enhanced mapping tool func + `mapping_key_search_tool` for seraching by mapping keys tool func.
- Checking `args.legacy`:
  - Currently remove it 
  - Now we only use the `run_enhanced_workflow` in `MultiAgentRunner`
- About the current performance of the system:
  - Current problem is that because we are working only on a sample of data. So maybe the number of company data is not aligned with the company of the patents. There would be case that for example the Intel company with hojinid of 701309 is not exist in the firm data but in the patent data. 
  ==> This suggests that we should immediately add and process more company data 
  - If you want to **check for information of a specific patents** and would love to see the market potential for it, this is currently ok for using now. 
  - Currently we handle the data for all the companies, both companies with patents and those not have 
  - The feature of **Enhanced Hybrid Search Results** (open streamlit app and on Enhanced Features tab) is working well for Motohashi requirements right now. 
  - Current problem for our systems: when asking a general questions like "tell me about the patents of company ABC" --> our system cannot understand this nice. Currently, it would perform RAG retrieval on company data. BUT the expected things to be done is that, it should get the ID or the name of the company, then perform that on the RAG retrieval for patents or performing retrieving exact patents information of the company.
  - And more, for example when asking a general question like "Tell me 3 patents that related to "Chemical Waste Industry". Then make a short comparisons", we can retrieve the patents and shortly compare them nicely. 

--- 

# 21/07/2025 



