batch_effects.py

Usage guidelines:

- Input: CSV. Expects a site column (hospital) and an outcome column. Covariates can be optionally mentioned.
 
Usage:

- ``` pip install google-genai pandas numpy scipy scikit-learn matplotlib```
- ```export GEMINI_API_KEY=...            ```
-  ``` python batch_agents.py data.csv --site site --outcome outcome --covariates age stage```
-    ```python batch_agents.py data.csv --mode baseline     # single-agent baseline```
-    ```python batch_agents.py data.csv --mode both         # run both, save both verdicts```