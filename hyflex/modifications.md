# MODIFICATIONS TO HYFLEX FRAMEWORK

This document lists all modifications made to the original HyFlex framework
for the SEAGE Hyper-Heuristic Project (3rd Year Dissertation).

## 1. Purpose of modifications
The modifications were necessary to:

- Extend **ProblemDomain interfaces** for custom hyper-heuristic support.
- Fix minor bugs or adjust methods to ensure compatibility with custom hyper-heuristics.

---

## 2. List of modifications

| File | Description of Change | Reason / Impact |
|------|---------------------|----------------|
| `hyflex\hyflex-ext\src\main\java\hfu\parsers\cfg\pep\EarleyParser.java` | Change all § into \u00A7 | Allows for Hyflex to built on Windows |
| `hyflex\hyflex-ext\src\main\java\hfu\parsers\cfg\pep\LLParser.java` | Change all § into \u00A7 | Allows for Hyflex to built on Windows |
| `hyflex\hyflex-ext\src\main\java\hfu\parsers\cfg\pep\Grammar.java` | Change all § into \u00A7 | Allows for Hyflex to built on Windows |
| `hyflex/MODIFICATIONS.md` | Created this file | Document all changes for clarity and reproducibility |

---

## 3. Notes on usage

- All modifications are **confined to specific files**; the rest of HyFlex remains unchanged.  
- Original HyFlex commit version: `<insert original commit hash>`  
- Forked HyFlex repository: [https://github.com/CliffHanger201/hyflex](https://github.com/CliffHanger201/hyflex)  
- Original HyFlex repository: [https://github.com/seage/hyflex](https://github.com/seage/hyflex)  

These modifications make the framework **ready to run experiments with the SEAGE hyper-heuristic project**, while preserving compatibility with the original HyFlex codebase.