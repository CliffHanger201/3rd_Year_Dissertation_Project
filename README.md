# 3rd_Year_Dissertation_Project
This is my final year dissertation project on the possibility of pre-training a hyper-heuristic and the resulting effects of hyper-heuristic transfer learning. This project contains my own hyper-heuristic, the pre-training code for that hyper-heuristic, and three hyper-heuristics chosen from the HyFlex CHeSC 2011 competition:

- AdapHH (AdapHH-GIHH)
- PHunter (PearlHunter)
- GenHive

## Problem Domains

|-|-|
|**Problem Domains**|**Number of Heuristics**|
| Bin Packing |  |
| SAT | |
| TSP | |
| VRP | |


## Commands

### Base Hyper-Heuristic Test Run

**To run the Python Code:**

python -m data_and_visualisation.src.run

*The results will be printed off in the results folder from the root directory called*: <b><font color = orange>"python_hh_all_domains_results.json"</font></b>

**To visualise the results, run the command:**

python -m data_and_visualisation.src.visualisation

*This will only visualise the fitness trace, heuristic call counts and heuristic runtimes of only this hyper-heuristic*

### Pre-Trained Hyper-Heuristics Run

**To run the pre-trained hyper-heuristic:**

python -m data_and_visualisation.src.run_pretrained

**To visualise the results

### Hyflex Hyper-Heuristics Run

**The following commands are ran through the data_and_visualisation directory**

Run from root directory

javac -cp "C:\Users\xxx\AppData\Local\Programs\Python\Python313\share\py4j\py4j0.10.9.5.jar" bridge\*.java

**1) To compile the Py4J gateway, run this command:**

javac -cp ".;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar" -d out src\bridge\*.java

**2) To start the Py4J gateway, run this command:**

<!-- java -cp "out;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar" bridge.HyflexGateway -->

java -cp "out;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar;src\bridge\slf4j-api-2.0.17.jar;src\bridge\slf4j-simple-2.0.0-alpha6.jar" bridge.HyflexGateway

**3) On a different terminal, run:**
python -m data_and_visualisation.src.run_hyflex



### Compilation Checks
javap -cp "..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar" be.kuleuven.kahosl.acceptance.AcceptanceCriterionType

## References
[1] - 
[2] - 
[3] - 