# 3rd Year Dissertation Project  
## Pre-Trained Hyper-Heuristics and Transfer Learning

This repository contains my final-year dissertation project, which investigates the feasibility of **pre-training hyper-heuristics** and evaluates the impact of **transfer learning** on their performance across multiple problem domains.

The project includes:
- A custom-built **Choice Function Hyper-Heuristic (CF-HH)**
- A **pre-training framework** for this hyper-heuristic
- Three established hyper-heuristics from the **HyFlex CHeSC 2011 competition**:
  - AdapHH (AdapHH-GIHH)
  - PHunter (PearlHunter)
  - GenHive

---

## ⏱️ Approximate Runtime

| Component                              | Runtime |
|----------------------------------------|--------|
| Choice Function Hyper-Heuristic        | ~4 hours |
| HyFlex Hyper-Heuristics (all three)    | ~3 hours |
| Pre-Trained CF Hyper-Heuristic         | ~7 hours |
| Pre-Training Extension                 | ~5 hours |
| **Total Runtime**                      | **~19 hours** |

---

## ⚙️ System Configuration

### Programming Languages
- Python  
- Java  
*(Additional languages may appear within the HyFlex repository)*

### Versions
- Python: 3.13.2  
- Java: 17.0.12  

### Development Environment
- Visual Studio Code (VSCode)

### Dependencies
All required Python packages are listed in: **Requirements.txt**

---

## ⚠️ Setup Recommendation

Before running any HyFlex-related code, ensure the HyFlex project is built correctly.

Refer to:
https://github.com/seage/hyflex

---

## 🧩 Problem Domains

| Problem Domain | Number of Heuristics |
|---------------|---------------------|
| Bin Packing   | 10 |
| SAT           | 11 |
| TSP           | 13 |
| VRP           | 8 |

---

## 🚀 Execution Commands

### 1. Base Hyper-Heuristic Run

Run: 
```
python -m data_and_visualisation.src.run
```

Output:

*python_hh_all_domains_results.json*

To visualise results:
```
python -m data_and_visualisation.src.visualisation
```

Visualisations include:
- Fitness trace  
- Heuristic call counts  
- Heuristic runtimes  

---

### 2. Pre-Trained Hyper-Heuristic

Run:
```
python -m data_and_visualisation.src.run_pretrained
```

---

### 3. HyFlex Hyper-Heuristics

#### Step 1: Compile Py4J Gateway
```
javac -cp ".;py4j0.10.9.9.jar;hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar" -d data_and_visualisation\out data_and_visualisation\src\bridge\*.java
```
#### Step 2: Start Py4J Gateway
```
java -cp "data_and_visualisation\out;py4j0.10.9.9.jar;hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;hyflex\hyflex-ext\build\libs\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar;data_and_visualisation\src\bridge\slf4j-api-2.0.17.jar;data_and_visualisation\src\bridge\slf4j-simple-2.0.0-alpha6.jar" bridge.HyflexGateway
```
#### Step 3: Run Python Interface (new terminal)
```
python -m data_and_visualisation.src.run_hyflex
```
#### Visualisation
```
python -m data_and_visualisation.src.pretrained_visualisation
```
---

### 4. Pre-Training Extension (Frozen Zero Q-Table Transfer)

Run: 
```
python -m data_and_visualisation.src.run_pretraining_extension
```
Visualise:
```
python -m data_and_visualisation.src.extension_visualisation
```
---

## 🧪 Compilation Check
```
javap -cp "\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar" be.kuleuven.kahosl.acceptance.AcceptanceCriterionType
```
---

## 🐞 Known Issues

- **TensorFlow Installation Issue (Windows)**  
  If installation fails due to long file path limitations:
  - Move the project to a shorter directory path  
  - Or enable long file paths in Windows settings  

---

## 📚 References

[1] Misir, M., De Causmaecker, P., Vanden Berghe, G., & Verbeeck, K.  
*An adaptive hyper-heuristic for CHeSC 2011.* OR53 Annual Conference, Nottingham, UK, 2011.

[2] Chan, C.-Y., Xue, F., Ip, W. H., & Cheung, C. F.  
*A hyper-heuristic inspired by pearl hunting.* LION Conference, 2012.

[3] Frankiewicz, M., et al.  
*Genetic Hive Hyper-Heuristic.* 2011.

---

## 📌 Notes

This project explores a relatively underdeveloped area of hyper-heuristic research.  
The findings highlight both the **potential and limitations of pre-training**, and open up opportunities for future work in **transfer learning for general-purpose optimisation systems**.

