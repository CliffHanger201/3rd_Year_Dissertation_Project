# 3rd_Year_Dissertation_Project
This is my final year dissertation project

## Commands

**To run the Python Code:**
python -m data_and_visualisation.src.run

Run from root directory

javac -cp "C:\Users\xxx\AppData\Local\Programs\Python\Python313\share\py4j\py4j0.10.9.5.jar" bridge\*.java

**The following commands are ran through the data_and_visualisation directory**

**To compile the Py4J gateway, run this command:**
javac -cp ".;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar" -d out src\bridge\*.java

**To start the Py4J gateway, run this command:**

<!-- java -cp "out;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar" bridge.HyflexGateway -->

java -cp "out;..\py4j0.10.9.9.jar;..\hyflex\hyflex\build\libs\hyflex-1.0-SNAPSHOT.jar;..\hyflex\hyflex-chesc-2011\build\libs\hyflex-chesc-2011-1.0-SNAPSHOT.jar;..\hyflex\hyflex-ext\build\libs\hyflex-ext-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-phunter\build\libs\hyflex-hh-phunter-1.0-SNAPSHOT.jar;..\hyflex\hyflex-hyperheuristics\hyflex-hh-genhive\build\libs\hyflex-hh-genhive-1.0-SNAPSHOT.jar;src\bridge\slf4j-api-2.0.17.jar;src\bridge\slf4j-simple-2.0.0-alpha6.jar" bridge.HyflexGateway



### Compilation Checks
javap -cp "..\hyflex\hyflex-hyperheuristics\hyflex-hh-adaphh\build\libs\hyflex-hh-adaphh-1.0-SNAPSHOT.jar" be.kuleuven.kahosl.acceptance.AcceptanceCriterionType