package bridge;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;

import BinPacking.BinPacking;
import travelingSalesmanProblem.TSP;
import VRP.VRP;
import SAT.SAT;

// CHeSC 2011 hyper-heuristics
import be.kuleuven.kahosl.hyperheuristic.GIHH;
import pearlHunter.PearlHunter;
import csput.CSPUTGeneticHiveHyperHeuristic;

public class HyflexRunner {

    public HyflexRunner() {}

    public String[] listDomains() {
        return new String[] { "BinPacking", "TSP", "VRP", "SAT" };
    }

    public String[] listHyperHeuristics() {
        return new String[] { "AdapHH", "PHunter", "GenHive" };
    }

    public RunResult runHyperHeuristic(
            String hhName,
            String domainName,
            long seed,
            long timeLimitMs,
            int instanceId,
            int memorySize,
            int[] initIndices
    ) {
        ProblemDomain problem = createDomain(domainName, seed);
        HyperHeuristic hh = createHyperHeuristic(hhName, seed);

        if (problem == null) {
            throw new IllegalArgumentException("Unknown domain: " + domainName);
        }
        if (hh == null) {
            throw new IllegalArgumentException("Unknown hyper-heuristic: " + hhName);
        }

        problem.loadInstance(instanceId);
        problem.setMemorySize(memorySize);
        for (int idx : initIndices) {
            problem.initialiseSolution(idx);
        }

        hh.setTimeLimit(timeLimitMs);
        hh.loadProblemDomain(problem);

        long startNs = System.nanoTime();
        hh.run();
        long endNs = System.nanoTime();

        RunResult result = new RunResult();
        result.hyperHeuristicName = hhName;
        result.domainName        = domainName;
        result.seed              = seed;
        result.instanceId        = instanceId;
        result.timeLimitMs       = timeLimitMs;
        result.memorySize        = memorySize;
        result.initIndices       = initIndices;
        result.wallTimeSeconds   = (endNs - startNs) / 1_000_000_000.0;
        result.bestValue         = problem.getBestSolutionValue();

        try {
            result.bestSolutionString = problem.bestSolutionToString();
        } catch (Exception e) {
            result.bestSolutionString = null;
        }

        return result;
    }

    private ProblemDomain createDomain(String domainName, long seed) {
        switch (domainName.toLowerCase()) {
            case "binpacking":
            case "binpack":
                return new BinPacking(seed);
            case "tsp":
                return new TSP(seed);
            case "vrp":
                return new VRP(seed);
            case "sat":
                return new SAT(seed);
            default:
                return null;
        }
    }

    private HyperHeuristic createHyperHeuristic(String hhName, long seed) {
        switch (hhName.toLowerCase()) {
            case "adaphh":
            case "adapthh":
                return new GIHH(seed);
            case "phunter":
                return new PearlHunter(seed);
            case "genhive":
                return new CSPUTGeneticHiveHyperHeuristic(seed);
            default:
                return null;
        }
    }
}