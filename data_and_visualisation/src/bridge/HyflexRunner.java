package bridge;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;

import BinPacking.BinPacking;
import travelingSalesmanProblem.TSP;
import VRP.VRP;
import SAT.SAT;

// CHeSC 2011 hyper-heuristics
import be.kuleuven.kahosl.hyperheuristic.GIHH; // AdapHH-GIHH
import pearlhunter.PearlHunter;                // PHunter
import csput.CSPUTGeneticHiveHyperHeuristic;   // GenHive

// adapHH tools
import be.kuleuven.kahosl.acceptance.AcceptanceCriterionType;
import be.kuleuven.kahosl.selection.SelectionMethodType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class HyflexRunner {

    /** How often (ms) the background thread samples getBestSolutionValue(). */
    private static final int POLL_INTERVAL_MS = 250;

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

        if (problem == null) {
            throw new IllegalArgumentException("Unknown domain: " + domainName);
        }

        problem.loadInstance(instanceId);
        problem.setMemorySize(memorySize);

        int numberOfHeuristics = problem.getNumberOfHeuristics();

        HyperHeuristic hh = createHyperHeuristic(hhName, seed, timeLimitMs, numberOfHeuristics);

        if (hh == null) {
            throw new IllegalArgumentException("Unknown hyper-heuristic: " + hhName);
        }

        for (int idx : initIndices) {
            problem.initialiseSolution(idx);
        }

        hh.setTimeLimit(timeLimitMs);
        hh.loadProblemDomain(problem);

        // --- Fitness-trace polling thread ---
        // Because AdapHH/PHunter/GenHive are third-party we cannot instrument
        // their inner loops, so we sample getBestSolutionValue() from a
        // background thread every POLL_INTERVAL_MS milliseconds.
        final List<Double> fitnessTrace = new ArrayList<>();
        final AtomicBoolean running     = new AtomicBoolean(true);

        Thread poller = new Thread(() -> {
            double bestSeen = Double.MAX_VALUE;
            while (running.get()) {
                try {
                    double current = problem.getBestSolutionValue();
                    // Record best-so-far so the trace is monotonically non-increasing
                    if (current < bestSeen) {
                        bestSeen = current;
                    }
                    synchronized (fitnessTrace) {
                        fitnessTrace.add(bestSeen);
                    }
                    Thread.sleep(POLL_INTERVAL_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception ignored) {
                    // getBestSolutionValue() may throw before first solution is set;
                    // just keep polling.
                }
            }
        });
        poller.setDaemon(true); // won't block JVM shutdown if something goes wrong
        poller.start();

        // --- Run ---
        long startNs = System.nanoTime();
        hh.run();
        long endNs = System.nanoTime();

        // Stop the poller and wait for it to finish its current sleep
        running.set(false);
        try {
            poller.join(POLL_INTERVAL_MS * 2L);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }

        // ------- Collect results --------
        RunResult result = new RunResult();
        result.hyperHeuristicName = hhName;
        result.domainName         = domainName;
        result.seed               = seed;
        result.instanceId         = instanceId;
        result.timeLimitMs        = timeLimitMs;
        result.memorySize         = memorySize;
        result.initIndices        = initIndices;
        result.wallTimeSeconds    = (endNs - startNs) / 1_000_000_000.0;
        result.bestValue          = problem.getBestSolutionValue();

        // Fitness trace recorded by the poller
        synchronized (fitnessTrace) {
            result.setFitnessTrace(new ArrayList<>(fitnessTrace));
        }

        // Heuristic call counts + times — exposed directly by ProblemDomain
        try {
            result.setHeuristicCallCounts(problem.getHeuristicCallRecord());
        } catch (Exception e) {
            result.setHeuristicCallCounts(new int[0]);
        }

        try {
            result.setHeuristicCallTimes(problem.getheuristicCallTimeRecord());
        } catch (Exception e) {
            result.setHeuristicCallTimes(new int[0]);
        }

        try {
            result.bestSolutionString = problem.bestSolutionToString();
        } catch (Exception e) {
            result.bestSolutionString = null;
        }

        return result;
    }

    // ----------- Factory methods -----------

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

    private HyperHeuristic createHyperHeuristic(
            String hhName, long seed, long timeLimitMs, int numberOfHeuristics) {
        switch (hhName.toLowerCase()) {
            case "adaphh":
            case "adapthh":
                return new GIHH(
                    seed,
                    numberOfHeuristics,
                    timeLimitMs,
                    "GIHH",
                    SelectionMethodType.AdaptiveLimitedLAassistedDHSMentorSTD,
                    AcceptanceCriterionType.AdaptiveIterationLimitedListBasedTA
                );
            case "phunter":
                return new PearlHunter(seed);
            case "genhive":
                return new CSPUTGeneticHiveHyperHeuristic(seed);
            default:
                return null;
        }
    }
}