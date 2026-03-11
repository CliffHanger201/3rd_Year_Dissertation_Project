package bridge;

import java.util.Arrays;
import java.util.List;

public class RunResult {
    public String hyperHeuristicName;
    public String domainName;
    public long seed;
    public int instanceId;
    public long timeLimitMs;
    public int memorySize;
    public int[] initIndices;
    public double wallTimeSeconds;
    public double bestValue;
    public String bestSolutionString;
    public List<Double> fitnessTrace;
    public int[] heuristicCallCounts;
    public int[] heuristicCallTimes;

    public RunResult() {}

    // Getters for py4j compatibility
    public String getHyperHeuristicName() { return hyperHeuristicName; }
    public String getDomainName()         { return domainName; }
    public long   getSeed()               { return seed; }
    public int    getInstanceId()         { return instanceId; }
    public long   getTimeLimitMs()        { return timeLimitMs; }
    public int    getMemorySize()         { return memorySize; }
    public int[]  getInitIndices()        { return initIndices; }
    public double getWallTimeSeconds()    { return wallTimeSeconds; }
    public double getBestValue()          { return bestValue; }
    public String getBestSolutionString() { return bestSolutionString; }
    public List<Double> getFitnessTrace() { return fitnessTrace; }
    public int[] getHeuristicCallCounts() { return heuristicCallCounts; }
    public int[] getHeuristicCallTimes() { return heuristicCallTimes; }

    // Setters for py4j compatability
    public void setFitnessTrace(List<Double> t) { this.fitnessTrace = t; }
    public void setHeuristicCallCounts(int[] c) { this.heuristicCallCounts = c; }
    public void setHeuristicCallTimes(int[] t) { this.heuristicCallTimes = t; }

    @Override
    public String toString() {
        return "RunResult{" +
                "hyperHeuristicName='" + hyperHeuristicName + '\'' +
                ", domainName='"       + domainName          + '\'' +
                ", seed="              + seed                       +
                ", instanceId="        + instanceId                 +
                ", timeLimitMs="       + timeLimitMs                +
                ", memorySize="        + memorySize                 +
                ", initIndices="       + Arrays.toString(initIndices)     +
                ", wallTimeSeconds="   + wallTimeSeconds             +
                ", bestValue="         + bestValue                  +
                ", bestSolutionString='" + bestSolutionString + '\'' +
                ", fitnessTrace.size=" + (fitnessTrace      != null ? fitnessTrace.size()              : "null") +
                ", heuristicCallCounts=" + Arrays.toString(heuristicCallCounts)                                  +
                ", heuristicCallTimes="  + Arrays.toString(heuristicCallTimes)                                   +
                '}';
    }
}