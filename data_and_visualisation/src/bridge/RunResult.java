package bridge;

import java.util.Arrays;

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

    @Override
    public String toString() {
        return "RunResult{" +
                "hyperHeuristicName='" + hyperHeuristicName + '\'' +
                ", domainName='" + domainName + '\'' +
                ", seed=" + seed +
                ", instanceId=" + instanceId +
                ", timeLimitMs=" + timeLimitMs +
                ", memorySize=" + memorySize +
                ", initIndices=" + Arrays.toString(initIndices) +
                ", wallTimeSeconds=" + wallTimeSeconds +
                ", bestValue=" + bestValue +
                ", bestSolutionString='" + bestSolutionString + '\'' +
                '}';
    }
}