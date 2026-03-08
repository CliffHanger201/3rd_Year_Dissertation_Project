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
