package bridge;

import py4j.GatewayServer;

public class HyflexGateway {
    public static void main(String[] args) {
        HyflexRunner runner = new HyflexRunner();
        GatewayServer server = new GatewayServer(runner);
        server.start();
        System.out.println("Hyflex Py4J Gateway started.");
    }
}