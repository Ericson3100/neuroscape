package dev.ericson.neuroscape.training;

public class TooManySessionsException extends RuntimeException {

    public TooManySessionsException(int limit) {
        super("Session limit of " + limit + " reached; try again later");
    }
}
