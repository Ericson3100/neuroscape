package dev.ericson.neuroscape.training;

public class SessionNotFoundException extends RuntimeException {

    public SessionNotFoundException(String id) {
        super("No training session with id " + id);
    }
}
