package dev.ericson.neuroscape.model;

public class ModelNotFoundException extends RuntimeException {

    public ModelNotFoundException(long id) {
        super("No saved model with id " + id);
    }
}
