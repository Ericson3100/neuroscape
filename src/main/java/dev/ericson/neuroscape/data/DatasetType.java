package dev.ericson.neuroscape.data;

/** The 2D toy problems the network can be trained on. All live roughly inside [-1, 1]². */
public enum DatasetType {
    SPIRAL,
    CIRCLES,
    XOR,
    GAUSSIAN,
    MOONS
}
