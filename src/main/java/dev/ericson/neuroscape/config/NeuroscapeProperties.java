package dev.ericson.neuroscape.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Tunables for the live training loop.
 *
 * @param gridResolution     side length of the prediction grid streamed for the decision surface
 * @param tickMillis         period of the training scheduler
 * @param epochsPerTick      epochs trained per tick for every running session
 * @param maxSessions        upper bound on concurrent in-memory sessions
 * @param sessionIdleTimeout idle sessions without subscribers are evicted after this long
 * @param sseTimeout         lifetime of one SSE connection (the browser reconnects automatically)
 * @param maxHistory         metric points kept per session before the history is down-sampled
 */
@ConfigurationProperties("neuroscape")
public record NeuroscapeProperties(
        @DefaultValue("48") int gridResolution,
        @DefaultValue("80") long tickMillis,
        @DefaultValue("2") int epochsPerTick,
        @DefaultValue("64") int maxSessions,
        @DefaultValue("30m") Duration sessionIdleTimeout,
        @DefaultValue("30m") Duration sseTimeout,
        @DefaultValue("600") int maxHistory) {
}
