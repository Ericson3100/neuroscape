package dev.ericson.neuroscape;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class NeuroscapeApplication {

    public static void main(String[] args) {
        SpringApplication.run(NeuroscapeApplication.class, args);
    }
}
