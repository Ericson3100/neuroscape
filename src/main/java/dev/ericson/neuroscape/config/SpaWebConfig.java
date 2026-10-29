package dev.ericson.neuroscape.config;

import java.io.IOException;

import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.servlet.resource.PathResourceResolver;

/**
 * Serves the bundled React app and falls back to {@code index.html} for client-side routes,
 * so deep links work. API and actuator paths are never rewritten.
 */
@Configuration
public class SpaWebConfig implements WebMvcConfigurer {

    private static final String STATIC = "classpath:/static/";

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/**")
                .addResourceLocations(STATIC)
                .resourceChain(true)
                .addResolver(new SpaFallbackResolver());
    }

    static final class SpaFallbackResolver extends PathResourceResolver {

        @Override
        protected Resource getResource(String resourcePath, Resource location) throws IOException {
            Resource requested = location.createRelative(resourcePath);
            if (requested.exists() && requested.isReadable()) {
                return requested;
            }
            boolean reserved = resourcePath.startsWith("api/") || resourcePath.startsWith("actuator");
            boolean looksLikeFile = resourcePath.substring(resourcePath.lastIndexOf('/') + 1).contains(".");
            if (reserved || looksLikeFile) {
                return null;
            }
            Resource index = new ClassPathResource("static/index.html");
            return index.exists() ? index : null;
        }
    }
}
