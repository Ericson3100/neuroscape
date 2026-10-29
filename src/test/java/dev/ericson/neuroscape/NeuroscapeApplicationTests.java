package dev.ericson.neuroscape;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import com.jayway.jsonpath.JsonPath;

import dev.ericson.neuroscape.model.SavedModelRepository;

/** End-to-end through the real context: H2, JPA, controllers, SSE and the SPA fallback. */
@SpringBootTest
@ActiveProfiles("test")
@AutoConfigureMockMvc
class NeuroscapeApplicationTests {

    @Autowired
    MockMvc mvc;

    @Autowired
    SavedModelRepository repository;

    @Test
    void trainSaveReloadAndPredict() throws Exception {
        String sessionId = JsonPath.read(mvc.perform(post("/api/sessions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"dataset":"GAUSSIAN","samples":200,"noise":0.0,"hiddenLayers":[4],
                                 "activation":"TANH","optimizer":"ADAM","learningRate":0.05,"batchSize":16,"seed":1}
                                """))
                .andExpect(status().isCreated())
                .andReturn().getResponse().getContentAsString(), "$.id");

        mvc.perform(post("/api/sessions/{id}/step", sessionId).param("epochs", "40"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metrics.testAccuracy").value(org.hamcrest.Matchers.greaterThan(0.9)));

        MvcResult saved = mvc.perform(post("/api/models").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"sessionId\":\"" + sessionId + "\",\"name\":\"blobs v1\"}"))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("blobs v1"))
                .andExpect(jsonPath("$.epoch").value(40))
                .andExpect(jsonPath("$.hiddenLayers", hasSize(1)))
                .andReturn();
        Integer modelId = JsonPath.read(saved.getResponse().getContentAsString(), "$.id");
        assertThat(repository.count()).isEqualTo(1);

        mvc.perform(get("/api/models")).andExpect(jsonPath("$", hasSize(1)));

        // Blob centred at (0.45, 0.45) is class 1.
        mvc.perform(post("/api/models/{id}/predict", modelId).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"x\":0.45,\"y\":0.45}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.label").value(1));

        mvc.perform(post("/api/models/{id}/load", modelId))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.snapshot.metrics.epoch").value(40))
                .andExpect(jsonPath("$.config.dataset").value("GAUSSIAN"));

        mvc.perform(delete("/api/models/{id}", modelId)).andExpect(status().isNoContent());
        mvc.perform(get("/api/models/{id}", modelId)).andExpect(status().isNotFound());
    }

    @Test
    void saveValidatesBody() throws Exception {
        mvc.perform(post("/api/models").contentType(MediaType.APPLICATION_JSON).content("{\"name\":\"\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors.sessionId").exists())
                .andExpect(jsonPath("$.errors.name").exists());
    }

    @Test
    void streamStartsAsServerSentEvents() throws Exception {
        String sessionId = JsonPath.read(mvc.perform(post("/api/sessions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"dataset":"MOONS","samples":100,"noise":0.1,"hiddenLayers":[3],
                                 "activation":"RELU","optimizer":"SGD","learningRate":0.1,"batchSize":8}
                                """))
                .andReturn().getResponse().getContentAsString(), "$.id");

        MvcResult result = mvc.perform(get("/api/sessions/{id}/stream", sessionId)
                        .accept(MediaType.TEXT_EVENT_STREAM))
                .andExpect(request().asyncStarted())
                .andReturn();
        assertThat(result.getResponse().getContentAsString()).startsWith("event:snapshot");
        mvc.perform(delete("/api/sessions/{id}", sessionId)).andExpect(status().isNoContent());
    }

    @Test
    void healthEndpointIsUp() throws Exception {
        mvc.perform(get("/actuator/health")).andExpect(jsonPath("$.status").value("UP"));
    }

    @Test
    void clientSideRoutesFallBackToIndexHtml() throws Exception {
        assumeTrue(new ClassPathResource("static/index.html").exists(), "frontend not built");
        mvc.perform(get("/some/deep/link"))
                .andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.containsString("<div id=\"root\">")));
        mvc.perform(get("/api/unknown")).andExpect(status().isNotFound());
    }
}
