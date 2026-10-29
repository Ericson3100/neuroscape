package dev.ericson.neuroscape.api;

import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.startsWith;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import com.jayway.jsonpath.JsonPath;

import dev.ericson.neuroscape.config.NeuroscapeProperties;
import dev.ericson.neuroscape.training.TrainingService;

@WebMvcTest(SessionController.class)
@ActiveProfiles("test")
@Import(TrainingService.class)
@EnableConfigurationProperties(NeuroscapeProperties.class)
class SessionControllerTest {

    static final String VALID = """
            {"dataset":"XOR","samples":100,"noise":0.1,"hiddenLayers":[4,4],
             "activation":"TANH","optimizer":"ADAM","learningRate":0.03,"batchSize":10,"seed":42}
            """;

    @Autowired
    MockMvc mvc;

    @Test
    void createsSessionWithDatasetAndFirstFrame() throws Exception {
        mvc.perform(post("/api/sessions").contentType(MediaType.APPLICATION_JSON).content(VALID))
                .andExpect(status().isCreated())
                .andExpect(header().string("Location", startsWith("/api/sessions/")))
                .andExpect(jsonPath("$.config.seed").value(42))
                .andExpect(jsonPath("$.train", hasSize(80)))
                .andExpect(jsonPath("$.test", hasSize(20)))
                .andExpect(jsonPath("$.snapshot.layers", hasSize(3)))
                .andExpect(jsonPath("$.snapshot.metrics.epoch").value(0));
    }

    @Test
    void rejectsInvalidConfigWithFieldErrors() throws Exception {
        String invalid = """
                {"dataset":"XOR","samples":5,"noise":0.9,"hiddenLayers":[0],
                 "activation":"TANH","optimizer":"ADAM","learningRate":0,"batchSize":10}
                """;
        mvc.perform(post("/api/sessions").contentType(MediaType.APPLICATION_JSON).content(invalid))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.title").value("Validation failed"))
                .andExpect(jsonPath("$.errors.samples").exists())
                .andExpect(jsonPath("$.errors.noise").exists())
                .andExpect(jsonPath("$.errors.learningRate").exists());
    }

    @Test
    void rejectsUnknownEnumValue() throws Exception {
        mvc.perform(post("/api/sessions").contentType(MediaType.APPLICATION_JSON)
                        .content(VALID.replace("XOR", "MNIST")))
                .andExpect(status().isBadRequest());
    }

    @Test
    void stepsAndPredicts() throws Exception {
        String id = createSession();
        mvc.perform(post("/api/sessions/{id}/step", id).param("epochs", "5"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metrics.epoch").value(5));
        mvc.perform(post("/api/sessions/{id}/predict", id).contentType(MediaType.APPLICATION_JSON)
                        .content("{\"x\":0.5,\"y\":0.5}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.activations", hasSize(4)))
                .andExpect(jsonPath("$.probability").isNumber());
    }

    @Test
    void rejectsOutOfRangeStep() throws Exception {
        String id = createSession();
        mvc.perform(post("/api/sessions/{id}/step", id).param("epochs", "0"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void unknownSessionIs404ProblemDetail() throws Exception {
        mvc.perform(get("/api/sessions/does-not-exist"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.title").value("Not found"));
    }

    private String createSession() throws Exception {
        MvcResult result = mvc.perform(post("/api/sessions").contentType(MediaType.APPLICATION_JSON).content(VALID))
                .andExpect(status().isCreated())
                .andReturn();
        return JsonPath.read(result.getResponse().getContentAsString(), "$.id");
    }
}
