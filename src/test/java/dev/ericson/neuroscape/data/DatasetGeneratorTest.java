package dev.ericson.neuroscape.data;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class DatasetGeneratorTest {

    @ParameterizedTest
    @EnumSource(DatasetType.class)
    void producesBalancedSplitInsideThePlane(DatasetType type) {
        Dataset data = DatasetGenerator.generate(type, 300, 0.1, 11);

        assertThat(data.train()).hasSize(240);
        assertThat(data.test()).hasSize(60);
        long positives = data.train().stream().filter(s -> s.label() == 1).count()
                + data.test().stream().filter(s -> s.label() == 1).count();
        assertThat(positives).isBetween(120L, 180L);
        assertThat(data.train()).allSatisfy(s -> {
            assertThat(Math.abs(s.x())).isLessThan(1.6);
            assertThat(Math.abs(s.y())).isLessThan(1.6);
        });
    }

    @ParameterizedTest
    @EnumSource(DatasetType.class)
    void isDeterministicForASeed(DatasetType type) {
        assertThat(DatasetGenerator.generate(type, 100, 0.2, 5))
                .isEqualTo(DatasetGenerator.generate(type, 100, 0.2, 5));
    }
}
