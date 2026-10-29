package dev.ericson.neuroscape.model;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface SavedModelRepository extends JpaRepository<SavedModel, Long> {

    List<SavedModel> findAllByOrderByCreatedAtDesc();
}
