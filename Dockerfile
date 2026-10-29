# syntax=docker/dockerfile:1

# ---- build: Maven wrapper builds the React app (via frontend-maven-plugin) and the Spring Boot jar ----
FROM eclipse-temurin:21-jdk AS build
WORKDIR /workspace
COPY .mvn .mvn
COPY mvnw pom.xml ./
RUN chmod +x mvnw && ./mvnw -B -q dependency:go-offline
COPY frontend/package.json frontend/package-lock.json frontend/
COPY frontend frontend
COPY src src
RUN ./mvnw -B -DskipTests package && cp target/neuroscape-*.jar app.jar

# ---- run: slim JRE, non-root, H2 file database on a volume ----
FROM eclipse-temurin:21-jre
RUN useradd --system --uid 10001 neuroscape && mkdir /data && chown neuroscape /data
USER neuroscape
WORKDIR /app
COPY --from=build /workspace/app.jar app.jar
ENV NEUROSCAPE_DATA_DIR=/data
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=75", "-jar", "/app/app.jar"]
