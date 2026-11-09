CREATE TABLE "LiteLLM_PerformanceAlertWork" (
    day date NOT NULL,
    resource_hash varchar(64) NOT NULL,
    priority double precision NOT NULL,
    data jsonb NOT NULL,
    evaluated boolean NOT NULL DEFAULT false,
    PRIMARY KEY (day, resource_hash)
);
CREATE INDEX "LiteLLM_PerformanceAlertWork_day_evaluated_priority_resource__idx"
    ON "LiteLLM_PerformanceAlertWork" (day, evaluated, priority DESC, resource_hash);
