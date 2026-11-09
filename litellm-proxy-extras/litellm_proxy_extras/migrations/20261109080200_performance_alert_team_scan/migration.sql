CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_DailyTeamSpend_date_id_idx"
    ON "LiteLLM_DailyTeamSpend" (date, id);
