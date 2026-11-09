CREATE INDEX CONCURRENTLY IF NOT EXISTS "LiteLLM_DailyUserSpend_date_id_idx"
    ON "LiteLLM_DailyUserSpend" (date, id);
