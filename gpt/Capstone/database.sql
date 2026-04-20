CREATE TABLE spice_files (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  circuit_name text NOT NULL,
  created_at timestamp default now()
);

CREATE TABLE spice_nets (
  id serial PRIMARY KEY,
  spice_id uuid REFERENCES spice_files(id) ON DELETE CASCADE,
  node_name text,
  simulated_voltage float,
  measured_voltage float,
  simulated_avg float,
  measured_avg float,
  simulated_max float,
  measured_max float,
  simulated_min float,
  measured_min float,
  simulated_pp float,
  measured_pp float,
  simulated_rms float,
  measured_rms float,
  simulated_period float,
  measured_period float,
  simulated_freq float,
  measured_freq float
);

CREATE TABLE spice_elements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  spice_id uuid REFERENCES spice_files(id) ON DELETE CASCADE,
  element_name text,
  type text,
  model text,
  value text,
  parameters jsonb,
  simulated_current float,
  measured_current float,
  numeric_value float
);

CREATE TABLE element_connections (
  id serial PRIMARY KEY,
  element_id uuid REFERENCES spice_elements(id) ON DELETE CASCADE,
  node_name text,
  node_order int
);

CREATE TABLE lab_sections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lab_name text,
  manual_version text,
  section_name text,
  content text,
  heading text,
  chunk_hash text UNIQUE,
  chunk_order int,
  token_count int,
  embedding vector(1024),
  page_num int2
);

CREATE INDEX IF NOT EXISTS lab_sections_lab_idx ON lab_sections (lab_name, section_name);
CREATE INDEX IF NOT EXISTS lab_sections_version_idx ON lab_sections (manual_version);
CREATE INDEX IF NOT EXISTS lab_sections_chunk_order_idx ON lab_sections (lab_name, manual_version, chunk_order);
CREATE INDEX IF NOT EXISTS lab_sections_embedding_idx ON lab_sections USING ivfflat (embedding vector_cosine_ops) WITH (lists=200);

DROP FUNCTION IF EXISTS match_lab_manuals;

CREATE OR REPLACE FUNCTION match_lab_manuals (
  query_embedding vector(1024),
  match_threshold float DEFAULT 0.62,
  match_count int DEFAULT 12,
  filter_lab_name text DEFAULT NULL,
  filter_section  text DEFAULT NULL,
  filter_manual_version text DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  content text,
  lab_name text,
  section_name text,
  manual_version text,
  page_num int2,
  similarity float
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
  RETURN QUERY
  SELECT
    ls.id,
    ls.content,
    ls.lab_name,
    ls.section_name,
    ls.manual_version,
    ls.page_num,
    1 - (ls.embedding <=> query_embedding) AS similarity
  FROM lab_sections ls
  WHERE
    ls.embedding IS NOT NULL
    AND 1 - (ls.embedding <=> query_embedding) >= match_threshold
    AND (filter_lab_name IS NULL OR ls.lab_name ILIKE '%' || filter_lab_name || '%')
    AND (filter_section  IS NULL OR ls.section_name ILIKE '%' || filter_section  || '%')
    AND (filter_manual_version IS NULL OR ls.manual_version = filter_manual_version)
  ORDER BY ls.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
