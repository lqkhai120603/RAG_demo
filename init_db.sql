CREATE EXTENSION IF NOT EXISTS pgvector;

CREATE SEQUENCE IF NOT EXISTS documents_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

CREATE TABLE IF NOT EXISTS documents(
    id        integer      NOT NULL DEFAULT nextval('documents_id_seq'::regclass),
    content   text         NOT NULL,
    embedding vector(768),
    CONSTRAINT documents_pkey PRIMARY KEY (id)
);
