CREATE EXTENSION IF NOT EXISTS vector;

CREATE SEQUENCE documents_id_seq;

CREATE TABLE public.documents(
    id        integer      NOT NULL DEFAULT nextval('documents_id_seq'::regclass),
    content   text         NOT NULL,
    embedding vector(768),
    CONSTRAINT documents_pkey PRIMARY KEY (id)
);
