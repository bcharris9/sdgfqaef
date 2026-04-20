# Capstone RAG System

This folder contains the retrieval-augmented generation side of the final
project. Its purpose is to answer lab-manual questions by retrieving relevant
manual sections and then using an LLM to respond with lab-aware context.

## Main Files

- `server.py`
  Main API/server logic for question answering.
- `embed.py`
  Builds embeddings and prepares manual chunks for retrieval.
- `supabase.py`
  Supabase integration helpers for storage and retrieval.
- `validate_rag.py`
  Validation script for measuring RAG quality against expected excerpts.
- `chat_terminal_client.py`
  Simple terminal client for testing the chat experience.
- `database.sql`
  Database schema/setup for the retrieval store.
- `MANUALS/`
  Source lab manuals and appendices used as the RAG knowledge base.

## What This Shows In The Final Project

- How the manuals were turned into retrievable chunks
- How retrieved context was selected and passed into the model
- How the RAG system was validated against expected manual answers

## Notes

- `rag_validation_results.json` is an example evaluation artifact produced by
  the validation workflow.
- Before publishing outside class, move any service credentials out of source
  files and into environment variables.
