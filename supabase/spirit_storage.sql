-- Persistent Kindred Spirit storage.
-- Run this file's contents in the Supabase SQL editor.

alter table public.profiles
  add column if not exists spirit_storage_path text,
  add column if not exists spirit_created_at timestamptz,
  add column if not exists spirit_configuration jsonb;

insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'kindred-spirits',
  'kindred-spirits',
  false,
  5242880,
  array['image/png']
)
on conflict (id) do update set
  name = excluded.name,
  public = false,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;