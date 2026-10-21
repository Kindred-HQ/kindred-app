-- Daily free-message cap.
-- In the Supabase SQL editor, paste THIS FILE'S CONTENTS and Run.
-- Do not type the filename (supabase/daily_usage.sql) — that is not SQL.
-- Kindred calls these functions from Streamlit with the anon key.
-- Row access is denied; only the security-definer functions can read or write.

create table if not exists public.daily_usage (
  fingerprint text not null,
  usage_date date not null,
  message_count integer not null default 0,
  updated_at timestamptz not null default now(),
  primary key (fingerprint, usage_date)
);

alter table public.daily_usage enable row level security;

revoke all on table public.daily_usage from public, anon, authenticated;

create or replace function public.get_daily_message_count(p_fingerprint text)
returns integer
language plpgsql
stable
security definer
set search_path = public
as $$
declare
  n integer;
begin
  select message_count into n
  from daily_usage
  where fingerprint = p_fingerprint
    and usage_date = (timezone('Africa/Johannesburg', now()))::date;
  return coalesce(n, 0);
end;
$$;

create or replace function public.consume_daily_message(p_fingerprint text, p_limit integer)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  today date := (timezone('Africa/Johannesburg', now()))::date;
  new_count integer;
  current_count integer;
begin
  if p_limit is null or p_limit < 1 then
    p_limit := 15;
  end if;

  insert into daily_usage as d (fingerprint, usage_date, message_count)
  values (p_fingerprint, today, 1)
  on conflict (fingerprint, usage_date)
  do update set
    message_count = d.message_count + 1,
    updated_at = now()
  where d.message_count < p_limit
  returning d.message_count into new_count;

  if new_count is not null then
    return jsonb_build_object(
      'allowed', true,
      'count', new_count,
      'limit', p_limit
    );
  end if;

  select d.message_count into current_count
  from daily_usage d
  where d.fingerprint = p_fingerprint and d.usage_date = today;

  return jsonb_build_object(
    'allowed', false,
    'count', coalesce(current_count, p_limit),
    'limit', p_limit
  );
end;
$$;

revoke all on function public.get_daily_message_count(text) from public;
revoke all on function public.consume_daily_message(text, integer) from public;
grant execute on function public.get_daily_message_count(text) to anon, authenticated;
grant execute on function public.consume_daily_message(text, integer) to anon, authenticated;
