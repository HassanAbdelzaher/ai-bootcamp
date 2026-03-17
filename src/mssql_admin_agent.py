"""
MSSQL Admin Agent
An intelligent agent for Microsoft SQL Server administration.
Uses the Claude Agent SDK with custom tools for database management.

Environment variables (required):
  MSSQL_SERVER    - Server hostname or IP (default: localhost)
  MSSQL_PORT      - Port (default: 1433)
  MSSQL_DATABASE  - Default database (default: master)
  MSSQL_USER      - SQL Server login username
  MSSQL_PASSWORD  - SQL Server login password

  Or provide a full connection string:
  MSSQL_CONNECTION_STRING - e.g. "DRIVER={ODBC Driver 18 for SQL Server};SERVER=...;..."

Usage:
  python mssql_admin_agent.py                    # Interactive mode
  python mssql_admin_agent.py "your question"    # Single query mode
"""

import os
import sys
import json
import textwrap
import anyio
import pyodbc

from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_connection_string() -> str:
    """Build MSSQL connection string from environment variables."""
    cs = os.environ.get("MSSQL_CONNECTION_STRING")
    if cs:
        return cs

    server   = os.environ.get("MSSQL_SERVER",   "localhost")
    port     = os.environ.get("MSSQL_PORT",     "1433")
    database = os.environ.get("MSSQL_DATABASE", "master")
    user     = os.environ.get("MSSQL_USER",     "")
    password = os.environ.get("MSSQL_PASSWORD", "")

    if user and password:
        auth = f"UID={user};PWD={password};"
    else:
        auth = "Trusted_Connection=yes;"

    return (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"TrustServerCertificate=yes;"
        + auth
    )


def _connect(database: str | None = None) -> pyodbc.Connection:
    """Open a pyodbc connection, optionally overriding the default database."""
    cs = _get_connection_string()
    if database:
        import re
        cs = re.sub(r"DATABASE=[^;]+;", f"DATABASE={database};", cs)
        if "DATABASE=" not in cs:
            cs += f"DATABASE={database};"
    conn = pyodbc.connect(cs, autocommit=False, timeout=30)
    conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")
    conn.setencoding(encoding="utf-8")
    return conn


def _rows_to_dicts(cursor: pyodbc.Cursor) -> list[dict]:
    """Convert cursor rows to a list of dicts (JSON-serialisable)."""
    if cursor.description is None:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _run_query(sql: str, database: str | None = None, params: list | None = None) -> dict:
    """Execute SQL and return rows + rowcount."""
    try:
        with _connect(database) as conn:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            rows = _rows_to_dicts(cur)
            rowcount = cur.rowcount
            conn.commit()
        return {"rows": rows, "rowcount": rowcount, "error": None}
    except pyodbc.Error as exc:
        return {"rows": [], "rowcount": -1, "error": str(exc)}


def _tool_result(data: dict | list | str) -> dict:
    text = json.dumps(data, default=str, indent=2) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------

@tool(
    "execute_query",
    "Execute any T-SQL statement (SELECT, INSERT, UPDATE, DELETE, DDL, etc.) "
    "against the SQL Server. Returns rows for SELECT; rowcount for DML.",
    {
        "sql":      str,
        "database": str,
    },
)
async def execute_query(args: dict) -> dict:
    sql      = args["sql"]
    database = args.get("database") or None
    result   = _run_query(sql, database=database)
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result)


@tool(
    "list_databases",
    "List all user databases on the SQL Server with size and state information.",
    {},
)
async def list_databases(_args: dict) -> dict:
    sql = textwrap.dedent("""\
        SELECT
            d.name                          AS database_name,
            d.state_desc                    AS state,
            d.recovery_model_desc           AS recovery_model,
            d.compatibility_level,
            d.create_date,
            CAST(SUM(mf.size) * 8.0 / 1024 AS DECIMAL(10,2)) AS size_mb
        FROM sys.databases d
        JOIN sys.master_files mf ON mf.database_id = d.database_id
        WHERE d.database_id > 4               -- skip system DBs (optional)
           OR d.name IN ('master','model','msdb','tempdb')
        GROUP BY d.name, d.state_desc, d.recovery_model_desc,
                 d.compatibility_level, d.create_date
        ORDER BY d.name;
    """)
    result = _run_query(sql, database="master")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "list_tables",
    "List tables and views in a database with row counts and size.",
    {"database": str},
)
async def list_tables(args: dict) -> dict:
    database = args.get("database", "master")
    sql = textwrap.dedent("""\
        SELECT
            s.name                              AS schema_name,
            t.name                              AS table_name,
            t.type_desc,
            p.rows                              AS row_count,
            CAST(
                (SUM(a.total_pages) * 8.0) / 1024
            AS DECIMAL(10,2))                   AS size_mb,
            t.create_date,
            t.modify_date
        FROM sys.tables t
        JOIN sys.schemas s  ON s.schema_id = t.schema_id
        JOIN sys.indexes i  ON i.object_id = t.object_id AND i.index_id <= 1
        JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id = i.index_id
        JOIN sys.allocation_units a ON a.container_id = p.partition_id
        GROUP BY s.name, t.name, t.type_desc, p.rows, t.create_date, t.modify_date
        ORDER BY s.name, t.name;
    """)
    result = _run_query(sql, database=database)
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_table_schema",
    "Get the column definitions, data types, nullability, and defaults for a table.",
    {"database": str, "table": str, "schema": str},
)
async def get_table_schema(args: dict) -> dict:
    database = args.get("database", "master")
    schema   = args.get("schema",   "dbo")
    table    = args["table"]
    sql = textwrap.dedent("""\
        SELECT
            c.column_id,
            c.name                      AS column_name,
            tp.name                     AS data_type,
            c.max_length,
            c.precision,
            c.scale,
            c.is_nullable,
            c.is_identity,
            c.is_computed,
            OBJECT_DEFINITION(c.default_object_id) AS column_default,
            ep.value                    AS description
        FROM sys.columns c
        JOIN sys.types tp   ON tp.user_type_id = c.user_type_id
        JOIN sys.tables t   ON t.object_id = c.object_id
        JOIN sys.schemas s  ON s.schema_id = t.schema_id
        LEFT JOIN sys.extended_properties ep
            ON ep.major_id = c.object_id
           AND ep.minor_id = c.column_id
           AND ep.name = 'MS_Description'
        WHERE s.name = ? AND t.name = ?
        ORDER BY c.column_id;
    """)
    result = _run_query(sql, database=database, params=[schema, table])
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_server_info",
    "Return SQL Server version, edition, configuration, and resource governor settings.",
    {},
)
async def get_server_info(_args: dict) -> dict:
    sql = textwrap.dedent("""\
        SELECT
            SERVERPROPERTY('ProductVersion')    AS version,
            SERVERPROPERTY('ProductLevel')      AS patch_level,
            SERVERPROPERTY('Edition')           AS edition,
            SERVERPROPERTY('EngineEdition')     AS engine_edition,
            SERVERPROPERTY('ServerName')        AS server_name,
            SERVERPROPERTY('Collation')         AS collation,
            SERVERPROPERTY('IsHadrEnabled')     AS is_hadr_enabled,
            SERVERPROPERTY('IsClustered')       AS is_clustered,
            @@CPU_BUSY                          AS cpu_busy_ticks,
            @@CONNECTIONS                       AS total_connections,
            @@MAX_CONNECTIONS                   AS max_connections,
            @@VERSION                           AS full_version;
    """)
    result = _run_query(sql, database="master")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"][0] if result["rows"] else {})


@tool(
    "get_running_queries",
    "Show currently executing queries with their duration, blocking status, and wait types.",
    {"min_duration_seconds": int},
)
async def get_running_queries(args: dict) -> dict:
    min_sec = args.get("min_duration_seconds", 0)
    sql = textwrap.dedent("""\
        SELECT
            r.session_id,
            r.status,
            r.blocking_session_id,
            r.wait_type,
            r.wait_time / 1000.0            AS wait_seconds,
            r.cpu_time / 1000.0             AS cpu_seconds,
            r.total_elapsed_time / 1000.0   AS elapsed_seconds,
            r.logical_reads,
            r.writes,
            r.row_count,
            DB_NAME(r.database_id)          AS database_name,
            s.login_name,
            s.host_name,
            s.program_name,
            SUBSTRING(qt.text, (r.statement_start_offset/2)+1,
                ((CASE r.statement_end_offset
                    WHEN -1 THEN DATALENGTH(qt.text)
                    ELSE r.statement_end_offset END
                 - r.statement_start_offset)/2)+1) AS current_statement,
            qt.text                         AS full_query_text
        FROM sys.dm_exec_requests r
        JOIN sys.dm_exec_sessions s ON s.session_id = r.session_id
        CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) qt
        WHERE r.session_id <> @@SPID
          AND r.total_elapsed_time / 1000 >= ?
        ORDER BY r.total_elapsed_time DESC;
    """)
    result = _run_query(sql, database="master", params=[min_sec])
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_blocking_queries",
    "Identify blocking chains — which sessions are blocking others and what SQL they are running.",
    {},
)
async def get_blocking_queries(_args: dict) -> dict:
    sql = textwrap.dedent("""\
        WITH BlockingChain AS (
            SELECT
                r.session_id,
                r.blocking_session_id,
                s.login_name,
                s.host_name,
                r.wait_type,
                r.wait_time / 1000.0    AS wait_seconds,
                DB_NAME(r.database_id)  AS database_name,
                qt.text                 AS sql_text
            FROM sys.dm_exec_requests r
            JOIN sys.dm_exec_sessions s ON s.session_id = r.session_id
            CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) qt
            WHERE r.blocking_session_id > 0
        )
        SELECT * FROM BlockingChain
        ORDER BY wait_seconds DESC;
    """)
    result = _run_query(sql, database="master")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_index_info",
    "Return index definitions, fragmentation, and usage statistics for a table.",
    {"database": str, "table": str, "schema": str},
)
async def get_index_info(args: dict) -> dict:
    database = args.get("database", "master")
    schema   = args.get("schema",   "dbo")
    table    = args["table"]
    sql = textwrap.dedent("""\
        SELECT
            i.name                          AS index_name,
            i.type_desc                     AS index_type,
            i.is_unique,
            i.is_primary_key,
            i.is_disabled,
            STRING_AGG(c.name, ', ')
                WITHIN GROUP (ORDER BY ic.key_ordinal) AS key_columns,
            ius.user_seeks,
            ius.user_scans,
            ius.user_lookups,
            ius.user_updates,
            ius.last_user_seek,
            ius.last_user_scan,
            CAST(ips.avg_fragmentation_in_percent AS DECIMAL(5,2)) AS fragmentation_pct,
            ips.page_count
        FROM sys.indexes i
        JOIN sys.tables t       ON t.object_id = i.object_id
        JOIN sys.schemas s      ON s.schema_id = t.schema_id
        JOIN sys.index_columns ic ON ic.object_id = i.object_id
                                 AND ic.index_id  = i.index_id
                                 AND ic.is_included_column = 0
        JOIN sys.columns c      ON c.object_id = ic.object_id
                                 AND c.column_id = ic.column_id
        LEFT JOIN sys.dm_db_index_usage_stats ius
            ON ius.object_id = i.object_id
           AND ius.index_id  = i.index_id
           AND ius.database_id = DB_ID()
        LEFT JOIN sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL, 'LIMITED') ips
            ON ips.object_id = i.object_id
           AND ips.index_id  = i.index_id
        WHERE s.name = ? AND t.name = ? AND i.type > 0
        GROUP BY
            i.name, i.type_desc, i.is_unique, i.is_primary_key, i.is_disabled,
            ius.user_seeks, ius.user_scans, ius.user_lookups, ius.user_updates,
            ius.last_user_seek, ius.last_user_scan,
            ips.avg_fragmentation_in_percent, ips.page_count
        ORDER BY i.name;
    """)
    result = _run_query(sql, database=database, params=[schema, table])
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_missing_indexes",
    "List missing index recommendations from the SQL Server DMVs for a given database.",
    {"database": str, "min_impact": float},
)
async def get_missing_indexes(args: dict) -> dict:
    database   = args.get("database", "master")
    min_impact = args.get("min_impact", 1000.0)
    sql = textwrap.dedent("""\
        SELECT TOP 20
            migs.avg_total_user_cost * migs.avg_user_impact
                * (migs.user_seeks + migs.user_scans)   AS improvement_measure,
            mid.statement                               AS table_name,
            mid.equality_columns,
            mid.inequality_columns,
            mid.included_columns,
            migs.unique_compiles,
            migs.user_seeks,
            migs.user_scans,
            migs.avg_total_user_cost,
            migs.avg_user_impact
        FROM sys.dm_db_missing_index_details mid
        JOIN sys.dm_db_missing_index_groups mig
            ON mig.index_handle = mid.index_handle
        JOIN sys.dm_db_missing_index_group_stats migs
            ON migs.group_handle = mig.index_group_handle
        WHERE DB_NAME(mid.database_id) = ?
          AND migs.avg_total_user_cost * migs.avg_user_impact
              * (migs.user_seeks + migs.user_scans) >= ?
        ORDER BY improvement_measure DESC;
    """)
    result = _run_query(sql, database="master", params=[database, min_impact])
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "list_jobs",
    "List SQL Server Agent jobs with their schedules, last run status, and duration.",
    {},
)
async def list_jobs(_args: dict) -> dict:
    sql = textwrap.dedent("""\
        SELECT
            j.name                          AS job_name,
            j.enabled,
            j.description,
            c.name                          AS category,
            l.name                          AS owner,
            jh.run_status,
            CASE jh.run_status
                WHEN 0 THEN 'Failed'
                WHEN 1 THEN 'Succeeded'
                WHEN 2 THEN 'Retry'
                WHEN 3 THEN 'Cancelled'
            END                             AS last_run_result,
            msdb.dbo.agent_datetime(jh.run_date, jh.run_time) AS last_run_datetime,
            jh.run_duration                 AS last_run_duration_hhmmss,
            js.next_run_date,
            js.next_run_time
        FROM msdb.dbo.sysjobs j
        LEFT JOIN msdb.dbo.syscategories c  ON c.category_id = j.category_id
        LEFT JOIN master.sys.server_principals l ON l.sid = j.owner_sid
        LEFT JOIN msdb.dbo.sysjobschedules js ON js.job_id = j.job_id
        LEFT JOIN (
            SELECT job_id, run_status, run_date, run_time, run_duration,
                   ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY run_date DESC, run_time DESC) AS rn
            FROM msdb.dbo.sysjobhistory
            WHERE step_id = 0
        ) jh ON jh.job_id = j.job_id AND jh.rn = 1
        ORDER BY j.name;
    """)
    result = _run_query(sql, database="msdb")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "list_logins_and_users",
    "List all SQL Server logins and their database user mappings with roles.",
    {"database": str},
)
async def list_logins_and_users(args: dict) -> dict:
    database = args.get("database", "master")
    # Server-level logins
    login_sql = textwrap.dedent("""\
        SELECT
            sp.name             AS login_name,
            sp.type_desc        AS login_type,
            sp.is_disabled,
            sp.create_date,
            sp.modify_date,
            sp.default_database_name,
            STUFF((
                SELECT ', ' + srp.name
                FROM sys.server_role_members srm
                JOIN sys.server_principals srp ON srp.principal_id = srm.role_principal_id
                WHERE srm.member_principal_id = sp.principal_id
                FOR XML PATH('')), 1, 2, '') AS server_roles
        FROM sys.server_principals sp
        WHERE sp.type IN ('S','U','G')   -- SQL, Windows user/group
        ORDER BY sp.name;
    """)
    # DB-level users
    user_sql = textwrap.dedent("""\
        SELECT
            dp.name             AS user_name,
            dp.type_desc        AS user_type,
            sp.name             AS mapped_login,
            dp.default_schema_name,
            STUFF((
                SELECT ', ' + rp.name
                FROM sys.database_role_members drm
                JOIN sys.database_principals rp ON rp.principal_id = drm.role_principal_id
                WHERE drm.member_principal_id = dp.principal_id
                FOR XML PATH('')), 1, 2, '') AS db_roles
        FROM sys.database_principals dp
        LEFT JOIN sys.server_principals sp ON sp.sid = dp.sid
        WHERE dp.type IN ('S','U','G')
        ORDER BY dp.name;
    """)
    logins = _run_query(login_sql, database="master")
    users  = _run_query(user_sql,  database=database)
    result = {
        "server_logins":    logins["rows"]  if not logins["error"]  else {"error": logins["error"]},
        "database_users":   users["rows"]   if not users["error"]   else {"error": users["error"]},
    }
    return _tool_result(result)


@tool(
    "get_wait_stats",
    "Return the top wait statistics — useful for diagnosing server-wide bottlenecks.",
    {"top_n": int},
)
async def get_wait_stats(args: dict) -> dict:
    top_n = args.get("top_n", 20)
    sql = textwrap.dedent(f"""\
        SELECT TOP {int(top_n)}
            wait_type,
            waiting_tasks_count,
            wait_time_ms,
            max_wait_time_ms,
            signal_wait_time_ms,
            wait_time_ms - signal_wait_time_ms  AS resource_wait_time_ms,
            CAST(100.0 * wait_time_ms / SUM(wait_time_ms) OVER()
                 AS DECIMAL(5,2))               AS pct_of_total
        FROM sys.dm_os_wait_stats
        WHERE wait_type NOT IN (
            'SLEEP_TASK','BROKER_TO_FLUSH','BROKER_TASK_STOP','CLR_AUTO_EVENT',
            'DISPATCHER_QUEUE_SEMAPHORE','FT_IFTS_SCHEDULER_IDLE_WAIT',
            'HADR_WORK_QUEUE','HADR_TIMER_TASK','HADR_TRANSPORT_DBTRANSPORT',
            'ONDEMAND_TASK_QUEUE','REQUEST_FOR_DEADLOCK_SEARCH','RESOURCE_QUEUE',
            'SERVER_IDLE_CHECK','SLEEP_DBSTARTUP','SLEEP_DBTASK','SLEEP_TEMPDBSTARTUP',
            'SNI_HTTP_ACCEPT','SP_SERVER_DIAGNOSTICS_SLEEP','SQLTRACE_BUFFER_FLUSH',
            'SQLTRACE_INCREMENTAL_FLUSH_SLEEP','WAIT_XTP_OFFLINE_CKPT_NEW_LOG',
            'XE_DISPATCHER_WAIT','XE_TIMER_EVENT','BROKER_EVENTHANDLER',
            'CHECKPOINT_QUEUE','DBMIRROR_EVENTS_QUEUE','SQLTRACE_WAIT_ENTRIES',
            'WAIT_XTP_RECOVERY','WAITFOR','XE_DISPATCHER_JOIN'
        )
          AND wait_time_ms > 0
        ORDER BY wait_time_ms DESC;
    """)
    result = _run_query(sql, database="master")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "get_backup_history",
    "Show recent backup history for a database.",
    {"database": str, "days": int},
)
async def get_backup_history(args: dict) -> dict:
    database = args.get("database", "master")
    days     = args.get("days", 7)
    sql = textwrap.dedent("""\
        SELECT TOP 50
            bs.database_name,
            CASE bs.type
                WHEN 'D' THEN 'Full'
                WHEN 'I' THEN 'Differential'
                WHEN 'L' THEN 'Log'
                ELSE bs.type
            END                                 AS backup_type,
            bs.backup_start_date,
            bs.backup_finish_date,
            DATEDIFF(SECOND, bs.backup_start_date, bs.backup_finish_date) AS duration_seconds,
            CAST(bs.backup_size / 1048576.0 AS DECIMAL(12,2))             AS size_mb,
            CAST(bs.compressed_backup_size / 1048576.0 AS DECIMAL(12,2))  AS compressed_size_mb,
            bs.recovery_model,
            bmf.physical_device_name
        FROM msdb.dbo.backupset bs
        JOIN msdb.dbo.backupmediafamily bmf ON bmf.media_set_id = bs.media_set_id
        WHERE bs.database_name = ?
          AND bs.backup_start_date >= DATEADD(DAY, -?, GETDATE())
        ORDER BY bs.backup_start_date DESC;
    """)
    result = _run_query(sql, database="msdb", params=[database, days])
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result(result["rows"])


@tool(
    "kill_session",
    "Kill (terminate) a SQL Server session by session_id. Use with caution.",
    {"session_id": int},
)
async def kill_session(args: dict) -> dict:
    session_id = int(args["session_id"])
    if session_id <= 50:
        return _tool_result({"error": f"Refusing to kill system session {session_id}."})
    sql = f"KILL {session_id};"
    result = _run_query(sql, database="master")
    if result["error"]:
        return _tool_result({"error": result["error"]})
    return _tool_result({"message": f"Session {session_id} killed successfully."})


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Microsoft SQL Server (MSSQL) database administrator agent.

You have access to tools that connect to a live SQL Server instance. Use them to:
- Investigate performance problems (blocking, wait stats, missing indexes, slow queries)
- Inspect schema, objects, and data
- Monitor running queries and active sessions
- Review jobs, backups, and security

Guidelines:
- Always confirm destructive operations (DROP, TRUNCATE, KILL) before executing.
- Prefer READ-ONLY investigation first; ask before making changes.
- When writing T-SQL in execute_query, prefer explicit column lists over SELECT *.
- Summarise findings clearly; include specific values (durations, sizes, counts).
- If a tool returns an error, explain what went wrong and suggest a fix.
"""


def _build_server() -> object:
    tools = [
        execute_query,
        list_databases,
        list_tables,
        get_table_schema,
        get_server_info,
        get_running_queries,
        get_blocking_queries,
        get_index_info,
        get_missing_indexes,
        list_jobs,
        list_logins_and_users,
        get_wait_stats,
        get_backup_history,
        kill_session,
    ]
    return create_sdk_mcp_server("mssql-admin", tools=tools)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

async def run_single(prompt: str) -> str:
    """Run a single prompt and return the result text."""
    server = _build_server()
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"mssql": server},
        max_turns=30,
    )
    result_text = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
                        result_text += block.text
            elif isinstance(message, ResultMessage):
                result_text = message.result
    return result_text


async def interactive_session():
    """REPL loop for multi-turn conversation with the agent."""
    server = _build_server()
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"mssql": server},
        max_turns=30,
    )

    print("MSSQL Admin Agent  (type 'quit' to exit)\n")

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="", flush=True)
                elif isinstance(message, SystemMessage) and message.subtype == "init":
                    pass  # session established
            print("\n")


async def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        await run_single(prompt)
        print()
    else:
        await interactive_session()


if __name__ == "__main__":
    anyio.run(main)
