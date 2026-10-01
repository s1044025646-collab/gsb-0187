// SQLite 持久化：图版本、输入集、求值结果、梯度/校验报告。
// 修改图即创建新版本，旧版本结果完整保留。

import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

export interface GraphVersionRow {
  id: number;
  graph_name: string;
  version: number;
  definition: string;
  created_at: string;
}

export interface InputSetRow {
  id: number;
  graph_id: number;
  label: string;
  payload: string;
  created_at: string;
}

export interface ReportRow {
  id: number;
  graph_id: number;
  input_set_id: number | null;
  kind: string;
  payload: string;
  created_at: string;
}

export class Store {
  readonly db: DatabaseSync;

  constructor(dbPath: string) {
    mkdirSync(dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath);
    this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec("PRAGMA foreign_keys = ON;");
    this.migrate();
  }

  close(): void {
    this.db.close();
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS graph_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graph_name TEXT NOT NULL,
        version INTEGER NOT NULL,
        definition TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(graph_name, version)
      );
      CREATE TABLE IF NOT EXISTS input_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graph_id INTEGER NOT NULL REFERENCES graph_versions(id) ON DELETE CASCADE,
        label TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graph_id INTEGER NOT NULL REFERENCES graph_versions(id) ON DELETE CASCADE,
        input_set_id INTEGER REFERENCES input_sets(id) ON DELETE SET NULL,
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
    `);
  }

  latestVersionNumber(name: string): number {
    const row = this.db
      .prepare(
        "SELECT MAX(version) AS v FROM graph_versions WHERE graph_name = ?",
      )
      .get(name) as { v: number | null } | undefined;
    return row?.v ?? 0;
  }

  // 插入新版本；定义完全相同则复用（返回既有行）。
  upsertGraph(
    name: string,
    definition: unknown,
  ): { id: number; version: number; reused: boolean } {
    const json = JSON.stringify(definition);
    const existing = this.db
      .prepare(
        "SELECT id, version FROM graph_versions WHERE graph_name = ? AND definition = ? ORDER BY version DESC LIMIT 1",
      )
      .get(name, json) as { id: number; version: number } | undefined;
    if (existing) {
      return { id: existing.id, version: existing.version, reused: true };
    }
    const next = this.latestVersionNumber(name) + 1;
    const info = this.db
      .prepare(
        "INSERT INTO graph_versions (graph_name, version, definition) VALUES (?, ?, ?)",
      )
      .run(name, next, json);
    return {
      id: Number(info.lastInsertRowid),
      version: next,
      reused: false,
    };
  }

  getGraph(id: number): GraphVersionRow | undefined {
    return this.db
      .prepare("SELECT * FROM graph_versions WHERE id = ?")
      .get(id) as GraphVersionRow | undefined;
  }

  resolveGraph(
    name: string,
    version?: number,
  ): GraphVersionRow | undefined {
    if (version === undefined) {
      return this.db
        .prepare(
          "SELECT * FROM graph_versions WHERE graph_name = ? ORDER BY version DESC LIMIT 1",
        )
        .get(name) as GraphVersionRow | undefined;
    }
    return this.db
      .prepare(
        "SELECT * FROM graph_versions WHERE graph_name = ? AND version = ?",
      )
      .get(name, version) as GraphVersionRow | undefined;
  }

  listGraphs(): Array<{ name: string; latest: number }> {
    return this.db
      .prepare(
        "SELECT graph_name AS name, MAX(version) AS latest FROM graph_versions GROUP BY graph_name ORDER BY graph_name",
      )
      .all() as Array<{ name: string; latest: number }>;
  }

  listVersions(name: string): GraphVersionRow[] {
    return this.db
      .prepare(
        "SELECT * FROM graph_versions WHERE graph_name = ? ORDER BY version",
      )
      .all(name) as unknown as [];
  }

  addInputSet(graphId: number, label: string, payload: unknown): number {
    const info = this.db
      .prepare(
        "INSERT INTO input_sets (graph_id, label, payload) VALUES (?, ?, ?)",
      )
      .run(graphId, label, JSON.stringify(payload));
    return Number(info.lastInsertRowid);
  }

  getInputSet(id: number): InputSetRow | undefined {
    return this.db
      .prepare("SELECT * FROM input_sets WHERE id = ?")
      .get(id) as InputSetRow | undefined;
  }

  listInputSets(graphId: number): InputSetRow[] {
    return this.db
      .prepare(
        "SELECT * FROM input_sets WHERE graph_id = ? ORDER BY id",
      )
      .all(graphId) as unknown as [];
  }

  addReport(
    graphId: number,
    kind: string,
    payload: unknown,
    inputSetId: number | null = null,
  ): number {
    const info = this.db
      .prepare(
        "INSERT INTO reports (graph_id, input_set_id, kind, payload) VALUES (?, ?, ?, ?)",
      )
      .run(graphId, inputSetId, kind, JSON.stringify(payload));
    return Number(info.lastInsertRowid);
  }

  listReports(graphId: number): ReportRow[] {
    return this.db
      .prepare("SELECT * FROM reports WHERE graph_id = ? ORDER BY id")
      .all(graphId) as unknown as [];
  }

  getReport(id: number): ReportRow | undefined {
    return this.db
      .prepare("SELECT * FROM reports WHERE id = ?")
      .get(id) as ReportRow | undefined;
  }
}
