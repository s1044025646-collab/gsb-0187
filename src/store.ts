import { DatabaseSync } from "node:sqlite";
import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { GraphDef } from "./types.js";

export interface GraphRow { id: number; name: string; version: number; def: GraphDef; createdAt: string; }

export class Store {
  private db: DatabaseSync;

  constructor(dbPath: string) {
    mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS graphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        hash TEXT NOT NULL,
        def TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(name, version)
      );
      CREATE TABLE IF NOT EXISTS input_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graph_id INTEGER NOT NULL REFERENCES graphs(id),
        label TEXT,
        point TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        graph_id INTEGER NOT NULL REFERENCES graphs(id),
        input_set_id INTEGER NOT NULL REFERENCES input_sets(id),
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        diagnostics TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
      );
    `);
  }

  close(): void { this.db.close(); }

  /** 导入图：同名同内容返回已有版本，同名不同内容创建新版本。 */
  importGraph(def: GraphDef): GraphRow {
    const name = def.name ?? "graph";
    const hash = createHash("sha256").update(JSON.stringify(canonical(def))).digest("hex");
    const existing = this.db
      .prepare("SELECT * FROM graphs WHERE name = ? AND hash = ? ORDER BY version DESC LIMIT 1")
      .get(name, hash) as any;
    if (existing) return rowToGraph(existing);
    const latest = this.db
      .prepare("SELECT * FROM graphs WHERE name = ? ORDER BY version DESC LIMIT 1")
      .get(name) as any;
    const version = latest ? latest.version + 1 : 1;
    const now = new Date().toISOString();
    const info = this.db
      .prepare("INSERT INTO graphs (name, version, hash, def, created_at) VALUES (?, ?, ?, ?, ?)")
      .run(name, version, hash, JSON.stringify(def), now);
    return { id: Number(info.lastInsertRowid), name, version, def, createdAt: now };
  }

  getGraph(name: string, version?: number): GraphRow | null {
    const row = version === undefined
      ? this.db.prepare("SELECT * FROM graphs WHERE name = ? ORDER BY version DESC LIMIT 1").get(name) as any
      : this.db.prepare("SELECT * FROM graphs WHERE name = ? AND version = ?").get(name, version) as any;
    return row ? rowToGraph(row) : null;
  }

  listGraphs(): GraphRow[] {
    const rows = this.db.prepare("SELECT * FROM graphs ORDER BY name, version").all() as any[];
    return rows.map(rowToGraph);
  }

  saveInputSet(graphId: number, point: Record<string, number>, label?: string): number {
    const info = this.db
      .prepare("INSERT INTO input_sets (graph_id, label, point, created_at) VALUES (?, ?, ?, ?)")
      .run(graphId, label ?? null, JSON.stringify(point), new Date().toISOString());
    return Number(info.lastInsertRowid);
  }

  saveResult(graphId: number, inputSetId: number, kind: string, payload: unknown, diagnostics: unknown[]): number {
    const info = this.db
      .prepare("INSERT INTO results (graph_id, input_set_id, kind, payload, diagnostics, created_at) VALUES (?, ?, ?, ?, ?, ?)")
      .run(graphId, inputSetId, kind, JSON.stringify(payload), JSON.stringify(diagnostics), new Date().toISOString());
    return Number(info.lastInsertRowid);
  }

  listResults(graphId?: number): any[] {
    const rows = graphId === undefined
      ? this.db.prepare("SELECT * FROM results ORDER BY id").all() as any[]
      : this.db.prepare("SELECT * FROM results WHERE graph_id = ? ORDER BY id").all(graphId) as any[];
    return rows.map((r) => ({ ...r, payload: JSON.parse(r.payload), diagnostics: JSON.parse(r.diagnostics) }));
  }

  getInputSet(id: number): { id: number; graph_id: number; label: string | null; point: Record<string, number> } | null {
    const r = this.db.prepare("SELECT * FROM input_sets WHERE id = ?").get(id) as any;
    return r ? { ...r, point: JSON.parse(r.point) } : null;
  }
}

function rowToGraph(r: any): GraphRow {
  return { id: r.id, name: r.name, version: r.version, def: JSON.parse(r.def), createdAt: r.created_at };
}

function canonical(def: GraphDef): unknown {
  return {
    inputs: def.inputs,
    constants: def.constants ?? {},
    nodes: def.nodes,
    outputs: def.outputs,
  };
}
