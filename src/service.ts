import { compileGraph } from "./graph.js";
import { forward, outputValues, reverseGradient, jacobian, directionalDerivative, toDiagnostics } from "./evaluate.js";
import { finiteDifferenceCheck } from "./fdcheck.js";
import { Store } from "./store.js";
import { Diagnostic, GraphDef } from "./types.js";

export class Service {
  constructor(public store: Store) {}

  importGraph(def: GraphDef) {
    const compiled = compileGraph(def); // 先验证再入库
    void compiled;
    return this.store.importGraph(def);
  }

  validate(def: GraphDef): { ok: boolean; error?: string } {
    try {
      compileGraph(def);
      return { ok: true };
    } catch (e) {
      return { ok: false, error: String(e instanceof Error ? e.message : e) };
    }
  }

  private compiled(name: string, version?: number) {
    const row = this.store.getGraph(name, version);
    if (!row) throw new Error(`图不存在: ${name}${version !== undefined ? "@" + version : ""}`);
    return { row, g: compileGraph(row.def) };
  }

  evaluate(name: string, point: Record<string, number>, version?: number, save = true) {
    const { row, g } = this.compiled(name, version);
    try {
      const values = outputValues(g, forward(g, point));
      if (save) this.persist(row.id, point, "evaluate", { values }, []);
      return { ok: true, values, diagnostics: [] as Diagnostic[] };
    } catch (e) {
      const diagnostics = toDiagnostics(e);
      if (save) this.persist(row.id, point, "evaluate", {}, diagnostics);
      return { ok: false, values: null, diagnostics };
    }
  }

  gradient(name: string, point: Record<string, number>, version?: number, weights?: Record<string, number>, save = true) {
    const { row, g } = this.compiled(name, version);
    try {
      const r = reverseGradient(g, point, weights);
      if (save) this.persist(row.id, point, "gradient", r, []);
      return { ok: true, ...r, diagnostics: [] as Diagnostic[] };
    } catch (e) {
      const diagnostics = toDiagnostics(e);
      if (save) this.persist(row.id, point, "gradient", {}, diagnostics);
      return { ok: false, values: null, gradient: null, diagnostics };
    }
  }

  jacobian(name: string, point: Record<string, number>, version?: number, save = true) {
    const { row, g } = this.compiled(name, version);
    try {
      const r = jacobian(g, point);
      if (save) this.persist(row.id, point, "jacobian", r, []);
      return { ok: true, ...r, diagnostics: [] as Diagnostic[] };
    } catch (e) {
      const diagnostics = toDiagnostics(e);
      if (save) this.persist(row.id, point, "jacobian", {}, diagnostics);
      return { ok: false, values: null, jacobian: null, diagnostics };
    }
  }

  directional(name: string, point: Record<string, number>, direction: Record<string, number>, version?: number, save = true) {
    const { row, g } = this.compiled(name, version);
    try {
      const r = directionalDerivative(g, point, direction);
      if (save) this.persist(row.id, point, "directional", { ...r, direction }, []);
      return { ok: true, ...r, diagnostics: [] as Diagnostic[] };
    } catch (e) {
      const diagnostics = toDiagnostics(e);
      if (save) this.persist(row.id, point, "directional", { direction }, diagnostics);
      return { ok: false, values: null, tangents: null, diagnostics };
    }
  }

  check(name: string, point: Record<string, number>, output: string, version?: number, step?: number, steps?: Record<string, number>, save = true) {
    const { row, g } = this.compiled(name, version);
    const report = finiteDifferenceCheck(g, point, output, step, steps);
    if (save) this.persist(row.id, point, "fdcheck", report, report.diagnostics);
    return report;
  }

  private persist(graphId: number, point: Record<string, number>, kind: string, payload: unknown, diagnostics: unknown[]) {
    const inputSetId = this.store.saveInputSet(graphId, point);
    this.store.saveResult(graphId, inputSetId, kind, payload, diagnostics);
  }
}
