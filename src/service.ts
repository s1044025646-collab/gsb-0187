import { Store } from "./db/store.js";
import {
  directional,
  evaluate,
  reverseJacobian,
  vectorJacobianProduct,
} from "./core/engine.js";
import { finiteCheck, CheckOptions } from "./core/finiteDiff.js";
import { validateGraph } from "./core/validate.js";
import { GraphDef, ValueMap } from "./core/types.js";

export interface ResolvedGraph {
  graphId: number;
  version: number;
  def: GraphDef;
}

export class Service {
  constructor(private readonly store: Store) {}

  getStore(): Store {
    return this.store;
  }

  importGraph(def: GraphDef): {
    graphId: number;
    version: number;
    reused: boolean;
    validation: ReturnType<typeof validateGraph>["order"];
  } {
    // 先校验，再落库；非法图不会产生版本。
    const vg = validateGraph(def);
    const saved = this.store.upsertGraph(def.name, def);
    return {
      graphId: saved.id,
      version: saved.version,
      reused: saved.reused,
      validation: vg.order,
    };
  }

  resolve(name: string, version?: number): ResolvedGraph {
    const row = this.store.resolveGraph(name, version);
    if (!row) throw new Error(`图不存在: ${name}${version ? `@${version}` : ""}`);
    const def = JSON.parse(row.definition) as GraphDef;
    return { graphId: row.id, version: row.version, def };
  }

  private prepare(name: string, version: number | undefined) {
    const resolved = this.resolve(name, version);
    const vg = validateGraph(resolved.def);
    return { resolved, vg };
  }

  validateOnly(name: string, version?: number) {
    const { resolved, vg } = this.prepare(name, version);
    return {
      graphId: resolved.graphId,
      version: resolved.version,
      order: vg.order,
      inputs: vg.def.inputs,
      outputs: vg.def.outputs,
    };
  }

  saveInputSet(
    name: string,
    version: number | undefined,
    label: string,
    inputs: ValueMap,
  ): number {
    const { resolved } = this.prepare(name, version);
    return this.store.addInputSet(resolved.graphId, label, inputs);
  }

  eval_(name: string, inputs: ValueMap, version?: number, save = true) {
    const { resolved, vg } = this.prepare(name, version);
    const result = evaluate(vg, inputs);
    let inputSetId: number | null = null;
    let reportId: number | null = null;
    if (save) {
      inputSetId = this.store.addInputSet(
        resolved.graphId,
        "eval",
        inputs,
      );
      reportId = this.store.addReport(resolved.graphId, "eval", result, inputSetId);
    }
    return { ...result, graphId: resolved.graphId, version: resolved.version, reportId };
  }

  gradient(name: string, inputs: ValueMap, version?: number, save = true) {
    const { resolved, vg } = this.prepare(name, version);
    const report = reverseJacobian(vg, inputs);
    let reportId: number | null = null;
    let inputSetId: number | null = null;
    if (save) {
      inputSetId = this.store.addInputSet(resolved.graphId, "gradient", inputs);
      reportId = this.store.addReport(
        resolved.graphId,
        "gradient",
        report,
        inputSetId,
      );
    }
    return { ...report, graphId: resolved.graphId, version: resolved.version, reportId };
  }

  directional_(
    name: string,
    inputs: ValueMap,
    direction: ValueMap,
    version?: number,
    save = true,
  ) {
    const { resolved, vg } = this.prepare(name, version);
    const report = directional(vg, inputs, direction);
    let reportId: number | null = null;
    if (save) {
      const inputSetId = this.store.addInputSet(
        resolved.graphId,
        "directional",
        { inputs, direction },
      );
      reportId = this.store.addReport(
        resolved.graphId,
        "directional",
        report,
        inputSetId,
      );
    }
    return { ...report, graphId: resolved.graphId, version: resolved.version, reportId };
  }

  vjp(
    name: string,
    inputs: ValueMap,
    weights: ValueMap,
    version?: number,
    save = true,
  ) {
    const { resolved, vg } = this.prepare(name, version);
    const result = vectorJacobianProduct(vg, inputs, weights);
    let reportId: number | null = null;
    if (save) {
      const inputSetId = this.store.addInputSet(
        resolved.graphId,
        "vjp",
        { inputs, weights },
      );
      reportId = this.store.addReport(resolved.graphId, "vjp", result, inputSetId);
    }
    return { ...result, graphId: resolved.graphId, version: resolved.version, reportId };
  }

  check(
    name: string,
    inputs: ValueMap,
    options: CheckOptions,
    version?: number,
    save = true,
  ) {
    const { resolved, vg } = this.prepare(name, version);
    const report = finiteCheck(vg, inputs, options);
    let reportId: number | null = null;
    if (save) {
      const inputSetId = this.store.addInputSet(
        resolved.graphId,
        "check",
        { inputs, options },
      );
      reportId = this.store.addReport(
        resolved.graphId,
        "check",
        report,
        inputSetId,
      );
    }
    return { ...report, graphId: resolved.graphId, version: resolved.version, reportId };
  }
}
