import { CompiledGraph } from "./graph.js";
import { forward, outputValues, reverseGradient, toDiagnostics } from "./evaluate.js";
import { Diagnostic } from "./types.js";

export interface FdPerInput {
  analytic: number;
  numeric: number | null;
  absError: number | null;
  relError: number | null;
  step: number;
  failure: string | null;
}

export interface FdReport {
  ok: boolean;
  output: string;
  diagnostics: Diagnostic[];
  results: Record<string, FdPerInput>;
}

/**
 * 中心有限差分校验（独立验证，非反向模式）。
 * 步长策略：h = baseStep * max(1, |x|)，支持按输入覆盖（steps 参数）。
 */
export function finiteDifferenceCheck(
  g: CompiledGraph,
  point: Record<string, number>,
  output: string,
  baseStep = 1e-5,
  steps?: Record<string, number>
): FdReport {
  if (!(output in g.outputs)) throw new Error(`未知输出 "${output}"`);
  const weights: Record<string, number> = {};
  for (const o of Object.keys(g.outputs)) weights[o] = o === output ? 1 : 0;
  let analytic: Record<string, number>;
  try {
    analytic = reverseGradient(g, point, weights).gradient;
  } catch (e) {
    return { ok: false, output, diagnostics: toDiagnostics(e), results: {} };
  }
  const results: Record<string, FdPerInput> = {};
  let ok = true;
  for (const name of g.inputs) {
    const x = point[name];
    const h = steps?.[name] ?? baseStep * Math.max(1, Math.abs(x));
    const plus = { ...point, [name]: x + h };
    const minus = { ...point, [name]: x - h };
    let fPlus: number, fMinus: number;
    try {
      fPlus = outputValues(g, forward(g, plus))[output];
    } catch (e) {
      results[name] = { analytic: analytic[name], numeric: null, absError: null, relError: null, step: h, failure: `正向扰动求值失败: ${toDiagnostics(e)[0].message}` };
      ok = false;
      continue;
    }
    try {
      fMinus = outputValues(g, forward(g, minus))[output];
    } catch (e) {
      results[name] = { analytic: analytic[name], numeric: null, absError: null, relError: null, step: h, failure: `负向扰动求值失败: ${toDiagnostics(e)[0].message}` };
      ok = false;
      continue;
    }
    const numeric = (fPlus - fMinus) / (2 * h);
    const absError = Math.abs(numeric - analytic[name]);
    const relError = absError / Math.max(1e-12, Math.abs(analytic[name]));
    results[name] = { analytic: analytic[name], numeric, absError, relError, step: h, failure: null };
  }
  return { ok, output, diagnostics: [], results };
}
