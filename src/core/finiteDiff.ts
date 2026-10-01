// 中心有限差分：独立于反向模式的数值梯度校验。
// (f(x+h) - f(x-h)) / (2h)

import { CheckEntry, CheckReport, GraphError, ValueMap } from "./types.js";
import { ValidatedGraph } from "./validate.js";
import { evaluate, reverseJacobian } from "./engine.js";

export interface CheckOptions {
  // 基础步长；最终按输入尺度缩放。
  baseStep?: number;
  // 为不同尺度输入单独指定步长。
  perInputStep?: Record<string, number>;
  // 绝对 / 相对容差。
  absTol?: number;
  relTol?: number;
}

// 按输入尺度选择步长，缓解尺度差异：h_i = max(baseStep, baseStep*|x_i|)。
export function stepForInput(
  name: string,
  value: number,
  baseStep: number,
  perInput?: Record<string, number>,
): number {
  const custom = perInput?.[name];
  if (typeof custom === "number" && custom > 0) return custom;
  return Math.max(baseStep, baseStep * Math.abs(value));
}

export function finiteCheck(
  vg: ValidatedGraph,
  inputs: ValueMap,
  opts: CheckOptions = {},
): CheckReport {
  const baseStep = opts.baseStep ?? 1e-5;
  const absTol = opts.absTol ?? 1e-6;
  const relTol = opts.relTol ?? 1e-4;
  if (!(baseStep > 0)) {
    throw new GraphError("GRAPH_INVALID", "baseStep 必须是正数", { baseStep });
  }

  // 解析梯度来自反向模式。
  const analytic = reverseJacobian(vg, inputs).jacobian;

  const entries: CheckEntry[] = [];
  const usedSteps: Record<string, number> = {};
  let allOk = true;

  for (const inputName of vg.def.inputs) {
    const base = inputs[inputName]!;
    const h = stepForInput(
      inputName,
      base,
      baseStep,
      opts.perInputStep,
    );
    usedSteps[inputName] = h;

    const plus: ValueMap = { ...inputs, [inputName]: base + h };
    const minus: ValueMap = { ...inputs, [inputName]: base - h };

    for (const out of vg.def.outputs) {
      const a = analytic[out]![inputName]!;
      let numeric: number | null = null;
      let absError: number | null = null;
      let relError: number | null = null;
      let ok = false;
      let reason: string | undefined;

      try {
        const vp = evaluate(vg, plus);
        const vm = evaluate(vg, minus);
        const fp = vp.outputs[out]!;
        const fm = vm.outputs[out]!;
        numeric = (fp - fm) / (2 * h);
        if (!Number.isFinite(numeric)) {
          reason = "有限差分结果非有限（可能步长过大或函数溢出）";
        } else {
          absError = Math.abs(numeric - a);
          const denom = Math.max(1, Math.abs(a));
          relError = absError / denom;
          ok = absError <= absTol || relError <= relTol;
          if (!ok) {
            reason = `误差超过容差（absTol=${absTol}, relTol=${relTol}）`;
          }
        }
      } catch (err) {
        if (err instanceof GraphError) {
          reason = `差分求值失败: ${err.code} - ${err.message}`;
        } else {
          reason = `差分求值失败: ${String(err)}`;
        }
      }

      if (!ok) allOk = false;
      entries.push({
        input: inputName,
        output: out,
        analytic: a,
        numeric,
        absError,
        relError,
        h,
        ok,
        reason,
      });
    }
  }

  return {
    baseStep,
    perInputStep: usedSteps,
    entries,
    allOk,
  };
}
