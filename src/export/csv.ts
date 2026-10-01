import { CheckReport, GradientReport } from "../core/types.js";

function cell(v: unknown): string {
  const s = v === null || v === undefined ? "" : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function row(values: unknown[]): string {
  return values.map(cell).join(",");
}

// 将多种报告扁平化为 CSV 行集合。
export function gradientToCsv(rep: GradientReport): string {
  const lines = ["output,input,gradient"];
  for (const [out, grads] of Object.entries(rep.jacobian)) {
    for (const [inp, val] of Object.entries(grads)) {
      lines.push(row([out, inp, val]));
    }
  }
  return lines.join("\n");
}

export function valuesToCsv(values: Record<string, number>): string {
  const lines = ["name,value"];
  for (const [name, val] of Object.entries(values)) {
    lines.push(row([name, val]));
  }
  return lines.join("\n");
}

export function checkToCsv(rep: CheckReport): string {
  const lines = [
    "input,output,h,analytic,numeric,absError,relError,ok,reason",
  ];
  for (const e of rep.entries) {
    lines.push(
      row([
        e.input,
        e.output,
        e.h,
        e.analytic,
        e.numeric,
        e.absError,
        e.relError,
        e.ok,
        e.reason ?? "",
      ]),
    );
  }
  return lines.join("\n");
}
