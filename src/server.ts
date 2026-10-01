import http from "node:http";
import path from "node:path";
import { Store } from "./store.js";
import { Service } from "./service.js";
import { examples } from "./examples.js";

const PORT = Number(process.env.PORT ?? 8077);
const DATA_DIR = process.env.DATA_DIR ?? path.join(process.cwd(), "data");
const service = new Service(new Store(path.join(DATA_DIR, "autodiff.db")));

async function readBody(req: http.IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function send(res: http.ServerResponse, status: number, body: unknown) {
  const s = JSON.stringify(body, null, 2);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(s);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const p = url.pathname;
  try {
    if (req.method === "GET" && p === "/health") return send(res, 200, { ok: true });
    if (req.method === "GET" && p === "/graphs") return send(res, 200, service.store.listGraphs());
    if (req.method === "GET" && p === "/examples") return send(res, 200, examples);
    if (req.method === "POST" && p === "/graphs/import") {
      const body = await readBody(req);
      return send(res, 200, service.importGraph(body));
    }
    if (req.method === "POST" && p === "/validate") {
      const body = await readBody(req);
      return send(res, 200, service.validate(body));
    }
    if (req.method === "GET" && p.startsWith("/graphs/")) {
      const [, , name, sub] = p.split("/");
      const version = url.searchParams.get("version");
      const v = version !== null ? Number(version) : undefined;
      if (!sub) {
        const row = service.store.getGraph(name, v);
        return row ? send(res, 200, row) : send(res, 404, { error: "图不存在" });
      }
      if (sub === "results") {
        const row = service.store.getGraph(name, v);
        if (!row) return send(res, 404, { error: "图不存在" });
        return send(res, 200, service.store.listResults(row.id));
      }
    }
    if (req.method === "POST" && p.startsWith("/graphs/")) {
      const [, , name, action] = p.split("/");
      const body = await readBody(req);
      const v = body.version !== undefined ? Number(body.version) : undefined;
      switch (action) {
        case "evaluate": return send(res, 200, service.evaluate(name, body.point, v));
        case "gradient": return send(res, 200, service.gradient(name, body.point, v, body.weights));
        case "jacobian": return send(res, 200, service.jacobian(name, body.point, v));
        case "directional": return send(res, 200, service.directional(name, body.point, body.direction, v));
        case "check": return send(res, 200, service.check(name, body.point, body.output, v, body.step, body.steps));
        default: return send(res, 404, { error: `未知操作 ${action}` });
      }
    }
    return send(res, 404, { error: "未找到路由" });
  } catch (e) {
    return send(res, 400, { error: String(e instanceof Error ? e.message : e) });
  }
});

server.listen(PORT, () => {
  console.log(`autodiff 服务已启动: http://localhost:${PORT}  数据目录: ${DATA_DIR}`);
});
