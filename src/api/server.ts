// 纯 Node.js HTTP JSON API（无第三方框架）。

import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { URL } from "node:url";
import { Service } from "../service.js";
import { GraphError } from "../core/types.js";
import { checkToCsv, gradientToCsv, valuesToCsv } from "../export/csv.js";

function send(res: ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body, null, 2);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(json);
}

function sendCsv(res: ServerResponse, csv: string): void {
  res.writeHead(200, {
    "content-type": "text/csv; charset=utf-8",
  });
  res.end(csv);
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString("utf-8");
  if (raw.length === 0) return {};
  return JSON.parse(raw) as unknown;
}

function errorResponse(res: ServerResponse, err: unknown): void {
  if (err instanceof GraphError) {
    send(res, 422, {
      error: {
        code: err.code,
        message: err.message,
        detail: err.detail ?? null,
      },
    });
    return;
  }
  if (err instanceof SyntaxError) {
    send(res, 400, { error: { code: "BAD_JSON", message: err.message } });
    return;
  }
  const message = err instanceof Error ? err.message : String(err);
  send(res, 500, { error: { code: "INTERNAL", message } });
}

export function createApp(service: Service) {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const path = url.pathname.replace(/\/+$/, "") || "/";
      const method = req.method ?? "GET";
      const fmt = url.searchParams.get("format");

      if (method === "GET" && path === "/health") {
        send(res, 200, { ok: true });
        return;
      }

      if (method === "GET" && path === "/graphs") {
        send(res, 200, { graphs: service.getStore().listGraphs() });
        return;
      }

      const versionsMatch = path.match(/^\/graphs\/([^/]+)\/versions$/);
      if (method === "GET" && versionsMatch) {
        const name = decodeURIComponent(versionsMatch[1]!);
        const rows = service.getStore().listVersions(name).map((r) => ({
          id: r.id,
          version: r.version,
          createdAt: r.created_at,
        }));
        send(res, 200, { name, versions: rows });
        return;
      }

      if (method === "POST" && path === "/graphs") {
        const body = (await readJson(req)) as { graph?: never } & Record<
          string,
          unknown
        >;
        const result = service.importGraph(body as never);
        send(res, result.reused ? 200 : 201, result);
        return;
      }

      const graphMatch = path.match(/^\/graphs\/([^/]+)\/(validate|eval|gradient|directional|vjp|check)$/);
      if (method === "POST" && graphMatch) {
        const name = decodeURIComponent(graphMatch[1]!);
        const action = graphMatch[2]!;
        const body = (await readJson(req)) as Record<string, unknown>;
        const version =
          typeof body.version === "number" ? body.version : undefined;
        const inputs = (body.inputs ?? {}) as Record<string, number>;

        switch (action) {
          case "validate":
            send(res, 200, service.validateOnly(name, version));
            return;
          case "eval": {
            const out = service.eval_(name, inputs, version);
            if (fmt === "csv") return sendCsv(res, valuesToCsv(out.outputs));
            send(res, 200, out);
            return;
          }
          case "gradient": {
            const out = service.gradient(name, inputs, version);
            if (fmt === "csv") return sendCsv(res, gradientToCsv(out));
            send(res, 200, out);
            return;
          }
          case "directional": {
            const direction = (body.direction ?? {}) as Record<string, number>;
            send(
              res,
              200,
              service.directional_(name, inputs, direction, version),
            );
            return;
          }
          case "vjp": {
            const weights = (body.weights ?? {}) as Record<string, number>;
            send(res, 200, service.vjp(name, inputs, weights, version));
            return;
          }
          case "check": {
            const options = (body.options ?? {}) as Parameters<
              typeof service.check
            >[2];
            const out = service.check(name, inputs, options, version);
            if (fmt === "csv") return sendCsv(res, checkToCsv(out));
            send(res, 200, out);
            return;
          }
        }
      }

      const reportMatch = path.match(/^\/reports\/(\d+)$/);
      if (method === "GET" && reportMatch) {
        const id = Number(reportMatch[1]);
        const row = service.getStore().getReport(id);
        if (!row) return send(res, 404, { error: { code: "NOT_FOUND" } });
        if (fmt === "csv") {
          const payload = JSON.parse(row.payload);
          if (row.kind === "gradient") return sendCsv(res, gradientToCsv(payload));
          if (row.kind === "check") return sendCsv(res, checkToCsv(payload));
          if (row.kind === "eval") return sendCsv(res, valuesToCsv(payload.outputs));
        }
        send(res, 200, {
          id: row.id,
          graphId: row.graph_id,
          kind: row.kind,
          createdAt: row.created_at,
          payload: JSON.parse(row.payload),
        });
        return;
      }

      send(res, 404, { error: { code: "NOT_FOUND", message: `无此路由: ${method} ${path}` } });
    } catch (err) {
      errorResponse(res, err);
    }
  });
}

export function startServer(service: Service, port: number) {
  const server = createApp(service);
  return new Promise<ReturnType<typeof createServer>>((resolve) => {
    server.listen(port, () => {
      console.log(`计算图微分服务已启动: http://localhost:${port}`);
      resolve(server);
    });
  });
}

