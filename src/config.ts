import { resolve } from "node:path";

export interface AppConfig {
  port: number;
  dataDir: string;
  dbPath: string;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const port = Number(env.PORT ?? 3000);
  const dataDir = resolve(
    process.cwd(),
    env.DATA_DIR ?? "data",
  );
  return {
    port: Number.isInteger(port) && port > 0 ? port : 3000,
    dataDir,
    dbPath: resolve(dataDir, "autodiff.db"),
  };
}
