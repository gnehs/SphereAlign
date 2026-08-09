import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

const VERSION_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const PACKAGE_NAME = "gs360studio";

const files = {
  packageJson: new URL("../package.json", import.meta.url),
  tauriConfig: new URL("../src-tauri/tauri.conf.json", import.meta.url),
  cargoManifest: new URL("../src-tauri/Cargo.toml", import.meta.url),
  cargoLock: new URL("../src-tauri/Cargo.lock", import.meta.url),
};

function usage() {
  console.error(
    "Usage: pnpm version:set <x.y.z> | pnpm version:check <x.y.z>",
  );
}

function replacePackageVersion(toml, version, fileName) {
  const packageHeader = "[package]";
  const packageStart = toml.indexOf(packageHeader);
  const packageEnd = toml.indexOf("\n[", packageStart + packageHeader.length);

  if (packageStart === -1) {
    throw new Error(`${fileName} does not contain a [package] table`);
  }

  const end = packageEnd === -1 ? toml.length : packageEnd;
  const packageTable = toml.slice(packageStart, end);
  const versionPattern = /^(version\s*=\s*)"[^"]+"/m;

  if (!versionPattern.test(packageTable)) {
    throw new Error(`${fileName} does not contain a package version`);
  }

  return `${toml.slice(0, packageStart)}${packageTable.replace(
    versionPattern,
    `$1"${version}"`,
  )}${toml.slice(end)}`;
}

function readPackageVersion(toml, fileName) {
  const packageHeader = "[package]";
  const packageStart = toml.indexOf(packageHeader);

  if (packageStart === -1) {
    throw new Error(`${fileName} does not contain a [package] table`);
  }

  const packageEnd = toml.indexOf(
    "\n[",
    packageStart + packageHeader.length,
  );
  const end = packageEnd === -1 ? toml.length : packageEnd;
  const version = toml
    .slice(packageStart, end)
    .match(/^version\s*=\s*"([^"]+)"/m)?.[1];

  if (!version) {
    throw new Error(`${fileName} does not contain a package version`);
  }

  return version;
}

function replaceCargoLockVersion(lockfile, version) {
  const packageBlocks = lockfile.split(/(?=^\[\[package\]\]$)/m);
  let found = false;

  const updatedBlocks = packageBlocks.map((block) => {
    if (!new RegExp(`^name = "${PACKAGE_NAME}"$`, "m").test(block)) {
      return block;
    }

    const versionPattern = /^(version\s*=\s*)"[^"]+"/m;
    if (!versionPattern.test(block)) {
      throw new Error(
        `src-tauri/Cargo.lock does not contain a version for ${PACKAGE_NAME}`,
      );
    }

    found = true;
    return block.replace(versionPattern, `$1"${version}"`);
  });

  if (!found) {
    throw new Error(
      `src-tauri/Cargo.lock does not contain the ${PACKAGE_NAME} package`,
    );
  }

  return updatedBlocks.join("");
}

async function main() {
  const checkOnly = process.argv[2] === "--check";
  const version = process.argv[checkOnly ? 3 : 2];

  if (!version || !VERSION_PATTERN.test(version)) {
    usage();
    process.exitCode = 1;
    return;
  }

  const [packageText, tauriText, cargoText, cargoLockText] =
    await Promise.all([
      readFile(files.packageJson, "utf8"),
      readFile(files.tauriConfig, "utf8"),
      readFile(files.cargoManifest, "utf8"),
      readFile(files.cargoLock, "utf8"),
    ]);

  const packageJson = JSON.parse(packageText);
  const tauriConfig = JSON.parse(tauriText);
  const currentVersions = {
    "package.json": packageJson.version,
    "src-tauri/tauri.conf.json": tauriConfig.version,
    "src-tauri/Cargo.toml": readPackageVersion(
      cargoText,
      "src-tauri/Cargo.toml",
    ),
    "src-tauri/Cargo.lock": cargoLockText
      .split(/(?=^\[\[package\]\]$)/m)
      .find((block) =>
        new RegExp(`^name = "${PACKAGE_NAME}"$`, "m").test(block),
      )
      ?.match(/^version\s*=\s*"([^"]+)"/m)?.[1],
  };

  if (checkOnly) {
    const mismatches = Object.entries(currentVersions).filter(
      ([, currentVersion]) => currentVersion !== version,
    );

    if (mismatches.length > 0) {
      for (const [fileName, currentVersion] of mismatches) {
        console.error(
          `${fileName}: expected ${version}, found ${currentVersion ?? "missing"}`,
        );
      }
      process.exitCode = 1;
      return;
    }

    console.log(`All application versions are ${version}.`);
    return;
  }

  packageJson.version = version;
  tauriConfig.version = version;

  await Promise.all([
    writeFile(
      files.packageJson,
      `${JSON.stringify(packageJson, null, 2)}\n`,
      "utf8",
    ),
    writeFile(
      files.tauriConfig,
      `${JSON.stringify(tauriConfig, null, 2)}\n`,
      "utf8",
    ),
    writeFile(
      files.cargoManifest,
      replacePackageVersion(
        cargoText,
        version,
        "src-tauri/Cargo.toml",
      ),
      "utf8",
    ),
    writeFile(
      files.cargoLock,
      replaceCargoLockVersion(cargoLockText, version),
      "utf8",
    ),
  ]);

  console.log(`Updated application version to ${version}.`);
}

await main();
