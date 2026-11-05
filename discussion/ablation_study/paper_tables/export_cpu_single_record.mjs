import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook} from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const latest = JSON.parse(await fs.readFile(path.join(here, '../cpu_single_record_results/latest_run.json'), 'utf8'));
const result = JSON.parse(await fs.readFile(path.join(latest.result_directory, 'measurement.json'), 'utf8'));
const workbook = Workbook.create();
const tables = [['summary', 'cpu_single_record_summary.csv'], ['updates', 'cpu_single_record_updates.csv'], ['latency_samples', 'cpu_single_record_latency_samples.csv']];
for (const [name] of tables) {
  const rows = result[name];
  const headers = Object.keys(rows[0]);
  const matrix = [headers, ...rows.map(row => headers.map(header => row[header]))];
  const sheet = workbook.worksheets.add(name);
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
}
workbook.recalculate();
const escape = value => /[",\r\n]/.test(String(value)) ? '"' + String(value).replaceAll('"', '""') + '"' : String(value);
for (const [name, filename] of tables) {
  const rows = result[name];
  const headers = Object.keys(rows[0]);
  const actual = workbook.worksheets.getItem(name).getRangeByIndexes(0, 0, rows.length + 1, headers.length).values;
  const expected = [headers, ...rows.map(row => headers.map(header => row[header]))];
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error('Export value mismatch');
  const csv = '\ufeff' + actual.map(row => row.map(escape).join(',')).join('\r\n') + '\r\n';
  await fs.writeFile(path.join(latest.result_directory, filename), csv, 'utf8');
  console.log(JSON.stringify({file: path.join(latest.result_directory, filename), rows: rows.length}));
}
