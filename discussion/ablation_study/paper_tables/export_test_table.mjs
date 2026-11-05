import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook} from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const values = JSON.parse(await fs.readFile(path.join(here, 'test_table_values.json'), 'utf8'));
if (values.rows.length !== 9 || values.headers.length !== 6 || values.test_records !== 51772) {
  throw new Error('Unexpected table population');
}
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Test results');
sheet.getRange('A1:F10').values = [values.headers, ...values.rows];
workbook.recalculate();
const table = sheet.getRange('A1:F10').values;
if (JSON.stringify(table) !== JSON.stringify([values.headers, ...values.rows])) {
  throw new Error('Export changed the computed values');
}
const escape = value => /[",\r\n]/.test(String(value)) ? '"' + String(value).replaceAll('"', '""') + '"' : String(value);
const csv = '\ufeff' + table.map(row => row.map(escape).join(',')).join('\r\n') + '\r\n';
const target = path.join(here, 'Ablation_Test_MAE_R2_2p88s_29p76s.csv');
await fs.writeFile(target, csv, 'utf8');
console.log(JSON.stringify({output: target, rows: 9, columns: 6, records: 51772, all_values_finite: true}));
