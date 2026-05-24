# RAGAS ESG Evaluation Dataset

Generated from the parsed ESG/report chunks supplied in `Arquivo(1).zip`.

Sampling: seeded random stratified sample of 30 deduplicated documents. The sampler balances report styles and caps repeated companies where possible. Seed: 7319.

Rows: 108 QA examples. Answers are exact source spans copied from one cited chunk; `contexts` contains the supporting excerpt used for RAGAS context-based metrics.

Files:
- `ragas_esg_eval_dataset.csv`: main dataset.
- `ragas_esg_eval_dataset.jsonl`: same dataset with JSON-native fields.
- `sampled_30_documents.csv/json`: selected documents and style/sector metadata.

RAGAS-ready fields: `question`, `ground_truth_answer`, `contexts`.
Citation fields: `source_file`, `page_start`, `page_end`, `chunk_id`, `section_title`, and full JSON `citation`.

Sampled document style coverage:
{
  "short_doc": 5,
  "integrated_or_annual_report": 6,
  "table_heavy_doc": 5,
  "clean_esg_or_sustainability_report": 7,
  "other_report_style": 2,
  "long_doc": 5
}

Sampled company coverage:
{
  "Airbus": 2,
  "BNP Paribas": 2,
  "Danone": 4,
  "ENGIE": 3,
  "Enel": 4,
  "Hermès": 1,
  "Iberdrola": 1,
  "L'Oréal": 1,
  "Schneider": 2,
  "Siemens": 3,
  "TotalEnergies": 2,
  "Volkswagen": 4,
  "Unknown": 1
}

Sampled sector coverage:
{
  "aerospace": 2,
  "banking_financial_services": 2,
  "food_beverage": 4,
  "utilities_energy": 3,
  "utilities_power": 5,
  "luxury_goods": 1,
  "consumer_beauty": 1,
  "industrial_energy_management": 2,
  "industrial_technology": 3,
  "oil_gas_energy": 2,
  "automotive": 4,
  "unknown": 1
}

Question row style coverage:
{
  "short_doc": 9,
  "integrated_or_annual_report": 24,
  "table_heavy_doc": 20,
  "clean_esg_or_sustainability_report": 28,
  "other_report_style": 7,
  "long_doc": 20
}

Top topic tags:
[
  [
    "environmental",
    56
  ],
  [
    "targets",
    55
  ],
  [
    "metrics_kpi",
    48
  ],
  [
    "emissions",
    46
  ],
  [
    "governance",
    42
  ],
  [
    "climate",
    36
  ],
  [
    "diversity_inclusion",
    35
  ],
  [
    "social",
    19
  ],
  [
    "biodiversity",
    19
  ],
  [
    "employees",
    18
  ],
  [
    "water",
    15
  ],
  [
    "renewables",
    14
  ],
  [
    "supply_chain",
    14
  ],
  [
    "ethics_compliance",
    7
  ],
  [
    "human_rights",
    6
  ],
  [
    "waste_circularity",
    5
  ],
  [
    "health_safety",
    4
  ]
]
