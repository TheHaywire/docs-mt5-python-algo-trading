# Data Pipeline

## 1. Data Ingestion Flow

```mermaid
flowchart TD
    A[MT5 Live Feed] --> B[Market Data Adapter]
    C[Historical Data Source] --> D[Historical Data Loader]
    B --> E[Data Normalizer]
    D --> E
    E --> F[Event Sourcing Raw Store]
    E --> G[Feature Engineering]
    G --> H[Feature Store]
    F --> I[Data Quality Monitor]
    I --> J[Alerts Logs]
```

---

## 2. Event Sourcing & Storage
- All raw and processed data events are stored for replay and audit.
- Data is versioned for reproducibility.
- Storage options: in-memory (for speed), persistent (CSV, Parquet, SQL, HDF5).

---

## 3. Data Quality Monitoring Sequence

```mermaid
sequenceDiagram
    participant DataNormalizer
    participant QualityMonitor
    participant Logger
    participant Alert

    DataNormalizer->>QualityMonitor: New data batch
    QualityMonitor->>QualityMonitor: Check for gaps, outliers, anomalies
    alt Issue detected
        QualityMonitor->>Alert: Trigger alert
        QualityMonitor->>Logger: Log issue
    else No issue
        QualityMonitor->>Logger: Log success
    end
```

---

## 4. Advanced Notes
- Adapters abstract all data sources (MT5, REST, FIX, CSV, etc.).
- Normalization ensures all data is in a canonical format (timestamp, symbol, bid/ask, volume, etc.).
- Feature engineering is modular and supports ML workflows.
- Automated data quality checks with alerting for missing or anomalous data.

---

> **TODO:** Add code/config references for adapters, normalization, and quality checks.
