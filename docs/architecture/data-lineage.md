# 📊 Data Lineage & Flow Architecture

## Overview

This document provides comprehensive data lineage tracking and flow visualization for the gogooku3-standalone system, showing how data moves through the entire ML pipeline from source to prediction.

## Data Sources

```mermaid
graph TD
    A[J-Quants API] --> B[Raw Stock Data]
    C[Company Fundamentals] --> B
    D[Market Indicators] --> B
    E[Economic Data] --> B

    B --> F[Data Ingestion Layer]
    F --> G[Raw Data Storage]
    G --> H[(MinIO S3)]
    G --> I[(Local Filesystem)]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#e8f5e8
    style H fill:#fff3e0
    style I fill:#fff3e0
```

## Data Processing Pipeline

```mermaid
graph TD
    A[Raw Data Sources] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Cleaning]
    D --> E[Feature Engineering]
    E --> F[Data Transformation]
    F --> G[Quality Assurance]

    C --> H[Validation Reports]
    D --> I[Cleaning Logs]
    E --> J[Feature Metadata]
    G --> K[Quality Metrics]

    G --> L[Processed Dataset]
    L --> M[Training Pipeline]
    L --> N[Validation Pipeline]
    L --> O[Inference Pipeline]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style L fill:#fff3e0
```

## Feature Engineering Data Flow

```mermaid
graph TD
    A[Raw OHLCV Data] --> B[Price Features]
    A --> C[Volume Features]
    A --> D[Temporal Features]

    B --> E[Technical Indicators]
    C --> E
    D --> E

    E --> F[Statistical Features]
    E --> G[Cross-sectional Features]
    E --> H[Derived Features]

    F --> I[Feature Selection]
    G --> I
    H --> I

    I --> J[Feature Store]
    J --> K[Model Training]
    J --> L[Feature Validation]

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
    style J fill:#fff3e0
```

## ML Training Pipeline Data Flow

```mermaid
graph TD
    A[Training Dataset] --> B[Data Splitter]
    B --> C[Train Set]
    B --> D[Validation Set]
    B --> E[Test Set]

    C --> F[ATFT-GAT-FAN Model]
    C --> G[LightGBM Baseline]

    F --> H[Model Training]
    G --> H

    H --> I[Hyperparameter Tuning]
    I --> J[Model Selection]

    J --> K[Model Validation]
    K --> L[Performance Metrics]

    L --> M[Model Registry]
    M --> N[Model Deployment]
    M --> O[Model Monitoring]

    style A fill:#e1f5fe
    style H fill:#f3e5f5
    style K fill:#e8f5e8
    style M fill:#fff3e0
```

## Model Inference Data Flow

```mermaid
graph TD
    A[New Market Data] --> B[Data Ingestion]
    B --> C[Feature Engineering]
    C --> D[Feature Validation]

    D --> E[Model Loading]
    E --> F[ATFT-GAT-FAN Inference]
    E --> G[Ensemble Prediction]

    F --> H[Prediction Results]
    G --> H

    H --> I[Confidence Scoring]
    I --> J[Prediction Validation]

    J --> K[Results Storage]
    K --> L[API Response]
    K --> M[Database Storage]
    K --> N[Alert Generation]

    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style I fill:#e8f5e8
    style K fill:#fff3e0
```

## Data Quality Pipeline

```mermaid
graph TD
    A[Raw Data] --> B[Data Quality Checks]
    B --> C[Completeness Check]
    B --> D[Accuracy Check]
    B --> E[Consistency Check]
    B --> F[Timeliness Check]

    C --> G[Quality Metrics]
    D --> G
    E --> G
    F --> G

    G --> H{Quality Threshold}
    H -->|Pass| I[Data Processing]
    H -->|Fail| J[Data Correction]
    H -->|Fail| K[Alert Generation]

    J --> I
    I --> L[Processed Data]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style G fill:#e8f5e8
    style L fill:#fff3e0
```

## Storage Layer Architecture

```mermaid
graph TD
    A[Application Layer] --> B[Storage Abstraction]
    B --> C[Object Storage]
    B --> D[Relational Database]
    B --> E[Cache Layer]
    B --> F[File System]

    C --> G[(MinIO S3)]
    D --> H[(ClickHouse OLAP)]
    E --> I[(Redis Cache)]
    F --> J[(Local Files)]

    G --> K[Data Backup]
    H --> L[Database Backup]
    I --> M[Cache Persistence]
    J --> N[File Backup]

    K --> O[Backup Storage]
    L --> O
    M --> O
    N --> O

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style G fill:#fff3e0
    style O fill:#ffebee
```

## Monitoring & Observability Data Flow

```mermaid
graph TD
    A[System Components] --> B[Metrics Collection]
    B --> C[Health Checks]
    B --> D[Performance Metrics]
    B --> E[Business Metrics]

    C --> F[Metrics Exporter]
    D --> F
    E --> F

    F --> G[Prometheus]
    G --> H[Grafana Dashboards]
    G --> I[Alert Manager]

    I --> J[Email Alerts]
    I --> K[Slack Alerts]
    I --> L[PagerDuty]

    H --> M[Real-time Monitoring]
    M --> N[Performance Analysis]
    M --> O[Capacity Planning]

    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style G fill:#e8f5e8
    style M fill:#fff3e0
```

## Backup & Recovery Data Flow

```mermaid
graph TD
    A[Production Data] --> B[Backup Scheduler]
    B --> C[Data Backup]
    B --> D[Database Backup]
    B --> E[Configuration Backup]

    C --> F[(MinIO Backup)]
    D --> G[(ClickHouse Backup)]
    E --> H[(Config Files Backup)]

    F --> I[Backup Validation]
    G --> I
    H --> I

    I --> J[Backup Storage]
    J --> K[Disaster Recovery]

    K --> L[Data Restore]
    K --> M[Database Restore]
    K --> N[Configuration Restore]

    L --> O[Application Recovery]
    M --> O
    N --> O

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style I fill:#e8f5e8
    style J fill:#fff3e0
```

## Data Lineage Tracking

### End-to-End Data Traceability

```mermaid
timeline
    title Data Lineage Timeline

    section Data Acquisition
        Raw API Data : J-Quants API
                    : Data Ingestion
                    : Initial Validation

    section Data Processing
        Cleaned Data : Data Cleaning
                     : Outlier Removal
                     : Missing Value Handling

    section Feature Engineering
        Feature Set : Technical Indicators
                    : Statistical Features
                    : Cross-sectional Features

    section Model Training
        Training Data : Feature Selection
                      : Data Splitting
                      : Model Training

    section Model Deployment
        Production Model : Model Validation
                         : Performance Testing
                         : Deployment

    section Inference
        Predictions : Real-time Inference
                    : Prediction Storage
                    : Result Analysis
```

## Data Quality Gates

```mermaid
flowchart TD
    A[Data Source] --> B{Quality Gate 1}
    B -->|Pass| C[Data Processing]
    B -->|Fail| D[Quality Alert]

    C --> E{Quality Gate 2}
    E -->|Pass| F[Feature Engineering]
    E -->|Fail| G[Processing Alert]

    F --> H{Quality Gate 3}
    H -->|Pass| I[Model Training]
    H -->|Fail| J[Feature Alert]

    I --> K{Quality Gate 4}
    K -->|Pass| L[Model Deployment]
    K -->|Fail| M[Training Alert]

    L --> N{Quality Gate 5}
    N -->|Pass| O[Production Inference]
    N -->|Fail| P[Deployment Alert]

    D --> Q[Quality Dashboard]
    G --> Q
    J --> Q
    M --> Q
    P --> Q

    style A fill:#e1f5fe
    style Q fill:#ffebee
```

## Performance Monitoring Data Flow

```mermaid
graph TD
    A[Application Metrics] --> B[RED Metrics]
    A --> C[SLA Metrics]
    A --> D[Custom Metrics]

    B --> E[Rate Monitoring]
    B --> F[Error Monitoring]
    B --> G[Duration Monitoring]

    C --> H[SLA Compliance]
    C --> I[Target Achievement]

    D --> J[Business KPIs]
    D --> K[Technical Metrics]

    E --> L[Prometheus]
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M[Grafana]
    L --> N[Alert Rules]

    M --> O[Real-time Dashboards]
    N --> P[Automated Alerts]

    style A fill:#e1f5fe
    style L fill:#f3e5f5
    style M fill:#e8f5e8
    style O fill:#fff3e0
```

## Security Data Flow

```mermaid
graph TD
    A[User Requests] --> B[Authentication]
    B --> C[Authorization]
    C --> D[Input Validation]
    D --> E[Data Processing]

    E --> F[Output Validation]
    F --> G[Audit Logging]

    B --> H[Security Monitoring]
    C --> H
    D --> H

    H --> I[Security Events]
    I --> J[SIEM System]
    I --> K[Security Dashboard]

    G --> L[Audit Storage]
    L --> M[Compliance Reports]

    style A fill:#e1f5fe
    style H fill:#f3e5f5
    style I fill:#ffebee
    style L fill:#fff3e0
```

## Integration Points

### External System Integration

```mermaid
graph TD
    A[gogooku3-standalone] --> B[J-Quants API]
    A --> C[MinIO S3]
    A --> D[ClickHouse]
    A --> E[Redis]
    A --> F[MLflow]

    B --> G[Market Data]
    C --> H[Object Storage]
    D --> I[Analytics Database]
    E --> J[Caching Layer]
    F --> K[Experiment Tracking]

    A --> L[Prometheus]
    A --> M[Grafana]
    A --> N[Alert Manager]

    L --> O[Metrics Collection]
    M --> P[Visualization]
    N --> Q[Alerting]

    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style O fill:#fff3e0
```

## Data Retention Policies

| Data Type | Retention Period | Storage Location | Backup Frequency |
|-----------|-----------------|------------------|------------------|
| Raw Market Data | 7 years | MinIO S3 | Daily |
| Processed Features | 2 years | MinIO S3 | Daily |
| Model Artifacts | 1 year | MLflow Registry | Weekly |
| Training Metrics | 6 months | ClickHouse | Daily |
| Prediction Results | 1 year | ClickHouse | Daily |
| Audit Logs | 3 years | MinIO S3 | Daily |
| System Logs | 90 days | Local Filesystem | Daily |
| Performance Metrics | 1 year | Prometheus | Continuous |

## Data Classification

```mermaid
pie title Data Classification
    "Public Market Data" : 60
    "Internal Analytics" : 25
    "Sensitive Financial Data" : 10
    "Personal Information" : 5
```

## Compliance & Governance

### Data Governance Framework

```mermaid
mindmap
  root((Data Governance))
    Data Quality
      Validation Rules
      Quality Metrics
      Data Profiling
    Data Security
      Encryption at Rest
      Access Controls
      Audit Logging
    Data Lineage
      Source Tracking
      Transformation History
      Usage Analytics
    Compliance
      Regulatory Requirements
      Data Retention
      Privacy Controls
```

---

## Implementation Details

### Data Flow Implementation

```python
# Example data flow implementation
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

class DataFlowTracker:
    """Tracks data flow through the system."""

    def __init__(self):
        self.flow_steps = []
        self.metadata = {}

    def track_step(self, step_name: str, input_data: Any, output_data: Any, metadata: Dict[str, Any]):
        """Track a data processing step."""
        step_info = {
            'step_name': step_name,
            'timestamp': pd.Timestamp.now(),
            'input_shape': getattr(input_data, 'shape', None) if hasattr(input_data, 'shape') else None,
            'output_shape': getattr(output_data, 'shape', None) if hasattr(output_data, 'shape') else None,
            'metadata': metadata
        }
        self.flow_steps.append(step_info)

    def get_lineage_report(self) -> Dict[str, Any]:
        """Generate data lineage report."""
        return {
            'total_steps': len(self.flow_steps),
            'steps': self.flow_steps,
            'start_time': self.flow_steps[0]['timestamp'] if self.flow_steps else None,
            'end_time': self.flow_steps[-1]['timestamp'] if self.flow_steps else None,
            'duration': (self.flow_steps[-1]['timestamp'] - self.flow_steps[0]['timestamp']) if len(self.flow_steps) > 1 else None
        }
```

### Quality Gate Implementation

```python
# Example quality gate implementation
from data_quality.great_expectations_suite import DataQualityValidator

class QualityGate:
    """Implements data quality gates in the pipeline."""

    def __init__(self, quality_thresholds: Dict[str, float]):
        self.validator = DataQualityValidator()
        self.thresholds = quality_thresholds

    def check_quality(self, data: pd.DataFrame, stage: str) -> Dict[str, Any]:
        """Check data quality for a specific pipeline stage."""
        result = self.validator.validate_dataset(data, f"{stage}_validation")

        # Apply stage-specific thresholds
        gate_passed = (
            result.passed and
            result.details.get('completeness', {}).get('overall_missing_ratio', 1.0) <= self.thresholds.get('missing_ratio', 0.05) and
            result.details.get('uniqueness', {}).get('duplicate_ratio', 1.0) <= self.thresholds.get('duplicate_ratio', 0.01)
        )

        return {
            'gate_passed': gate_passed,
            'validation_result': result,
            'stage': stage,
            'recommendations': result.recommendations
        }
```

---

## Contact & Support

- **Data Architecture**: [Data Engineering Team]
- **Data Quality**: [Data Quality Team]
- **ML Engineering**: [ML Engineering Team]
- **DevOps/SRE**: [DevOps Team]

---

*Last Updated: 2024-01-XX*
*Version: 2.0.0*
*Document Owner: Data Engineering Team*
