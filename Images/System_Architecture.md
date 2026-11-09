# Module architecture (editable Mermaid)

The figure below is the simplified overview used by the root README. See the [integration README](../Integration/README.md#data-flow) for timing and processing details.

```mermaid
flowchart LR
    CV["External CV<br/>ID, XYZ"] --> L["Live input<br/>ObservationSession"]
    CSV["Recorded data<br/>CSV"] --> T["Test loader<br/>TrackingSession"]
    L --> P["Prediction<br/>prediction_method.py"]
    T --> P
    P --> R["Session estimates<br/>and diagnostics"]
    R --> G["GUI supervision"]
    R --> O["Outputs / exports"]
    R -.-> V["ROV interface<br/>placeholders only"]
```

**Read:** Both Test and Run use one supported prediction method, not two algorithms. Live observations are individually timestamped on receipt and update only their own ID. Recorded frames have known timing. ROV actuation is not implemented.
