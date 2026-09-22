# Graph Report - specialist-factory  (2026-09-22)

## Corpus Check
- Corpus is ~4,933 words - fits in a single context window. You may not need a graph.

## Summary
- 191 nodes · 462 edges · 14 communities (9 shown, 5 thin omitted)
- Extraction: 84% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 71 edges (avg confidence: 0.93)
- Token cost: 59,493 input · 0 output

## Community Hubs (Navigation)
- CLI & Training Pipeline
- Architecture Docs & Demo Profiles
- Student Encoders & Heads
- Label Aggregation & Schemas
- Teacher Factory & Mock Teachers
- Demo Teacher & Aggregation Config
- ModernBERT Laya Student
- Teacher Signal Cache
- Task Schema & Teacher Protocol
- HuggingFace Teacher Adapters
- Training Dataset
- OpenRouter Vision Teacher
- Package Init
- Package Metadata

## God Nodes (most connected - your core abstractions)
1. `Sample` - 34 edges
2. `TeacherSignal` - 25 edges
3. `TaskDefinition` - 20 edges
4. `TeacherConfig` - 14 edges
5. `AppConfig` - 14 edges
6. `TeacherCache` - 12 edges
7. `annotate()` - 11 edges
8. `_signal()` - 11 edges
9. `SpecialistModule` - 11 edges
10. `SignalType` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_laya_style_head_scores_class_markers()` --uses--> `LayaStyleDecisionHead`  [INFERRED]
  tests/test_laya_head.py → src/specialist_factory/models/modernbert_laya.py
- `test_cache_roundtrip()` --uses--> `SignalType`  [INFERRED]
  tests/test_cache.py → src/specialist_factory/schemas.py
- `test_cache_roundtrip()` --uses--> `ClassDefinition`  [INFERRED]
  tests/test_cache.py → src/specialist_factory/schemas.py
- `task()` --uses--> `TaskDefinition`  [INFERRED]
  tests/test_aggregation.py → src/specialist_factory/schemas.py
- `test_cache_roundtrip()` --uses--> `TaskDefinition`  [INFERRED]
  tests/test_cache.py → src/specialist_factory/schemas.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Teacher-signal to student distillation flow** — readme_teachers, readme_provenance, readme_aggregation, readme_students, readme_lightning_distillation, readme_evaluation [EXTRACTED 1.00]
- **CPU-only reproducible demo profile (mock teachers, seed 7)** — readme_mock_teachers, config_text_demo_mock_text_a, config_vision_demo_mock_vision_a, config_multimodal_demo_mock_multimodal_a, config_text_demo_training, config_vision_demo_training, config_multimodal_demo_training [INFERRED 0.95]
- **Swappable production teacher adapters** — readme_teachers, config_text_demo_huggingface_text, config_vision_demo_huggingface_vision, config_multimodal_demo_openrouter_vision, config_modernbert_jev_demo_jev_teacher [INFERRED 0.85]

## Communities (14 total, 5 thin omitted)

### Community 0 - "CLI & Training Pipeline"
Cohesion: 0.15
Nodes (23): command, annotate(), load_annotations(), annotate(), evaluate(), generate(), predict(), train() (+15 more)

### Community 1 - "Architecture Docs & Demo Profiles"
Cohesion: 0.11
Nodes (30): Jev Demo Aggregation (thresholds disabled), ecommerce_intent_jev_distillation Task, Synthetic Human Label Generator (rate 1.0), jev Teacher (jev-1.13, systemone-noul-v1), laya_style_marker_decision Head, modernbert Student Encoder (answerdotai/ModernBERT-base, frozen), data/modernbert_jev_teacher_cache.sqlite, Jev Demo Training Settings (20 epochs, lr 1e-4, batch 4) (+22 more)

### Community 2 - "Student Encoders & Heads"
Cohesion: 0.18
Nodes (9): FeatureEncoder, image_features(), MLPHead, Tensor, text_features(), TinyTextEncoder, TinyVisionEncoder, StudentConfig (+1 more)

### Community 3 - "Label Aggregation & Schemas"
Cohesion: 0.27
Nodes (14): BaseModel, Enum, aggregate(), validate_signal(), AggregatedLabel, AggregationConfig, ClassDefinition, Modality (+6 more)

### Community 4 - "Teacher Factory & Mock Teachers"
Cohesion: 0.20
Nodes (9): Any, TeacherConfig, build_teacher(), JevTeacher, MockTeacher, Deterministic offline teacher used for reproducible demo flows., TypeSafe Jev NOUL adapter; every class gets the same evidence and a binary…, _signal() (+1 more)

### Community 5 - "Demo Teacher & Aggregation Config"
Cohesion: 0.13
Nodes (16): Multimodal Demo Aggregation (weighted_mean, 0.5/0.55), mock_multimodal_a Teacher, mock_multimodal_b Teacher (weight 0.9), vlm Teacher (openrouter_vision, commented), Multimodal Demo Training Settings, Text Demo Aggregation (weighted_mean, 0.5/0.55), hf_text Teacher (facebook/bart-large-mnli, commented), mock_text_a Teacher (+8 more)

### Community 6 - "ModernBERT Laya Student"
Cohesion: 0.22
Nodes (6): device, LayaStyleDecisionHead, ModernBertLayaStudent, Tensor, Laya-inspired typed decision head: type embedding → head transformer → marker…, test_laya_style_head_scores_class_markers()

### Community 7 - "Teacher Signal Cache"
Cohesion: 0.24
Nodes (6): field_validator, Path, TeacherCache, TeacherSignal, Path, test_cache_roundtrip()

### Community 8 - "Task Schema & Teacher Protocol"
Cohesion: 0.27
Nodes (5): Protocol, stable_hash(), Sample, TaskDefinition, Teacher

## Ambiguous Edges - Review These
- `self_reported_probability Signal Type` → `jev Teacher (jev-1.13, systemone-noul-v1)`  [AMBIGUOUS]
  config/modernbert_jev_demo.yaml · relation: conceptually_related_to
- `ecommerce_intent Text Task` → `laya_style_marker_decision Head`  [AMBIGUOUS]
  config/modernbert_jev_demo.yaml · relation: conceptually_related_to

## Knowledge Gaps
- **9 isolated node(s):** `specialist-factory`, `Modality-neutral JSONL Sample Schema`, `mock_text_b Teacher (weight 0.9)`, `data/text_teacher_cache.sqlite`, `mock_vision_b Teacher (weight 0.9)` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `self_reported_probability Signal Type` and `jev Teacher (jev-1.13, systemone-noul-v1)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `ecommerce_intent Text Task` and `laya_style_marker_decision Head`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Sample` connect `Task Schema & Teacher Protocol` to `CLI & Training Pipeline`, `Student Encoders & Heads`, `Label Aggregation & Schemas`, `Teacher Factory & Mock Teachers`, `ModernBERT Laya Student`, `Teacher Signal Cache`, `HuggingFace Teacher Adapters`, `Training Dataset`, `OpenRouter Vision Teacher`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `TeacherSignal` connect `Teacher Signal Cache` to `CLI & Training Pipeline`, `Label Aggregation & Schemas`, `Teacher Factory & Mock Teachers`, `Task Schema & Teacher Protocol`, `HuggingFace Teacher Adapters`, `OpenRouter Vision Teacher`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `TeacherConfig` connect `Teacher Factory & Mock Teachers` to `Label Aggregation & Schemas`, `Teacher Signal Cache`, `Task Schema & Teacher Protocol`, `HuggingFace Teacher Adapters`, `OpenRouter Vision Teacher`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Sample` (e.g. with `annotate()` and `TeacherCache`) actually correct?**
  _`Sample` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `TeacherSignal` (e.g. with `aggregate()` and `validate_signal()`) actually correct?**
  _`TeacherSignal` has 8 INFERRED edges - model-reasoned connections that need verification._