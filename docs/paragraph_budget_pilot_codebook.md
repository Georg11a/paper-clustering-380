# Paragraph-Budget Pilot Codebook

Version: 2.0

Locked: 2026-07-27

Applies from: DT-04 onward, with earlier calibration judgments recoded if needed.

## Relevance to the focal design-theory contribution

The relevance score measures how directly a passage contributes to the focal
design theory. It does **not** measure whether the passage is indispensable;
that is recorded separately as `critical_evidence_yes_no`.

### Score 2 — direct theory evidence

The passage directly defines, differentiates, develops, operationalizes,
instantiates, evaluates, or limits the focal design theory.

Domain-specific implementation content receives a 2 when it is a core
operationalization or instantiation of the claimed theory. It is not reduced to
a 1 merely because it is specific to information services, information-security
labs, sonification, service ecosystems, or another application domain.

### Score 1 — supporting context or optional illustration

The passage helps interpret the focal design theory but does not itself
constitute a central theoretical component, operationalization, instantiation,
evaluation, or boundary. Examples include replaceable implementation details,
optional illustrations, and background that supports but does not directly
advance the theory.

### Score 0 — outside the focal contribution

The passage neither develops the focal design theory nor provides meaningful
support for interpreting it. Examples include publisher notices, institutional
metadata, navigation text, and unrelated rhetorical material.

## Decision sequence

1. Does the passage make a direct theory claim, specify a theory component, or
   operationalize/instantiate/evaluate the theory's central contribution?
   If yes, assign 2.
2. Otherwise, does it provide useful but replaceable context, illustration, or
   implementation support? If yes, assign 1.
3. Otherwise, assign 0.

The paper abstract may be used to identify the paper's claimed deliverables,
but an abstract promise does not automatically make every passage about that
deliverable a 2. The passage must still provide direct evidence for the focal
design-theory contribution.

## Distinction from critical evidence

- `relevance = 2, critical = Yes`: direct and indispensable/uniquely important.
- `relevance = 2, critical = No`: direct but redundant, replaceable, or not
  individually necessary.
- `relevance = 1, critical = Yes`: possible but uncommon; supporting context is
  necessary to interpret otherwise ambiguous theory evidence.

## Locked DT-04 anchors

Paper: *Process design theory for digital information services*

All four passages below are score 2 because they directly specify or instantiate
the paper's focal process design theory:

- `R-A1BC40`: distinguishes product-oriented and process-oriented design theory.
- `R-2FBFCC`: identifies the three design aspects.
- `R-26B2C5`: maps models/techniques to the theory's aspects or design layers.
- `R-8C9CA2`: describes design scenarios that instantiate the proposed process
  design theory.

DT-04 does not provide reliable score-0 or score-1 anchors under this locked
interpretation. Real lower-score anchors must be added only after they are
independently identified during DT-05 calibration; they must not be invented to
force all levels of the scale to appear.
