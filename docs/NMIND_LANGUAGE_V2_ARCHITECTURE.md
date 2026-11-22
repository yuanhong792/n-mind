# Language V2 progressive learning architecture

Data provenance and license gate → offline cleaning/deduplication → train/dev/sealed manifests → reversible Tokenizer V2 → random-initialized N-Cell/Router/Workspace → prediction and action candidates → independent LanguageWorld oracle → Memory V2 episodic experience → curriculum/planning queue → explicitly authorized candidate training.

ExternalKnowledge stays retrieval-only. EpisodicExperience stays Memory V2. NeuralCandidateTraining accepts only approved, validated records under a separate positive optimizer budget. A response is always attributed as neural, memory, retrieval, tool, rule or abstain.
