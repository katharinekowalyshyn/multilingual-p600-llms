# Experimental Stimulus Sets: P600 & Garden Path Sentences

This document describes the datasets of example sentences collected for testing both **P600-eliciting grammatical violations** and **garden-path syntactic ambiguities**.  
These can be used in probing LLM behavior.

---

## 📑 Dataset 1: P600-Eliciting Sentences

**Description:**  
These 25 sentences contain **grammatical violations** (morphosyntactic, agreement, phrase-structure) that have been shown to reliably elicit a **P600 ERP component** in human participants. They are outright ungrammatical or contain feature clashes that force syntactic repair.

**Subcategories:**
- **Subject–Verb Agreement** (First 5 examples)  
  e.g., *“The child throw the toy.”*  
- **Tense Agreement** (Second 5 examples)  
  e.g., *“Yesterday he walks to the store.”*  
- **Number Agreement** (Third 5 examples)  
  e.g., *“The team of scientists were conducting an experiment.”*  
- **Gender / Case Agreement** (Fourth 5 examples, in gender-marked languages such as Spanish or German)  
  e.g., *“La niña bonito juega en el jardín.”*  
- **Phrase Structure / Word Order Violations** (5 examples)  
  e.g., *“The hearty meal was devouring the kids.”*  

**Intended Use:**  
- Tests sensitivity to **overt ungrammaticality**.  
- Probes whether participants (or models) show clear signs of reanalysis/repair.  
- Baseline for “hard” syntactic violations.

---

## 📑 Dataset 2: Garden-Path Sentences

**Description:**  
These 25 sentences are **temporarily ambiguous** and lead the parser toward an initially plausible but ultimately incorrect interpretation. Disambiguation requires syntactic reanalysis, which typically elicits a **P600** in humans.

**Examples of structures included:**
- **Reduced relative clauses**  
  e.g., *“The horse raced past the barn fell.”*  
- **Object–subject ambiguities**  
  e.g., *“The dog chased the cat meowed.”*  
- **Attachment ambiguities**  
  e.g., *“The witness said the defendant lied was nervous.”*  
- **Center embeddings with misleading cues**  
  e.g., *“The author the critics praised won the award.”*  

**Intended Use:**  
- Tests incremental parsing strategies (serial vs. parallel).  
- Distinguishes between simple grammaticality judgment and real-time **reanalysis mechanisms**.  
- Allows comparison across populations (monolingual vs. bilingual vs. schizophrenia) or models (small vs. large LLMs).

---

## ✅ Summary

- **P600 Dataset (25 items):** Grammatical violations → strong elicitors of the P600 as **syntactic anomaly repair signals**.  
- **Garden Path Dataset (25 items):** Temporary ambiguities → P600 elicited through **syntactic reanalysis and disambiguation**.  

**Together, these sets form a balanced experimental paradigm** for comparing human ERP data with LLM surprisal, entropy, or hidden activation patterns.