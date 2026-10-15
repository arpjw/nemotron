# BANKING77 attribution

BANKING77 was released by PolyAI and described by Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić in *Efficient Intent Detection with Dual Sentence Encoders*, Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, 2020, pp. 38–45.

- Publisher: https://github.com/PolyAI-LDN/task-specific-datasets
- Paper: https://aclanthology.org/2020.nlp4convai-1.5/
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Immutable source commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`.

Changes: official training text was grouped using Unicode normalization, case folding and punctuation/whitespace normalization. Duplicate groups were represented once; one conflicting-label group was quarantined. A deterministic development split and capped training subset were created, with instruction prompts and provenance added. The official test was released once after development selection; its rows and labels were not modified, and its references are withheld from this source repository. These changes are ours; the publisher does not endorse this project. Preserve this attribution, license and change notice when redistributing the dataset or its adaptations. The full license is in `DATA_LICENSE`.
