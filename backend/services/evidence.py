"""Build verifiable requirement evidence from text extracted from a resume."""
import re


def build_requirement_evidence(resume_text: str, parsed_jd: dict) -> list[dict]:
    source = re.sub(r"\s+", " ", resume_text or "").strip()
    folded = source.casefold()
    results = []
    for group, requirements in (("must_have", "Required"), ("nice_to_have", "Preferred")):
        for requirement in parsed_jd.get(group, []) or []:
            label = str(requirement).strip()
            if not label:
                continue
            match = re.search(re.escape(label), source, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 100)
                end = min(len(source), match.end() + 140)
                excerpt = source[start:end]
                results.append({
                    "requirement": label,
                    "category": group,
                    "priority": requirements,
                    "status": "evidence_found",
                    "confidence": 0.95,
                    "evidence": excerpt,
                })
                continue

            # Short technical terms often appear in a longer JD phrase. Only mark
            # them as evidence when every meaningful token occurs in the source.
            terms = [t for t in re.findall(r"[\w+#.\-]+", label.casefold()) if len(t) > 1]
            present = [t for t in terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", folded)]
            if terms and len(present) == len(terms):
                results.append({
                    "requirement": label,
                    "category": group,
                    "priority": requirements,
                    "status": "partial_evidence",
                    "confidence": 0.65,
                    "evidence": "Resume mentions the requirement's terms, but not as the same phrase.",
                })
            else:
                results.append({
                    "requirement": label,
                    "category": group,
                    "priority": requirements,
                    "status": "not_found",
                    "confidence": 0.9,
                    "evidence": "No matching phrase was found in the extracted resume text.",
                })
    return results
