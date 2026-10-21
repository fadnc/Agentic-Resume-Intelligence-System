"""Render validated resume data as a Word document or editable LaTeX source."""
from io import BytesIO


def _items(value):
    return value if isinstance(value, list) else []


def build_docx(resume: dict) -> bytes:
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)

    name = str(resume.get("name") or "Resume")
    doc.add_heading(name, 0)
    contact = " | ".join(str(resume.get(k, "")).strip() for k in ("email", "phone", "location", "linkedin", "website") if resume.get(k))
    if contact:
        doc.add_paragraph(contact)

    def section_heading(title):
        doc.add_heading(title, level=1)

    summary = resume.get("summary")
    if summary:
        section_heading("Professional Summary")
        doc.add_paragraph(str(summary))

    skills = _items(resume.get("skills"))
    if skills:
        section_heading("Skills")
        doc.add_paragraph(" • ".join(map(str, skills)))

    experience = _items(resume.get("experience"))
    if experience:
        section_heading("Experience")
        for job in experience:
            heading = " — ".join(str(job.get(k, "")).strip() for k in ("title", "company") if job.get(k))
            if job.get("duration"):
                heading = f"{heading} | {job['duration']}"
            doc.add_heading(heading or "Experience", level=2)
            for bullet in _items(job.get("bullets")):
                doc.add_paragraph(str(bullet), style="List Bullet")

    education = _items(resume.get("education"))
    if education:
        section_heading("Education")
        for item in education:
            text = " — ".join(str(item.get(k, "")).strip() for k in ("degree", "institution", "year") if item.get(k))
            if item.get("gpa"):
                text += f" | GPA: {item['gpa']}"
            doc.add_paragraph(text, style="List Bullet")

    certifications = _items(resume.get("certifications"))
    if certifications:
        section_heading("Certifications")
        for item in certifications:
            doc.add_paragraph(str(item), style="List Bullet")

    output = BytesIO()
    doc.save(output)
    return output.getvalue()


def _tex(value) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value or ""))


def build_latex(resume: dict) -> str:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{textcomp}",
        r"\usepackage{enumitem}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlist[itemize]{nosep,leftmargin=*}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\LARGE \textbf{" + _tex(resume.get("name") or "Resume") + r"}}\\[4pt]",
    ]
    contact = [resume.get(k) for k in ("email", "phone", "location", "linkedin", "website") if resume.get(k)]
    if contact:
        lines.append(r" \textbar{} ".join(_tex(item) for item in contact) + r"\\")
    lines.extend([r"\end{center}", r"\vspace{-6pt}"])

    def heading(title):
        lines.extend([r"\section*{" + _tex(title) + "}"])

    if resume.get("summary"):
        heading("Professional Summary")
        lines.append(_tex(resume["summary"]))
    if _items(resume.get("skills")):
        heading("Skills")
        lines.append(r" \textbar{} ".join(_tex(skill) for skill in resume["skills"]))
    if _items(resume.get("experience")):
        heading("Experience")
        for job in resume["experience"]:
            title = " -- ".join(str(job.get(k, "")).strip() for k in ("title", "company") if job.get(k))
            if job.get("duration"):
                title += " | " + str(job["duration"])
            lines.append(r"\textbf{" + _tex(title) + "}\\")
            bullets = _items(job.get("bullets"))
            if bullets:
                lines.append(r"\begin{itemize}")
                lines.extend(r"\item " + _tex(item) for item in bullets)
                lines.append(r"\end{itemize}")
    if _items(resume.get("education")):
        heading("Education")
        lines.append(r"\begin{itemize}")
        for item in resume["education"]:
            value = " -- ".join(str(item.get(k, "")).strip() for k in ("degree", "institution", "year") if item.get(k))
            if item.get("gpa"):
                value += " | GPA: " + str(item["gpa"])
            lines.append(r"\item " + _tex(value))
        lines.append(r"\end{itemize}")
    if _items(resume.get("certifications")):
        heading("Certifications")
        lines.append(r"\begin{itemize}")
        lines.extend(r"\item " + _tex(item) for item in resume["certifications"])
        lines.append(r"\end{itemize}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"
