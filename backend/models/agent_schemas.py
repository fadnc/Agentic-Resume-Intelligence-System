"""Bounded output schemas for model generated agent data."""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ExperienceItem(AgentOutput):
    title: str = Field(default="", max_length=200)
    company: str = Field(default="", max_length=200)
    duration: str = Field(default="", max_length=100)
    bullets: list[str] = Field(default_factory=list, max_length=30)


class EducationItem(AgentOutput):
    degree: str = Field(default="", max_length=200)
    institution: str = Field(default="", max_length=200)
    year: Optional[str] = Field(default=None, max_length=30)
    gpa: Optional[str] = Field(default=None, max_length=40)


class ProjectItem(AgentOutput):
    name: str = Field(default="", max_length=200)
    technologies: list[str] = Field(default_factory=list, max_length=50)
    duration: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = Field(default=None, max_length=2000)
    bullets: list[str] = Field(default_factory=list, max_length=20)
    url: Optional[str] = Field(default=None, max_length=500)


class ResumeStyle(AgentOutput):
    accent_color: str = Field(default="222222", pattern=r"^[0-9A-Fa-f]{6}$")
    secondary_color: str = Field(default="414141", pattern=r"^[0-9A-Fa-f]{6}$")


class ParsedResume(AgentOutput):
    name: str = Field(default="", max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)
    phone: Optional[str] = Field(default=None, max_length=100)
    location: Optional[str] = Field(default=None, max_length=200)
    linkedin: Optional[str] = Field(default=None, max_length=400)
    website: Optional[str] = Field(default=None, max_length=400)
    skills: list[str] = Field(default_factory=list, max_length=100)
    experience: list[ExperienceItem] = Field(default_factory=list, max_length=30)
    projects: list[ProjectItem] = Field(default_factory=list, max_length=30)
    education: list[EducationItem] = Field(default_factory=list, max_length=20)
    summary: Optional[str] = Field(default=None, max_length=3000)
    certifications: list[str] = Field(default_factory=list, max_length=50)
    style: ResumeStyle = Field(default_factory=ResumeStyle)


class ParsedJob(AgentOutput):
    title: str = Field(default="Unknown role", max_length=200)
    company: Optional[str] = Field(default=None, max_length=200)
    seniority: str = Field(default="unknown", max_length=40)
    domain: str = Field(default="unknown", max_length=200)
    must_have: list[str] = Field(default_factory=list, max_length=100)
    nice_to_have: list[str] = Field(default_factory=list, max_length=100)
    keywords: list[str] = Field(default_factory=list, max_length=150)
    responsibilities: list[str] = Field(default_factory=list, max_length=100)


class MissingSkill(AgentOutput):
    skill: str = Field(max_length=200)
    importance: str = Field(default="important", max_length=40)
    learnable_in_weeks: Optional[int] = Field(default=None, ge=0, le=520)


class GapAnalysis(AgentOutput):
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list, max_length=100)
    missing_skills: list[MissingSkill] = Field(default_factory=list, max_length=100)
    gap_summary: str = Field(default="", max_length=3000)
    quick_wins: list[str] = Field(default_factory=list, max_length=30)


class RewrittenBullet(AgentOutput):
    section: str = Field(default="", max_length=300)
    original: str = Field(max_length=2000)
    rewritten: str = Field(max_length=2000)
    reason: str = Field(default="", max_length=1000)


class InterviewQuestion(AgentOutput):
    question: str = Field(max_length=1000)
    answer_outline: str = Field(default="", max_length=3000)
    evidence_to_use: str = Field(default="", max_length=1000)


class RewriterOutput(AgentOutput):
    rewritten_bullets: list[RewrittenBullet] = Field(default_factory=list, max_length=50)
    tailored_summary: str = Field(default="", max_length=3000)
    improvement_tips: list[str] = Field(default_factory=list, max_length=30)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list, max_length=12)


class InterviewFeedback(AgentOutput):
    feedback: str = Field(max_length=3000)
    strengths: list[str] = Field(default_factory=list, max_length=8)
    next_try: str = Field(default="", max_length=1500)
