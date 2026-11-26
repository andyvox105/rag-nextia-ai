from dataclasses import dataclass

AUDIENCES = ("staff", "public")

@dataclass(frozen=True)
class Filters:
    as_of: str | None = None  # YYYY-MM-DD, or None for no date filter
    audience: str = "staff"

    def __post_init__(self):
        if self.audience not in AUDIENCES:
            raise ValueError(f"audience must be one of {AUDIENCES}, not {self.audience!r}")

    def allows(self, chunk) -> bool:
        if self.audience == "public" and (getattr(chunk, "access", "") or "staff") != "public":
            return False
        if self.as_of:
            if chunk.effective_from and chunk.effective_from > self.as_of:
                return False
            if chunk.effective_to and self.as_of > chunk.effective_to:
                return False
        return True